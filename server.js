#!/usr/bin/env node
/**
 * Kanban + KB API server (SQLite-backed)
 * 
 * Replaces the JSON-loading server with SQLite queries.
 * - POST /api/ai-search — natural language KB search via FTS5 + LLM
 * - POST /api/search    — direct SQLite search (no LLM, fast)
 * - GET  /api/stats     — KB statistics
 * - GET  /api/cards     — return kanban data.json
 * - GET  /api/health    — health check
 * 
 * Runs on port 3001, behind Tailscale.
 */

const express = require('express');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

// Import the KB database module
const kbDb = require(path.join(process.env.HOME, 'clawd/scripts/kb-db'));

const app = express();
const PORT = process.env.PORT || 3001;
const DATA_PATH = path.join(__dirname, 'data.json');
const API_TOKEN = 'ufwbX6Zztw3hyWKwEVUrU1cz';

// Get OpenRouter key for chat completions
function getOpenRouterKey() {
  try {
    const config = JSON.parse(fs.readFileSync(
      path.join(process.env.HOME, '.openclaw/openclaw.json'), 'utf8'
    ));
    const profiles = config.auth?.profiles || {};
    for (const [key, val] of Object.entries(profiles)) {
      if (val.provider === 'openrouter' && val.apiKey) return val.apiKey;
    }
    const credsPath = path.join(process.env.HOME, '.openclaw/credentials.json');
    if (fs.existsSync(credsPath)) {
      const creds = JSON.parse(fs.readFileSync(credsPath, 'utf8'));
      return creds['openrouter:default']?.apiKey || null;
    }
    return null;
  } catch { return null; }
}

// Get OpenAI key (fallback)
function getOpenAIKey() {
  try {
    const config = JSON.parse(fs.readFileSync(
      path.join(process.env.HOME, '.openclaw/openclaw.json'), 'utf8'
    ));
    return config.agents?.defaults?.memorySearch?.remote?.apiKey || null;
  } catch { return null; }
}

app.use(cors());
app.use(express.json());

// Auth middleware
function requireAuth(req, res, next) {
  const auth = req.headers.authorization;
  if (!auth || auth !== `Bearer ${API_TOKEN}`) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  next();
}

// Rate limiting
const rateLimits = new Map();
function rateLimit(maxPerMinute) {
  return (req, res, next) => {
    const ip = req.ip || req.connection.remoteAddress;
    const now = Date.now();
    const window = 60000;
    if (!rateLimits.has(ip)) rateLimits.set(ip, []);
    const timestamps = rateLimits.get(ip).filter(t => now - t < window);
    if (timestamps.length >= maxPerMinute) {
      return res.status(429).json({ error: 'Too many requests. Try again in a minute.' });
    }
    timestamps.push(now);
    rateLimits.set(ip, timestamps);
    next();
  };
}

// --- Endpoints ---

// AI Search: FTS5 + LLM summary
app.post('/api/ai-search', requireAuth, rateLimit(10), async (req, res) => {
  try {
    const { query, category, dateFrom, dateTo } = req.body;
    if (!query) return res.status(400).json({ error: 'Query required' });

    // Use FTS5 for initial search
    let results = kbDb.search({
      query,
      category: category || undefined,
      since: dateFrom || undefined,
      until: dateTo || undefined,
      limit: 30,
    });

    // If FTS returns nothing, fall back to category/date browsing
    if (results.length === 0 && (category || dateFrom)) {
      results = kbDb.search({
        category: category || undefined,
        since: dateFrom || undefined,
        until: dateTo || undefined,
        orderBy: 'quality',
        limit: 30,
      });
    }

    // Build context for LLM
    const context = results.slice(0, 30).map((e, i) =>
      `[${i + 1}] ${e.title}\n${e.summary || 'No summary'}\nSource: ${e.source || 'unknown'} | ${e.date || 'unknown date'}\nURL: ${e.url || 'N/A'}`
    ).join('\n\n');

    // Call LLM for summary
    let apiKey = getOpenRouterKey();
    let apiUrl = 'https://openrouter.ai/api/v1/chat/completions';
    let model = 'openai/gpt-4.1-mini';

    if (!apiKey) {
      apiKey = getOpenAIKey();
      apiUrl = 'https://api.openai.com/v1/chat/completions';
      model = 'gpt-4.1-mini';
    }

    let summary = `Found ${results.length} matching entries.`;

    if (apiKey && results.length > 0) {
      try {
        const llmResponse = await fetch(apiUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
          },
          body: JSON.stringify({
            model,
            messages: [
              {
                role: 'system',
                content: 'You are a GenAI knowledge base search assistant. Summarise what the KB knows about the user\'s query. Cite entry numbers [1], [2] etc. Be concise and direct. Format with markdown.'
              },
              {
                role: 'user',
                content: `Question: ${query}\n\nKnowledge Base entries:\n${context || 'No matching entries found.'}`
              }
            ],
            max_tokens: 500,
            temperature: 0.3
          })
        });

        const llmData = await llmResponse.json();
        summary = llmData.choices?.[0]?.message?.content || summary;
      } catch (err) {
        console.error('LLM summary failed:', err.message);
      }
    }

    res.json({
      summary,
      results: results.slice(0, 10),
      totalMatches: results.length
    });

  } catch (err) {
    console.error('AI search error:', err);
    res.status(500).json({ error: err.message });
  }
});

// Direct search: fast, no LLM
app.post('/api/search', requireAuth, rateLimit(30), (req, res) => {
  try {
    const { query, category, sourceType, freshnessTag, minQuality, since, until, freshOnly, orderBy, order, limit, offset } = req.body;

    const results = kbDb.search({
      query, category, sourceType, freshnessTag,
      minQuality, since, until, freshOnly,
      orderBy: orderBy || 'date', order, 
      limit: limit || 20, offset: offset || 0
    });

    const total = kbDb.count({
      category, sourceType, freshnessTag, minQuality, freshOnly, since
    });

    res.json({ results, total, returned: results.length });
  } catch (err) {
    console.error('Search error:', err);
    res.status(500).json({ error: err.message });
  }
});

// KB Statistics
app.get('/api/stats', requireAuth, (req, res) => {
  try {
    res.json(kbDb.stats());
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Kanban cards
app.get('/api/cards', (req, res) => {
  try {
    const data = JSON.parse(fs.readFileSync(DATA_PATH, 'utf8'));
    res.json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Health check
app.get('/api/health', (req, res) => {
  try {
    const s = kbDb.stats();
    res.json({
      status: 'ok',
      backend: 'sqlite',
      entries: s.total,
      fresh: s.fresh,
      avgQuality: s.avgQuality,
      dateRange: s.dateRange,
      timestamp: new Date().toISOString()
    });
  } catch (err) {
    res.json({
      status: 'degraded',
      error: err.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received, closing DB...');
  kbDb.close();
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT received, closing DB...');
  kbDb.close();
  process.exit(0);
});

app.listen(PORT, '127.0.0.1', () => {
  const s = kbDb.stats();
  console.log(`Kanban API server (SQLite) running on port ${PORT}`);
  console.log(`KB: ${s.total} entries | ${s.fresh} fresh | avg quality: ${s.avgQuality}`);
  console.log(`Endpoints:`);
  console.log(`  POST /api/ai-search  — FTS5 search + LLM summary`);
  console.log(`  POST /api/search     — Direct SQLite search`);
  console.log(`  GET  /api/stats      — KB statistics`);
  console.log(`  GET  /api/cards      — Kanban board`);
  console.log(`  GET  /api/health     — Health check`);
});
