#!/usr/bin/env node
/**
 * Kanban + KB API server
 * - POST /api/ai-search — natural language KB search via OpenAI
 * - GET /api/cards — return kanban data.json
 * - PATCH /api/cards/:id — update card
 * - POST /api/cards — create card
 * 
 * Runs on port 3001, behind Tailscale.
 */

const express = require('express');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = 3001;
const KB_PATH = path.join(process.env.HOME, '.openclaw/workspace/genai-kb.json');
const DATA_PATH = path.join(__dirname, 'data.json');

// Get OpenAI key from OpenClaw config (for embeddings/chat)
function getOpenAIKey() {
  try {
    const config = JSON.parse(fs.readFileSync(
      path.join(process.env.HOME, '.openclaw/openclaw.json'), 'utf8'
    ));
    // Try memorySearch key first (it's the OpenAI key)
    return config.agents?.defaults?.memorySearch?.remote?.apiKey || null;
  } catch { return null; }
}

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
    // Fallback: read from credentials store
    const credsPath = path.join(process.env.HOME, '.openclaw/credentials.json');
    if (fs.existsSync(credsPath)) {
      const creds = JSON.parse(fs.readFileSync(credsPath, 'utf8'));
      return creds['openrouter:default']?.apiKey || null;
    }
    return null;
  } catch { return null; }
}

app.use(cors());
app.use(express.json());

// AI Search endpoint
app.post('/api/ai-search', async (req, res) => {
  try {
    const { query, category, dateFrom, dateTo } = req.body;
    if (!query) return res.status(400).json({ error: 'Query required' });

    // Load KB
    const kb = JSON.parse(fs.readFileSync(KB_PATH, 'utf8'));
    let entries = kb.entries || [];

    // Filter by category
    if (category) {
      entries = entries.filter(e => e.category === category);
    }

    // Filter by date range
    if (dateFrom) {
      entries = entries.filter(e => e.addedAt >= dateFrom);
    }
    if (dateTo) {
      entries = entries.filter(e => e.addedAt <= dateTo + 'T23:59:59Z');
    }

    // Simple keyword pre-filter to reduce context (top 50 most relevant)
    const queryTerms = query.toLowerCase().split(/\s+/);
    const scored = entries.map(e => {
      const text = `${e.title || ''} ${e.summary || ''} ${(e.tags || []).join(' ')}`.toLowerCase();
      let score = 0;
      for (const term of queryTerms) {
        if (text.includes(term)) score += 1;
        if ((e.title || '').toLowerCase().includes(term)) score += 2; // title boost
      }
      return { entry: e, score };
    }).filter(s => s.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 30);

    // Build context for LLM
    const context = scored.map((s, i) => 
      `[${i + 1}] ${s.entry.title}\n${s.entry.summary || 'No summary'}\nSource: ${s.entry.source || 'unknown'} | ${s.entry.addedAt || 'unknown date'}\nURL: ${s.entry.url || 'N/A'}`
    ).join('\n\n');

    // Call OpenRouter for summary (cheaper than OpenAI direct)
    let apiKey = getOpenRouterKey();
    let apiUrl = 'https://openrouter.ai/api/v1/chat/completions';
    let model = 'openai/gpt-4.1-mini';

    // Fallback to OpenAI if no OpenRouter key
    if (!apiKey) {
      apiKey = getOpenAIKey();
      apiUrl = 'https://api.openai.com/v1/chat/completions';
      model = 'gpt-4.1-mini';
    }

    if (!apiKey) {
      return res.status(500).json({ error: 'No API key configured' });
    }

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
            content: `You are a GenAI knowledge base assistant. Answer the user's question using ONLY the provided KB entries. Be concise, direct, and cite entry numbers [1], [2] etc. If no entries are relevant, say so. Format with markdown.`
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
    const summary = llmData.choices?.[0]?.message?.content || 'No summary generated.';

    // Return top results + summary
    const results = scored.slice(0, 10).map(s => s.entry);
    
    res.json({ summary, results, totalMatches: scored.length });

  } catch (err) {
    console.error('AI search error:', err);
    res.status(500).json({ error: err.message });
  }
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Kanban API server running on port ${PORT}`);
});
