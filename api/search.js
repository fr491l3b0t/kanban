/**
 * GenAI Knowledge Base Search API
 * 
 * A simple Express server that provides semantic search using OpenAI embeddings
 * and generates natural language summaries using Claude/GPT.
 * 
 * Usage:
 *   node api/search.js
 * 
 * Environment variables:
 *   - PORT: Server port (default: 3456)
 *   - OPENAI_API_KEY: OpenAI API key for embeddings
 *   - ANTHROPIC_API_KEY: Anthropic API key for summaries (optional, falls back to OpenAI)
 * 
 * API Endpoints:
 *   POST /search - Search with natural language query
 *     Body: { query: string, category?: string, dateFrom?: string, dateTo?: string, limit?: number }
 *     Response: { summary: string, results: Entry[], total: number }
 */

const express = require('express');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3456;

// API Keys
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const ANTHROPIC_API_KEY = process.env.ANTHROPIC_API_KEY;

// Cache
let kbData = null;
let embeddingsCache = null;
const EMBEDDINGS_CACHE_FILE = path.join(__dirname, '.embeddings-cache.json');

// Middleware
app.use(cors());
app.use(express.json());

// Load KB data
function loadKBData() {
  if (kbData) return kbData;
  
  const kbPath = path.join(__dirname, '..', 'genai-kb.json');
  const data = JSON.parse(fs.readFileSync(kbPath, 'utf8'));
  kbData = data;
  return data;
}

// Load or generate embeddings cache
async function getEmbeddingsCache() {
  if (embeddingsCache) return embeddingsCache;
  
  // Try to load from file
  if (fs.existsSync(EMBEDDINGS_CACHE_FILE)) {
    try {
      const cache = JSON.parse(fs.readFileSync(EMBEDDINGS_CACHE_FILE, 'utf8'));
      embeddingsCache = cache;
      console.log(`Loaded ${Object.keys(cache).length} embeddings from cache`);
      return cache;
    } catch (err) {
      console.error('Failed to load embeddings cache:', err.message);
    }
  }
  
  // Generate embeddings if needed
  if (!OPENAI_API_KEY) {
    console.log('No OPENAI_API_KEY set, skipping embeddings generation');
    return {};
  }
  
  return await generateEmbeddings();
}

// Generate embeddings for all entries
async function generateEmbeddings() {
  const data = loadKBData();
  const cache = {};
  
  console.log(`Generating embeddings for ${data.entries.length} entries...`);
  
  const batchSize = 100;
  for (let i = 0; i < data.entries.length; i += batchSize) {
    const batch = data.entries.slice(i, i + batchSize);
    const texts = batch.map(e => `${e.title}. ${e.summary || ''} ${e.category} ${(e.tags || []).join(' ')}`);
    
    try {
      const response = await fetch('https://api.openai.com/v1/embeddings', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${OPENAI_API_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: 'text-embedding-3-small',
          input: texts
        })
      });
      
      const result = await response.json();
      
      if (result.error) {
        throw new Error(result.error.message);
      }
      
      result.data.forEach((item, idx) => {
        cache[batch[idx].id] = item.embedding;
      });
      
      console.log(`  Batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(data.entries.length/batchSize)} complete`);
    } catch (err) {
      console.error(`Failed to generate embeddings for batch ${i}:`, err.message);
    }
  }
  
  // Save cache
  fs.writeFileSync(EMBEDDINGS_CACHE_FILE, JSON.stringify(cache));
  embeddingsCache = cache;
  console.log('Embeddings cache saved');
  
  return cache;
}

// Calculate cosine similarity
function cosineSimilarity(a, b) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Get embedding for query
async function getQueryEmbedding(query) {
  if (!OPENAI_API_KEY) return null;
  
  const response = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${OPENAI_API_KEY}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'text-embedding-3-small',
      input: query
    })
  });
  
  const result = await response.json();
  
  if (result.error) {
    throw new Error(result.error.message);
  }
  
  return result.data[0].embedding;
}

// Semantic search
async function semanticSearch(query, options = {}) {
  const { category, dateFrom, dateTo, limit = 10 } = options;
  const data = loadKBData();
  const cache = await getEmbeddingsCache();
  
  // If no embeddings available, fall back to text search
  if (!Object.keys(cache).length || !OPENAI_API_KEY) {
    return textSearch(query, options);
  }
  
  // Get query embedding
  const queryEmbedding = await getQueryEmbedding(query);
  
  // Calculate similarities
  let results = data.entries.map(entry => {
    const embedding = cache[entry.id];
    if (!embedding) return { entry, score: 0 };
    
    const similarity = cosineSimilarity(queryEmbedding, embedding);
    return { entry, score: similarity };
  });
  
  // Sort by score
  results.sort((a, b) => b.score - a.score);
  
  // Apply filters
  if (category && category !== 'all') {
    results = results.filter(r => r.entry.category === category);
  }
  if (dateFrom) {
    results = results.filter(r => !r.entry.date || r.entry.date >= dateFrom);
  }
  if (dateTo) {
    results = results.filter(r => !r.entry.date || r.entry.date <= dateTo);
  }
  
  return results.slice(0, limit);
}

// Fallback text search (no embeddings needed)
function textSearch(query, options = {}) {
  const { category, dateFrom, dateTo, limit = 10 } = options;
  const data = loadKBData();
  const queryLower = query.toLowerCase();
  const queryWords = queryLower.split(/\s+/).filter(w => w.length > 2);
  
  let results = data.entries.map(entry => {
    const text = `${entry.title} ${entry.summary || ''} ${entry.category} ${entry.source} ${(entry.tags || []).join(' ')}`.toLowerCase();
    
    // Simple scoring: count matching words
    let score = 0;
    queryWords.forEach(word => {
      if (text.includes(word)) score += 1;
    });
    
    // Boost exact matches
    if (entry.title.toLowerCase().includes(queryLower)) score += 5;
    if (entry.summary?.toLowerCase().includes(queryLower)) score += 3;
    
    return { entry, score: score / (queryWords.length + 5) };
  });
  
  // Sort by score
  results.sort((a, b) => b.score - a.score);
  
  // Apply filters
  if (category && category !== 'all') {
    results = results.filter(r => r.entry.category === category);
  }
  if (dateFrom) {
    results = results.filter(r => !r.entry.date || r.entry.date >= dateFrom);
  }
  if (dateTo) {
    results = results.filter(r => !r.entry.date || r.entry.date <= dateTo);
  }
  
  // Only return results with some relevance
  return results.filter(r => r.score > 0).slice(0, limit);
}

// Generate summary using Claude or GPT
async function generateSummary(query, results) {
  if (!ANTHROPIC_API_KEY && !OPENAI_API_KEY) {
    return null;
  }
  
  const context = results.map(r => {
    const e = r.entry;
    return `[${e.category}] ${e.title}\nBy: ${e.source} (${e.date})\n${e.summary || 'No description'}\nURL: ${e.url}`;
  }).join('\n\n---\n\n');
  
  const prompt = `The user asked: "${query}"

Here are the most relevant entries from their GenAI knowledge base:

${context}

Please provide a helpful, natural language summary of these findings. Focus on:
1. What the user is looking for based on their query
2. Key insights from the relevant entries
3. Any patterns or trends across the results
4. Actionable takeaways if applicable

Keep it concise (2-4 paragraphs) and conversational.`;

  // Try Claude first, fall back to GPT
  if (ANTHROPIC_API_KEY) {
    try {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'x-api-key': ANTHROPIC_API_KEY,
          'anthropic-version': '2023-06-01',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: 'claude-3-haiku-20240307',
          max_tokens: 1000,
          messages: [{ role: 'user', content: prompt }]
        })
      });
      
      const result = await response.json();
      
      if (result.error) {
        throw new Error(result.error.message);
      }
      
      return result.content[0].text;
    } catch (err) {
      console.error('Claude failed, falling back to GPT:', err.message);
    }
  }
  
  // Fallback to GPT
  if (OPENAI_API_KEY) {
    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${OPENAI_API_KEY}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          model: 'gpt-4o-mini',
          messages: [
            { role: 'system', content: 'You are a helpful assistant summarizing GenAI knowledge base search results.' },
            { role: 'user', content: prompt }
          ],
          max_tokens: 1000
        })
      });
      
      const result = await response.json();
      
      if (result.error) {
        throw new Error(result.error.message);
      }
      
      return result.choices[0].message.content;
    } catch (err) {
      console.error('GPT failed:', err.message);
    }
  }
  
  return null;
}

// Health check
app.get('/health', (req, res) => {
  const data = loadKBData();
  res.json({
    status: 'ok',
    entries: data.entries.length,
    categories: data.categories.length,
    embeddingsCached: embeddingsCache ? Object.keys(embeddingsCache).length : 0,
    aiAvailable: !!(OPENAI_API_KEY || ANTHROPIC_API_KEY)
  });
});

// Main search endpoint
app.post('/search', async (req, res) => {
  try {
    const { query, category, dateFrom, dateTo, limit = 10, includeSummary = true } = req.body;
    
    if (!query || !query.trim()) {
      return res.status(400).json({ error: 'Query is required' });
    }
    
    console.log(`Search: "${query}" (${category || 'all'})`);
    
    // Perform search
    const searchResults = await semanticSearch(query, { category, dateFrom, dateTo, limit });
    
    if (searchResults.length === 0) {
      return res.json({
        summary: "I couldn't find any entries matching your query. Try different keywords or broaden your search.",
        results: [],
        total: 0
      });
    }
    
    // Generate summary if requested
    let summary = null;
    if (includeSummary) {
      summary = await generateSummary(query, searchResults);
    }
    
    res.json({
      summary: summary || `Found ${searchResults.length} relevant entries.`,
      results: searchResults.map(r => ({
        ...r.entry,
        relevance: Math.round(r.score * 100)
      })),
      total: searchResults.length
    });
    
  } catch (err) {
    console.error('Search error:', err);
    res.status(500).json({ error: err.message });
  }
});

// Simple text search endpoint (no AI needed)
app.post('/text-search', (req, res) => {
  try {
    const { query, category, dateFrom, dateTo, limit = 20 } = req.body;
    
    if (!query || !query.trim()) {
      return res.status(400).json({ error: 'Query is required' });
    }
    
    const results = textSearch(query, { category, dateFrom, dateTo, limit });
    
    res.json({
      results: results.map(r => ({
        ...r.entry,
        relevance: Math.round(r.score * 100)
      })),
      total: results.length
    });
    
  } catch (err) {
    console.error('Text search error:', err);
    res.status(500).json({ error: err.message });
  }
});

// Get entry by ID
app.get('/entry/:id', (req, res) => {
  const data = loadKBData();
  const entry = data.entries.find(e => e.id === parseInt(req.params.id));
  
  if (!entry) {
    return res.status(404).json({ error: 'Entry not found' });
  }
  
  res.json(entry);
});

// Get categories
app.get('/categories', (req, res) => {
  const data = loadKBData();
  res.json(data.categories);
});

// Start server
app.listen(PORT, () => {
  console.log(`GenAI KB Search API running on port ${PORT}`);
  console.log(`Endpoints:`);
  console.log(`  GET  /health        - Health check`);
  console.log(`  POST /search        - Semantic search with AI summary`);
  console.log(`  POST /text-search   - Simple text search`);
  console.log(`  GET  /entry/:id     - Get single entry`);
  console.log(`  GET  /categories    - Get category list`);
  
  // Pre-load data
  loadKBData();
  console.log(`\nLoaded ${kbData.entries.length} entries, ${kbData.categories.length} categories`);
  
  if (OPENAI_API_KEY) {
    console.log('✓ OpenAI API key configured');
    // Pre-generate embeddings in background
    getEmbeddingsCache().then(() => {
      console.log('✓ Embeddings ready');
    });
  } else {
    console.log('✗ OpenAI API key not set (set OPENAI_API_KEY env var)');
  }
  
  if (ANTHROPIC_API_KEY) {
    console.log('✓ Anthropic API key configured');
  }
});

module.exports = { semanticSearch, textSearch, generateSummary };
