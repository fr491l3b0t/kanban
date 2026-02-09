/**
 * Search Agent for Telegram Integration
 * 
 * This module provides a search interface that can be called from Telegram
 * or other chat platforms to search the GenAI Knowledge Base.
 * 
 * Usage:
 *   const searchAgent = require('./search-agent');
 *   const results = await searchAgent.search("latest from karpathy");
 * 
 * CLI Usage:
 *   node search-agent.js "your search query"
 */

const fs = require('fs');
const path = require('path');

// Configuration
const KB_PATH = path.join(__dirname, 'genai-kb.json');
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const API_BASE_URL = process.env.KB_API_URL || 'http://localhost:3456';

// Cache
let kbCache = null;
let kbLastModified = null;

/**
 * Load the knowledge base data
 */
function loadKB() {
  try {
    const stats = fs.statSync(KB_PATH);
    
    // Reload if file changed
    if (!kbCache || stats.mtime.getTime() !== kbLastModified) {
      const data = JSON.parse(fs.readFileSync(KB_PATH, 'utf8'));
      kbCache = data;
      kbLastModified = stats.mtime.getTime();
    }
    
    return kbCache;
  } catch (err) {
    throw new Error(`Failed to load KB: ${err.message}`);
  }
}

/**
 * Simple text-based search (no API calls needed)
 */
function localSearch(query, options = {}) {
  const { category, limit = 10 } = options;
  const data = loadKB();
  const queryLower = query.toLowerCase();
  const queryWords = queryLower.split(/\s+/).filter(w => w.length > 2);
  
  let results = data.entries.map(entry => {
    const text = `${entry.title} ${entry.summary || ''} ${entry.category} ${entry.source} ${(entry.tags || []).join(' ')}`.toLowerCase();
    
    // Scoring
    let score = 0;
    let matches = [];
    
    queryWords.forEach(word => {
      if (text.includes(word)) {
        score += 1;
        matches.push(word);
      }
    });
    
    // Boosts
    const titleLower = entry.title.toLowerCase();
    if (titleLower.includes(queryLower)) score += 10;
    else if (queryWords.every(w => titleLower.includes(w))) score += 5;
    
    if (entry.summary?.toLowerCase().includes(queryLower)) score += 3;
    if (entry.source?.toLowerCase().includes(queryLower)) score += 2;
    
    // Date boost (prefer newer)
    if (entry.date) {
      const entryDate = new Date(entry.date);
      const now = new Date();
      const daysOld = (now - entryDate) / (1000 * 60 * 60 * 24);
      if (daysOld < 30) score += 2;
      else if (daysOld < 90) score += 1;
    }
    
    return { entry, score, matches: [...new Set(matches)] };
  });
  
  // Apply category filter
  if (category && category !== 'all') {
    results = results.filter(r => r.entry.category.toLowerCase() === category.toLowerCase());
  }
  
  // Sort and filter
  results.sort((a, b) => b.score - a.score);
  return results.filter(r => r.score > 0).slice(0, limit);
}

/**
 * API-based semantic search
 */
async function apiSearch(query, options = {}) {
  const { category, dateFrom, dateTo, limit = 10, includeSummary = true } = options;
  
  try {
    const response = await fetch(`${API_BASE_URL}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        category,
        dateFrom,
        dateTo,
        limit,
        includeSummary
      })
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
  } catch (err) {
    // Fall back to local search
    console.error('API search failed, falling back to local:', err.message);
    const results = localSearch(query, { category, limit });
    return {
      summary: null,
      results: results.map(r => ({ ...r.entry, relevance: Math.min(r.score * 10, 100) })),
      total: results.length,
      fallback: true
    };
  }
}

/**
 * Format search results for display
 */
function formatResults(data, options = {}) {
  const { query, format = 'telegram', includeUrls = true } = options;
  
  if (!data.results || data.results.length === 0) {
    return `🔍 No results found for "${query}"`;
  }
  
  const lines = [];
  
  // Header
  lines.push(`🔍 *Search Results: "${escapeMarkdown(query)}"*`);
  lines.push(`Found ${data.total} entries\\n`);
  
  // AI Summary
  if (data.summary) {
    lines.push('✦ *AI Summary*');
    lines.push(escapeMarkdown(data.summary.substring(0, 500)));
    if (data.summary.length > 500) lines.push('...');
    lines.push('');
  }
  
  // Results
  lines.push('*Top Results:*');
  
  data.results.slice(0, 5).forEach((entry, i) => {
    const relevance = entry.relevance || Math.round((1 - (i * 0.1)) * 100);
    const title = escapeMarkdown(entry.title.substring(0, 60));
    const category = escapeMarkdown(entry.category);
    const source = escapeMarkdown(entry.source || 'Unknown');
    
    lines.push(`${i + 1}\. *${title}*${entry.title.length > 60 ? '...' : ''}`);
    lines.push(`   📁 ${category} · 👤 ${source}`);
    
    if (includeUrls) {
      const url = entry.url.replace(/\)/g, '\\)');
      lines.push(`   🔗 [View Entry](${url})`);
    }
    
    if (entry.summary && format === 'detailed') {
      const summary = escapeMarkdown(entry.summary.substring(0, 120));
      lines.push(`   📝 ${summary}${entry.summary.length > 120 ? '...' : ''}`);
    }
    
    lines.push(`   📊 ${relevance}% match`);
    lines.push('');
  });
  
  if (data.results.length > 5) {
    lines.push(`_...and ${data.results.length - 5} more results_`);
  }
  
  if (data.fallback) {
    lines.push('\\n⚠️ _Using local search (API unavailable)_');
  }
  
  return lines.join('\\n');
}

/**
 * Escape Markdown special characters
 */
function escapeMarkdown(text) {
  if (!text) return '';
  return text
    .replace(/\\/g, '\\\\')
    .replace(/\*/g, '\\*')
    .replace(/_/g, '\\_')
    .replace(/\[/g, '\\[')
    .replace(/\]/g, '\\]')
    .replace(/\(/g, '\\(')
    .replace(/\)/g, '\\)')
    .replace(/~/g, '\\~')
    .replace(/`/g, '\\`')
    .replace(/>/g, '\\>');
}

/**
 * Main search function - smart routing
 */
async function search(query, options = {}) {
  const { useApi = true, ...searchOptions } = options;
  
  // Try API first if available
  if (useApi) {
    try {
      const data = await apiSearch(query, searchOptions);
      return {
        success: true,
        data,
        formatted: formatResults(data, { query, ...options })
      };
    } catch (err) {
      console.error('API search failed:', err.message);
    }
  }
  
  // Fall back to local search
  const results = localSearch(query, searchOptions);
  const data = {
    summary: null,
    results: results.map(r => ({ ...r.entry, relevance: Math.min(r.score * 10, 100) })),
    total: results.length,
    local: true
  };
  
  return {
    success: true,
    data,
    formatted: formatResults(data, { query, ...options })
  };
}

/**
 * Get quick stats about the KB
 */
function getStats() {
  const data = loadKB();
  return {
    totalEntries: data.entries.length,
    categories: data.categories,
    sources: [...new Set(data.entries.map(e => e.source))].filter(Boolean).length,
    lastUpdated: data.lastUpdated
  };
}

/**
 * Get random entry
 */
function getRandomEntry(category = null) {
  const data = loadKB();
  let entries = data.entries;
  
  if (category && category !== 'all') {
    entries = entries.filter(e => e.category.toLowerCase() === category.toLowerCase());
  }
  
  if (entries.length === 0) return null;
  
  return entries[Math.floor(Math.random() * entries.length)];
}

// Export for module use
module.exports = {
  search,
  localSearch,
  apiSearch,
  formatResults,
  getStats,
  getRandomEntry,
  loadKB
};

// CLI usage
if (require.main === module) {
  const query = process.argv.slice(2).join(' ');
  
  if (!query) {
    console.log('GenAI KB Search Agent');
    console.log('');
    console.log('Usage:');
    console.log('  node search-agent.js "your search query"');
    console.log('  node search-agent.js --stats');
    console.log('  node search-agent.js --random [category]');
    console.log('');
    console.log('Environment Variables:');
    console.log('  OPENAI_API_KEY  - For API-based semantic search');
    console.log('  KB_API_URL      - Search API endpoint (default: http://localhost:3456)');
    process.exit(0);
  }
  
  if (query === '--stats') {
    const stats = getStats();
    console.log('Knowledge Base Stats');
    console.log('====================');
    console.log(`Total Entries: ${stats.totalEntries}`);
    console.log(`Categories: ${stats.categories.join(', ')}`);
    console.log(`Unique Sources: ${stats.sources}`);
    console.log(`Last Updated: ${stats.lastUpdated}`);
    process.exit(0);
  }
  
  if (query.startsWith('--random')) {
    const category = query.split(' ')[1] || null;
    const entry = getRandomEntry(category);
    if (!entry) {
      console.log('No entries found' + (category ? ` in category "${category}"` : ''));
      process.exit(1);
    }
    console.log('Random Entry');
    console.log('============');
    console.log(`Title: ${entry.title}`);
    console.log(`Category: ${entry.category}`);
    console.log(`Source: ${entry.source}`);
    console.log(`Date: ${entry.date || 'N/A'}`);
    console.log(`URL: ${entry.url}`);
    if (entry.summary) {
      console.log(`\nSummary: ${entry.summary.substring(0, 200)}...`);
    }
    process.exit(0);
  }
  
  // Run search
  console.log(`Searching: "${query}"...\n`);
  
  search(query, { format: 'detailed' }).then(result => {
    console.log(result.formatted.replace(/\\n/g, '\n').replace(/\\([*_\[\]\(\)~`>])/g, '$1'));
  }).catch(err => {
    console.error('Search failed:', err.message);
    process.exit(1);
  });
}
