const { pipeline } = require('@xenova/transformers');

let embedder = null;

/**
 * Get or initialize the embedding pipeline using HuggingFace all-MiniLM-L6-v2.
 * The model runs locally — no API key required.
 * @returns {Promise<Function>} The feature-extraction pipeline
 */
async function getEmbedder() {
  if (!embedder) {
    embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  return embedder;
}

/**
 * Generate an embedding vector for the given text using all-MiniLM-L6-v2.
 * Runs locally via @xenova/transformers — completely free, no API key needed.
 *
 * @param {string} text - The text to embed
 * @returns {Promise<number[]>} 384-dimensional embedding vector
 */
async function embed(text) {
  const extractor = await getEmbedder();
  const output = await extractor(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

/**
 * Generate embedding vectors for multiple texts in a single batch.
 *
 * @param {string[]} texts - Array of texts to embed
 * @returns {Promise<number[][]>} Array of 384-dimensional embedding vectors
 */
async function embedBatch(texts) {
  const extractor = await getEmbedder();
  const results = [];
  for (const text of texts) {
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    results.push(Array.from(output.data));
  }
  return results;
}

/**
 * Compute cosine similarity between two embedding vectors.
 *
 * @param {number[]} a - First embedding vector
 * @param {number[]} b - Second embedding vector
 * @returns {number} Cosine similarity score between -1 and 1
 */
function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

module.exports = { embed, embedBatch, cosineSimilarity, getEmbedder };
