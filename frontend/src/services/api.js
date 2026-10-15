/**
 * Dedicated API Service communicating with the FastAPI FraudLens backend.
 */

const BASE_URL = '/api';

async function handleResponse(response) {
  if (!response.ok) {
    let errorDetail = `HTTP Error ${response.status}: ${response.statusText}`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorDetail = typeof errorJson.detail === 'string'
          ? errorJson.detail
          : JSON.stringify(errorJson.detail);
      }
    } catch {
      // Fallback to generic message
    }
    throw new Error(errorDetail);
  }
  return response.json();
}

export const api = {
  // System & Health
  async getHealth() {
    const res = await fetch(`${BASE_URL}/health`);
    return handleResponse(res);
  },

  async getAlgorithms() {
    const res = await fetch(`${BASE_URL}/algorithms`);
    return handleResponse(res);
  },

  // Dataset
  async getDatasetSummary(forceRefresh = false) {
    const res = await fetch(`${BASE_URL}/dataset/summary?force_refresh=${forceRefresh}`);
    return handleResponse(res);
  },

  async getDatasetPreview(limit = 10) {
    const res = await fetch(`${BASE_URL}/dataset/preview?limit=${limit}`);
    return handleResponse(res);
  },

  // Mining Algorithms
  async runApriori(params) {
    const res = await fetch(`${BASE_URL}/apriori/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse(res);
  },

  async runAdaptiveApriori(params) {
    const res = await fetch(`${BASE_URL}/adaptive-apriori/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse(res);
  },

  async runUtilityMining(params) {
    const res = await fetch(`${BASE_URL}/utility-mining/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse(res);
  },

  // Random Forest Classification
  async trainRandomForest(params) {
    const res = await fetch(`${BASE_URL}/random-forest/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse(res);
  },

  async getRFEvaluation() {
    const res = await fetch(`${BASE_URL}/random-forest/evaluation`);
    return handleResponse(res);
  },

  async predictTransactions(transactions, threshold = null) {
    const payload = { transactions };
    if (threshold !== null) {
      payload.decision_threshold = parseFloat(threshold);
    }
    const res = await fetch(`${BASE_URL}/random-forest/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse(res);
  },

  // Job Management
  async getJob(jobId) {
    const res = await fetch(`${BASE_URL}/jobs/${jobId}`);
    return handleResponse(res);
  },

  getExportUrl(resultId) {
    return `${BASE_URL}/exports/${resultId}`;
  },
};
