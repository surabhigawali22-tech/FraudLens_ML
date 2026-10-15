import React, { useState } from 'react';
import { GitFork, Play, Download, Info, CheckCircle2, AlertTriangle, ArrowRight, Filter } from 'lucide-react';
import { api } from '../services/api';

export function AprioriPage() {
  const [minSupport, setMinSupport] = useState(0.08);
  const [minConfidence, setMinConfidence] = useState(0.50);
  const [maxLen, setMaxLen] = useState(3);
  const [sampleSize, setSampleSize] = useState(25000);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [activeView, setActiveView] = useState('rules'); // 'rules' | 'itemsets'

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.runApriori({
        min_support: parseFloat(minSupport),
        min_confidence: parseFloat(minConfidence),
        max_len: parseInt(maxLen, 10),
        sample_size: parseInt(sampleSize, 10),
        include_retrospective_target: false,
      });
      setResults(data);
    } catch (err) {
      setError(err.message || 'Failed running Standard Apriori');
    } finally {
      setLoading(false);
    }
  };

  const getInterpretation = (rule) => {
    const ant = rule.antecedent.map(a => a.replace(/^(Format|PaymentCurr|AmountBand|Transfer|Currency):/, '')).join(' AND ');
    const con = rule.consequent.map(c => c.replace(/^(Format|PaymentCurr|AmountBand|Transfer|Currency):/, '')).join(' AND ');
    const confPct = (rule.confidence * 100).toFixed(1);
    const liftX = Number(rule.lift).toFixed(1);
    return `When transactions involve [${ant}], ${confPct}% of the time they also involve [${con}]. This co-occurrence is ${liftX}x more frequent than random chance.`;
  };

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <h1 className="page-title">Standard Apriori — Common Transaction Patterns</h1>
        <p className="page-description">
          Discovers frequent behavioral combinations in banking transactions. Learns associative rules between
          payment formats, transfer destinations, currency types, and volume tiers.
        </p>
      </div>

      {/* Metric Explanations for Compliance Staff */}
      <div className="grid-cols-3" style={{ marginBottom: '1.5rem' }}>
        <div className="stat-card" style={{ padding: '1rem 1.25rem' }}>
          <div className="stat-label">Metric: Support</div>
          <div style={{ fontSize: '0.875rem', color: '#334155', marginTop: '0.25rem' }}>
            The percentage of all analysed transactions that contain this exact combination. Higher support indicates widespread baseline bank activity.
          </div>
        </div>

        <div className="stat-card" style={{ padding: '1rem 1.25rem' }}>
          <div className="stat-label">Metric: Confidence</div>
          <div style={{ fontSize: '0.875rem', color: '#334155', marginTop: '0.25rem' }}>
            The conditional probability of the outcome given the condition. (e.g. 70% confidence means 7 out of 10 times the outcome co-occurs).
          </div>
        </div>

        <div className="stat-card" style={{ padding: '1rem 1.25rem' }}>
          <div className="stat-label">Metric: Lift</div>
          <div style={{ fontSize: '0.875rem', color: '#334155', marginTop: '0.25rem' }}>
            Strength of relationship over random independence. <strong>Lift &gt; 1</strong> means the items attract each other; <strong>Lift = 1</strong> means independence.
          </div>
        </div>
      </div>

      {/* Control Bar */}
      <div className="filter-bar">
        <div className="form-group">
          <label className="form-label">Minimum Support (0.01 – 0.50)</label>
          <select className="form-select" value={minSupport} onChange={(e) => setMinSupport(e.target.value)}>
            <option value="0.05">0.05 (5% occurrence)</option>
            <option value="0.08">0.08 (8% occurrence)</option>
            <option value="0.10">0.10 (10% occurrence)</option>
            <option value="0.15">0.15 (15% occurrence)</option>
            <option value="0.20">0.20 (20% occurrence)</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Minimum Confidence (0.2 – 0.9)</label>
          <select className="form-select" value={minConfidence} onChange={(e) => setMinConfidence(e.target.value)}>
            <option value="0.40">0.40 (40% reliability)</option>
            <option value="0.50">0.50 (50% reliability)</option>
            <option value="0.60">0.60 (60% reliability)</option>
            <option value="0.75">0.75 (75% reliability)</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Max Pattern Length</label>
          <select className="form-select" value={maxLen} onChange={(e) => setMaxLen(e.target.value)}>
            <option value="2">2 items (Pairings)</option>
            <option value="3">3 items (Triplets)</option>
            <option value="4">4 items (Quadruplets)</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Analysis Sample Window</label>
          <select className="form-select" value={sampleSize} onChange={(e) => setSampleSize(e.target.value)}>
            <option value="10000">10,000 transactions (Fast Trial)</option>
            <option value="25000">25,000 transactions (Recommended)</option>
            <option value="50000">50,000 transactions (Comprehensive)</option>
          </select>
        </div>

        <button className="btn btn-primary" onClick={handleRun} disabled={loading} style={{ height: '38px' }}>
          <Play size={15} />
          <span>{loading ? 'Mining Patterns...' : 'Run Analysis'}</span>
        </button>

        {results && results.export_url && (
          <a
            href={results.export_url}
            download
            className="btn btn-secondary"
            style={{ height: '38px', textDecoration: 'none' }}
          >
            <Download size={15} />
            <span>Download CSV</span>
          </a>
        )}
      </div>

      {error && (
        <div className="alert-banner warning">
          <AlertTriangle size={18} />
          <div>{error}</div>
        </div>
      )}

      {/* Results View */}
      {results ? (
        <div className="card">
          <div className="card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <span style={{ fontWeight: 600, color: '#0f172a' }}>
                Discovered: {results.rules_count} Association Rules, {results.itemsets_count} Frequent Itemsets
              </span>
              <span className="badge badge-info">
                Analyzed {Number(results.sample_size).toLocaleString()} records in {results.runtime_seconds}s
                {results.cached && ' (from deterministic cache)'}
              </span>
            </div>

            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button
                className={`btn ${activeView === 'rules' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setActiveView('rules')}
                style={{ padding: '0.35rem 0.75rem', fontSize: '0.8125rem' }}
              >
                Association Rules ({results.top_rules.length})
              </button>
              <button
                className={`btn ${activeView === 'itemsets' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setActiveView('itemsets')}
                style={{ padding: '0.35rem 0.75rem', fontSize: '0.8125rem' }}
              >
                Frequent Itemsets ({results.top_itemsets.length})
              </button>
            </div>
          </div>

          {activeView === 'rules' ? (
            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th style={{ width: '22%' }}>Condition (Antecedent)</th>
                    <th style={{ width: '22%' }}>Associated Outcome (Consequent)</th>
                    <th>Support</th>
                    <th>Confidence</th>
                    <th>Lift</th>
                    <th style={{ width: '35%' }}>Plain-Language Bank Interpretation</th>
                  </tr>
                </thead>
                <tbody>
                  {results.top_rules.length === 0 ? (
                    <tr>
                      <td colSpan={6} style={{ textAlign: 'center', padding: '2rem', color: '#64748b' }}>
                        No rules met the selected support ({minSupport}) and confidence ({minConfidence}) thresholds. Try lowering thresholds.
                      </td>
                    </tr>
                  ) : (
                    results.top_rules.map((rule, idx) => (
                      <tr key={idx}>
                        <td>
                          {rule.antecedent.map((item, i) => (
                            <span key={i} className="item-chip">{item}</span>
                          ))}
                        </td>
                        <td>
                          {rule.consequent.map((item, i) => (
                            <span key={i} className="item-chip" style={{ backgroundColor: '#e0f2fe', borderColor: '#bae6fd' }}>
                              {item}
                            </span>
                          ))}
                        </td>
                        <td>{(rule.support * 100).toFixed(2)}%</td>
                        <td><strong>{(rule.confidence * 100).toFixed(1)}%</strong></td>
                        <td>
                          <span className={`badge ${rule.lift >= 3 ? 'badge-risk-mid' : 'badge-info'}`}>
                            {Number(rule.lift).toFixed(2)}x
                          </span>
                        </td>
                        <td style={{ fontSize: '0.8125rem', color: '#334155' }}>
                          {getInterpretation(rule)}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="table-responsive">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Itemset Combination</th>
                    <th>Items Count</th>
                    <th>Transaction Count</th>
                    <th>Relative Support</th>
                  </tr>
                </thead>
                <tbody>
                  {results.top_itemsets.map((itemset, idx) => (
                    <tr key={idx}>
                      <td>
                        {itemset.items.map((item, i) => (
                          <span key={i} className="item-chip">{item}</span>
                        ))}
                      </td>
                      <td>{itemset.items.length} items</td>
                      <td>{Number(itemset.support_count).toLocaleString()} txs</td>
                      <td><strong>{(itemset.support * 100).toFixed(2)}%</strong></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ) : (
        <div className="state-box">
          <GitFork size={48} className="state-icon" color="#94a3b8" />
          <h3 className="state-title">No Analysis Loaded Yet</h3>
          <p className="state-subtext">
            Configure the support and confidence settings above and click "Run Analysis" to discover common transaction patterns.
          </p>
          <button className="btn btn-primary" onClick={handleRun} disabled={loading}>
            <Play size={16} />
            <span>Launch Initial Analysis (25,000 Rows)</span>
          </button>
        </div>
      )}
    </div>
  );
}
