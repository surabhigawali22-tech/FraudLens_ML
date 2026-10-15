import React, { useState } from 'react';
import { Gem, Play, Download, Info, CheckCircle2, AlertTriangle, DollarSign, Layers } from 'lucide-react';
import { api } from '../services/api';

export function UtilityMiningPage() {
  const [minUtilityPct, setMinUtilityPct] = useState(0.05);
  const [maxLen, setMaxLen] = useState(3);
  const [sampleSize, setSampleSize] = useState(25000);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.runUtilityMining({
        min_utility_pct: parseFloat(minUtilityPct),
        max_len: parseInt(maxLen, 10),
        sample_size: parseInt(sampleSize, 10),
      });
      setResults(data);
    } catch (err) {
      setError(err.message || 'Failed running Utility Mining');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <h1 className="page-title">Utility-Based Mining — Highest-Utility Patterns</h1>
        <p className="page-description">
          Discovers transaction combinations that generate high cumulative dollar volume exposure,
          using Two-Phase High-Utility Itemset Mining with Transaction-Weighted Utility (TWU).
        </p>
      </div>

      {/* Critical Banking Disclaimer Box */}
      <div className="alert-banner warning" style={{ marginBottom: '1.75rem' }}>
        <DollarSign size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
        <div>
          <strong>Financial Exposure Proxy Limitation:</strong> Transaction amount is an objective proxy representing{' '}
          <em>cumulative financial volume throughput</em>. High-utility itemsets identify pattern combinations where
          large monetary sums flow. <em>Transaction volume is never synonymous with confirmed criminal profit, bank loss, or laundering severity.</em>
        </div>
      </div>

      {/* Mathematical Principles Box */}
      <div className="card" style={{ marginBottom: '1.75rem' }}>
        <div className="card-body">
          <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#0f172a', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Layers size={18} color="#d97706" />
            <span>Why Two-Phase HUIM with Transaction-Weighted Utility (TWU)?</span>
          </h3>
          <p style={{ fontSize: '0.875rem', color: '#475569', lineHeight: 1.6, marginBottom: '0.5rem' }}>
            Standard Apriori only counts frequency: a $10 transfer and a $10,000,000 transfer count identically.
            However, raw utility is <em>non-monotonic</em> (adding items increases the sum), which prevents standard downward-closure pruning.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem', fontSize: '0.8125rem' }}>
            <div style={{ backgroundColor: '#f8fafc', padding: '0.75rem', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
              <strong>Phase 1 (TWU Downward-Closure):</strong>
              <div style={{ color: '#64748b', marginTop: '0.25rem' }}>
                Computes Transaction-Weighted Utility. If TWU(X) &lt; min_utility, all supersets are mathematically guaranteed to have utility &lt; min_utility and are pruned safely.
              </div>
            </div>
            <div style={{ backgroundColor: '#f8fafc', padding: '0.75rem', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
              <strong>Phase 2 (Exact Verification):</strong>
              <div style={{ color: '#64748b', marginTop: '0.25rem' }}>
                Scans surviving candidates to compute exact dollar utilities, filtering out TWU overestimates and returning genuine high-volume patterns.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Control Bar */}
      <div className="filter-bar">
        <div className="form-group">
          <label className="form-label">Minimum Volume Threshold (% of Sample Volume)</label>
          <select className="form-select" value={minUtilityPct} onChange={(e) => setMinUtilityPct(e.target.value)}>
            <option value="0.02">0.02 (2% of Total Dollar Volume)</option>
            <option value="0.05">0.05 (5% of Total Dollar Volume)</option>
            <option value="0.10">0.10 (10% of Total Dollar Volume)</option>
            <option value="0.15">0.15 (15% of Total Dollar Volume)</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Max Itemset Length</label>
          <select className="form-select" value={maxLen} onChange={(e) => setMaxLen(e.target.value)}>
            <option value="2">2 items (Pairings)</option>
            <option value="3">3 items (Triplets)</option>
            <option value="4">4 items (Quadruplets)</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Sample Window</label>
          <select className="form-select" value={sampleSize} onChange={(e) => setSampleSize(e.target.value)}>
            <option value="10000">10,000 transactions</option>
            <option value="25000">25,000 transactions (Recommended)</option>
            <option value="50000">50,000 transactions</option>
          </select>
        </div>

        <button className="btn btn-primary" onClick={handleRun} disabled={loading} style={{ height: '38px', backgroundColor: '#d97706' }}>
          <Play size={15} />
          <span>{loading ? 'Calculating Utility...' : 'Run Utility Mining'}</span>
        </button>

        {results && results.export_url && (
          <a href={results.export_url} download className="btn btn-secondary" style={{ height: '38px', textDecoration: 'none' }}>
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
                Discovered: {results.itemsets_count} High-Utility Itemsets
              </span>
              <span className="badge badge-info">
                Sample Volume: ${Number(results.parameters.total_sample_volume || 0).toLocaleString()} | Threshold: ${Number(results.parameters.effective_min_utility || 0).toLocaleString()}
              </span>
            </div>
          </div>

          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ width: '35%' }}>Pattern Combination</th>
                  <th>Exact Utility ($ Volume Exposure)</th>
                  <th>TWU Upper Bound</th>
                  <th>Frequency (Count)</th>
                  <th style={{ width: '25%' }}>Why This Pattern Appears</th>
                </tr>
              </thead>
              <tbody>
                {results.top_itemsets.length === 0 ? (
                  <tr>
                    <td colSpan={5} style={{ textAlign: 'center', padding: '2rem', color: '#64748b' }}>
                      No patterns exceeded the selected utility volume threshold. Try lowering the percentage threshold.
                    </td>
                  </tr>
                ) : (
                  results.top_itemsets.map((hui, idx) => (
                    <tr key={idx}>
                      <td>
                        {hui.items.map((item, i) => (
                          <span key={i} className="item-chip" style={{ backgroundColor: '#fffbeb', borderColor: '#fde68a' }}>
                            {item}
                          </span>
                        ))}
                      </td>
                      <td>
                        <strong style={{ color: '#b45309' }}>
                          ${Number(hui.utility).toLocaleString()}
                        </strong>
                      </td>
                      <td style={{ color: '#64748b' }}>
                        ${Number(hui.twu).toLocaleString()}
                      </td>
                      <td>
                        {Number(hui.support_count).toLocaleString()} txs
                      </td>
                      <td style={{ fontSize: '0.8125rem', color: '#334155' }}>
                        High-dollar concentration; transactions sharing this combination carry substantial financial throughput.
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="state-box">
          <Gem size={48} className="state-icon" color="#94a3b8" />
          <h3 className="state-title">No Utility Analysis Loaded</h3>
          <p className="state-subtext">
            Choose a percentage of total transaction dollar volume and run Two-Phase HUIM to discover patterns carrying high monetary exposure.
          </p>
          <button className="btn btn-primary" onClick={handleRun} disabled={loading} style={{ backgroundColor: '#d97706' }}>
            <Play size={16} />
            <span>Launch Utility Mining (25,000 Rows)</span>
          </button>
        </div>
      )}
    </div>
  );
}
