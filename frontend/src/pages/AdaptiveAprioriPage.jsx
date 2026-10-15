import React, { useState } from 'react';
import { ShieldCheck, Play, Download, Info, CheckCircle2, AlertTriangle, ArrowRight, Layers } from 'lucide-react';
import { api } from '../services/api';

export function AdaptiveAprioriPage() {
  const [baseMinSupport, setBaseMinSupport] = useState(0.10);
  const [minConfidence, setMinConfidence] = useState(0.40);
  const [rareTargetMis, setRareTargetMis] = useState(0.0005);
  const [sampleSize, setSampleSize] = useState(25000);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  const handleRun = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.runAdaptiveApriori({
        base_min_support: parseFloat(baseMinSupport),
        min_confidence: parseFloat(minConfidence),
        mis_dict: {
          'Laundering:True': parseFloat(rareTargetMis),
          'Format:Bitcoin': 0.005,
          'Currency:CrossCurrency': 0.02,
          'AmountBand:VeryHigh(>50k)': 0.02,
        },
        max_len: 3,
        sample_size: parseInt(sampleSize, 10),
        include_retrospective_target: true,
      });
      setResults(data);
    } catch (err) {
      setError(err.message || 'Failed running Adaptive Apriori');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <h1 className="page-title">Adaptive Apriori — Adaptive Pattern Discovery</h1>
        <p className="page-description">
          Uses Multiple Minimum Supports Apriori (MSApriori) to uncover rare, high-risk patterns without
          generating millions of mundane everyday combinations.
        </p>
      </div>

      {/* The Strategic Rationale (Why Adaptive?) */}
      <div className="card" style={{ backgroundColor: '#ffffff', borderLeft: '4px solid #0f766e', marginBottom: '1.75rem' }}>
        <div className="card-body">
          <h3 style={{ fontSize: '1rem', fontWeight: 600, color: '#0f766e', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Layers size={18} />
            <span>The AML "Rare Item Dilemma" & Strategic Solution</span>
          </h3>
          <p style={{ fontSize: '0.875rem', color: '#334155', lineHeight: 1.6, marginBottom: '0.75rem' }}>
            In banking, suspicious transactions represent only ~0.10% of total activity. A standard uniform threshold of 5% eliminates
            all rare suspicious transfers. Lowering the uniform threshold to 0.05% causes <em>candidate explosion</em>, filling reports with
            millions of trivial combinations of domestic ACH and US Dollar transfers.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', fontSize: '0.8125rem' }}>
            <div style={{ backgroundColor: '#f8fafc', padding: '0.75rem', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
              <strong>Standard Apriori (Global 10%):</strong>
              <div style={{ color: '#64748b', marginTop: '0.25rem' }}>Prunes all rare patterns. Misses high-risk transfers entirely.</div>
            </div>
            <div style={{ backgroundColor: '#f8fafc', padding: '0.75rem', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
              <strong>Standard Apriori (Global 0.1%):</strong>
              <div style={{ color: '#64748b', marginTop: '0.25rem' }}>Memory exhaustion. Millions of mundane pairing permutations.</div>
            </div>
            <div style={{ backgroundColor: '#ccfbf1', padding: '0.75rem', borderRadius: '6px', border: '1px solid #99f6e4' }}>
              <strong style={{ color: '#0f766e' }}>Adaptive Apriori (MSApriori):</strong>
              <div style={{ color: '#115e59', marginTop: '0.25rem' }}>Mundane items stay at 10% MIS; rare high-risk items adapt to 0.05% MIS.</div>
            </div>
          </div>
        </div>
      </div>

      {/* Control Bar */}
      <div className="filter-bar">
        <div className="form-group">
          <label className="form-label">Base MIS (Mundane Items)</label>
          <select className="form-select" value={baseMinSupport} onChange={(e) => setBaseMinSupport(e.target.value)}>
            <option value="0.08">0.08 (8% Base Threshold)</option>
            <option value="0.10">0.10 (10% Base Threshold)</option>
            <option value="0.15">0.15 (15% Base Threshold)</option>
            <option value="0.20">0.20 (20% Base Threshold)</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Rare Item MIS Floor</label>
          <select className="form-select" value={rareTargetMis} onChange={(e) => setRareTargetMis(e.target.value)}>
            <option value="0.0005">0.0005 (0.05% - True AML Floor)</option>
            <option value="0.001">0.0010 (0.10% - Standard Rare)</option>
            <option value="0.005">0.0050 (0.50% - Moderate Sensitivity)</option>
          </select>
        </div>

        <div className="form-group">
          <label className="form-label">Minimum Confidence</label>
          <select className="form-select" value={minConfidence} onChange={(e) => setMinConfidence(e.target.value)}>
            <option value="0.30">0.30 (30% reliability)</option>
            <option value="0.40">0.40 (40% reliability)</option>
            <option value="0.50">0.50 (50% reliability)</option>
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

        <button className="btn btn-teal" onClick={handleRun} disabled={loading} style={{ height: '38px' }}>
          <Play size={15} />
          <span>{loading ? 'Discovering...' : 'Run Adaptive Discovery'}</span>
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

      {/* Results Table */}
      {results ? (
        <div className="card">
          <div className="card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
              <span style={{ fontWeight: 600, color: '#0f172a' }}>
                Discovered: {results.rules_count} Adaptive Rules ({results.itemsets_count} Itemsets)
              </span>
              <span className="badge badge-info">
                Analyzed {Number(results.sample_size).toLocaleString()} records in {results.runtime_seconds}s
                {results.cached && ' (cached)'}
              </span>
            </div>
          </div>

          <div className="table-responsive">
            <table className="data-table">
              <thead>
                <tr>
                  <th style={{ width: '25%' }}>Condition (Antecedent)</th>
                  <th style={{ width: '25%' }}>Associated Outcome</th>
                  <th>Support</th>
                  <th>Confidence</th>
                  <th>Lift</th>
                  <th style={{ width: '30%' }}>Operational Explanation</th>
                </tr>
              </thead>
              <tbody>
                {results.top_rules.map((rule, idx) => {
                  const hasRareItem = [...rule.antecedent, ...rule.consequent].some(i =>
                    i.includes('Bitcoin') || i.includes('CrossCurrency') || i.includes('Laundering') || i.includes('>50k')
                  );
                  return (
                    <tr key={idx}>
                      <td>
                        {rule.antecedent.map((item, i) => (
                          <span key={i} className="item-chip">{item}</span>
                        ))}
                      </td>
                      <td>
                        {rule.consequent.map((item, i) => (
                          <span key={i} className="item-chip" style={{ backgroundColor: '#ccfbf1', borderColor: '#99f6e4' }}>
                            {item}
                          </span>
                        ))}
                      </td>
                      <td>{(rule.support * 100).toFixed(2)}%</td>
                      <td><strong>{(rule.confidence * 100).toFixed(1)}%</strong></td>
                      <td>
                        <span className={`badge ${rule.lift >= 1.5 ? 'badge-risk-mid' : 'badge-info'}`}>
                          {Number(rule.lift).toFixed(2)}x
                        </span>
                      </td>
                      <td style={{ fontSize: '0.8125rem', color: '#334155' }}>
                        {hasRareItem ? (
                          <span style={{ color: '#0f766e', fontWeight: 500 }}>
                            Surfaced via adapted lower MIS. Standard Apriori at {baseMinSupport * 100}% would have pruned this pattern.
                          </span>
                        ) : (
                          <span>Common transaction pattern passing the higher {baseMinSupport * 100}% base threshold.</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="state-box">
          <ShieldCheck size={48} className="state-icon" color="#94a3b8" />
          <h3 className="state-title">No Adaptive Analysis Active</h3>
          <p className="state-subtext">
            Configure the base support and rare item floor, then run adaptive discovery to view patterns that cross multiple frequency levels.
          </p>
          <button className="btn btn-teal" onClick={handleRun} disabled={loading}>
            <Play size={16} />
            <span>Launch Adaptive Run (25,000 Rows)</span>
          </button>
        </div>
      )}
    </div>
  );
}
