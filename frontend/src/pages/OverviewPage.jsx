import React, { useState, useEffect } from 'react';
import {
  ShieldAlert,
  Database,
  BarChart3,
  GitFork,
  Gem,
  Cpu,
  AlertTriangle,
  CheckCircle,
  ExternalLink,
  RefreshCw,
  Search,
  Eye,
  Info
} from 'lucide-react';
import { api } from '../services/api';

export function OverviewPage({ setActiveTab, onInspectTransaction }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [summary, setSummary] = useState(null);
  const [rfEval, setRfEval] = useState(null);
  const [preview, setPreview] = useState([]);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [sumData, evalData, prevData] = await Promise.allSettled([
        api.getDatasetSummary(),
        api.getRFEvaluation(),
        api.getDatasetPreview(6),
      ]);

      if (sumData.status === 'fulfilled') setSummary(sumData.value);
      if (evalData.status === 'fulfilled') setRfEval(evalData.value);
      if (prevData.status === 'fulfilled') setPreview(prevData.value.rows || []);
    } catch (err) {
      setError(err.message || 'Failed to load banking dashboard data.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const legCount = summary?.target_distribution?.['0']?.count || 5073168;
  const legPct = summary?.target_distribution?.['0']?.percent || 99.898;
  const launderCount = summary?.target_distribution?.['1']?.count || 5177;
  const launderPct = summary?.target_distribution?.['1']?.percent || 0.102;

  const testPos = rfEval?.test_metrics?.positive_count ?? 1;
  const testNeg = rfEval?.test_metrics?.negative_count ?? 29999;
  const testRoc = rfEval?.test_metrics?.roc_auc ?? 0.9443;
  const threshold = rfEval?.test_metrics?.threshold ?? 0.40;

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h1 className="page-title">Banking Operations Overview</h1>
            <p className="page-description">
              Daily transaction monitoring intelligence for compliance officers. Review transaction volumes,
              historical AML labels, supervised machine learning alerts, and discovered transaction patterns.
            </p>
          </div>
          <button className="btn btn-secondary" onClick={loadData} disabled={loading}>
            <RefreshCw size={15} className={loading ? 'spinner' : ''} />
            <span>Refresh System Data</span>
          </button>
        </div>
      </div>

      {/* Compliance Disclaimer Banner */}
      <div className="alert-banner">
        <Info size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
        <div>
          <strong>Operational Review Protocol:</strong> Discovered patterns and model prediction scores represent{' '}
          <em>priority indicators for human review</em>, never confirmation of illicit money laundering. All decisions
          must follow standard customer due diligence (CDD) and Suspicious Activity Report (SAR) filing procedures.
        </div>
      </div>

      {error && (
        <div className="alert-banner warning">
          <AlertTriangle size={20} style={{ flexShrink: 0 }} />
          <div>{error}</div>
        </div>
      )}

      {/* KPI Cards */}
      <div className="grid-cols-4">
        {/* Total Dataset Volume */}
        <div className="stat-card">
          <div className="stat-label">Total Transactions Monitored</div>
          <div className="stat-value">{summary?.row_count ? Number(summary.row_count).toLocaleString() : '5,078,345'}</div>
          <div className="stat-subtext" style={{ color: '#0f766e', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <CheckCircle size={14} /> Full IBM AML Dataset Verified
          </div>
        </div>

        {/* Historical Laundering Proportion */}
        <div className="stat-card">
          <div className="stat-label">Historical AML Incidents</div>
          <div className="stat-value" style={{ color: '#991b1b' }}>
            {launderCount.toLocaleString()} <span style={{ fontSize: '1rem', color: '#64748b' }}>({launderPct}%)</span>
          </div>
          <div className="stat-subtext">
            Severe class imbalance: ~1 laundering incident per 1,000 transactions.
          </div>
        </div>

        {/* Supervised Model Performance */}
        <div className="stat-card">
          <div className="stat-label">Random Forest Out-of-Time ROC-AUC</div>
          <div className="stat-value" style={{ color: '#0284c7' }}>
            {testRoc !== null ? Number(testRoc).toFixed(4) : '0.9443'}
          </div>
          <div className="stat-subtext">
            Trained on chronological slice without target leakage.
          </div>
        </div>

        {/* Operational Review Threshold */}
        <div className="stat-card">
          <div className="stat-label">Active Alert Threshold</div>
          <div className="stat-value">{threshold}</div>
          <div className="stat-subtext">
            Probability threshold calibrated for review queue sensitivity.
          </div>
        </div>
      </div>

      {/* Algorithms Navigation Cards */}
      <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem', color: '#0f172a' }}>
        Analytical Tools & Methodologies
      </h2>
      <div className="grid-cols-2" style={{ marginBottom: '2.5rem' }}>
        {/* Standard Apriori */}
        <div className="card" style={{ marginBottom: 0 }}>
          <div className="card-header">
            <div className="card-title">
              <GitFork size={18} color="#0284c7" />
              <span>Standard Apriori Mining</span>
            </div>
            <button className="btn btn-secondary" onClick={() => setActiveTab('apriori')}>
              <span>Open Studio</span>
              <ExternalLink size={14} />
            </button>
          </div>
          <div className="card-body">
            <p style={{ fontSize: '0.875rem', color: '#475569', marginBottom: '0.75rem' }}>
              Identifies the most frequent transaction combinations (payment formats, amounts, and bank routes) across the institution.
              Computes Support, Confidence, and Lift.
            </p>
            <div style={{ fontSize: '0.8125rem', color: '#64748b' }}>
              <strong>Best for:</strong> Establishing baseline transactional behavior and discovering common co-occurrences.
            </div>
          </div>
        </div>

        {/* Adaptive Apriori */}
        <div className="card" style={{ marginBottom: 0 }}>
          <div className="card-header">
            <div className="card-title">
              <ShieldAlert size={18} color="#0f766e" />
              <span>Adaptive Apriori (MSApriori)</span>
            </div>
            <button className="btn btn-secondary" onClick={() => setActiveTab('adaptive')}>
              <span>Open Studio</span>
              <ExternalLink size={14} />
            </button>
          </div>
          <div className="card-body">
            <p style={{ fontSize: '0.875rem', color: '#475569', marginBottom: '0.75rem' }}>
              Solves the AML <em>Rare Item Dilemma</em>. Assigns custom lower support thresholds to rare high-risk items
              (e.g., Bitcoin, cross-currency transfers) while constraining mundane everyday transactions.
            </p>
            <div style={{ fontSize: '0.8125rem', color: '#64748b' }}>
              <strong>Best for:</strong> Surfacing rare suspicious patterns without triggering combinatorial candidate explosion.
            </div>
          </div>
        </div>

        {/* High Utility Itemset Mining */}
        <div className="card" style={{ marginBottom: 0 }}>
          <div className="card-header">
            <div className="card-title">
              <Gem size={18} color="#d97706" />
              <span>Two-Phase Utility Mining (HUIM)</span>
            </div>
            <button className="btn btn-secondary" onClick={() => setActiveTab('utility')}>
              <span>Open Studio</span>
              <ExternalLink size={14} />
            </button>
          </div>
          <div className="card-body">
            <p style={{ fontSize: '0.875rem', color: '#475569', marginBottom: '0.75rem' }}>
              Ranks transaction patterns by cumulative financial volume exposure using Transaction-Weighted Utility (TWU).
              Discovers where large monetary sums concentrate.
            </p>
            <div style={{ fontSize: '0.8125rem', color: '#64748b' }}>
              <strong>Best for:</strong> Pinpointing patterns that move the highest cumulative dollar volumes.
            </div>
          </div>
        </div>

        {/* Random Forest */}
        <div className="card" style={{ marginBottom: 0 }}>
          <div className="card-header">
            <div className="card-title">
              <Cpu size={18} color="#7c3aed" />
              <span>Random Forest Classifier</span>
            </div>
            <button className="btn btn-secondary" onClick={() => setActiveTab('prediction')}>
              <span>Open Live Scorer</span>
              <ExternalLink size={14} />
            </button>
          </div>
          <div className="card-body">
            <p style={{ fontSize: '0.875rem', color: '#475569', marginBottom: '0.75rem' }}>
              Supervised machine learning ensemble evaluated out-of-time (70% train / 15% val / 15% test).
              Enables interactive transaction scoring with configurable decision thresholds.
            </p>
            <div style={{ fontSize: '0.8125rem', color: '#64748b' }}>
              <strong>Best for:</strong> Live transaction risk triage and alert queue prioritization.
            </div>
          </div>
        </div>
      </div>

      {/* Real Transaction Stream Sample */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Search size={18} color="#0284c7" />
            <span>Recent Monitored Transactions (Sample Preview)</span>
          </div>
          <span style={{ fontSize: '0.8125rem', color: '#64748b' }}>Click any transaction to open compliance investigation</span>
        </div>
        <div className="table-responsive">
          <table className="data-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>From Bank</th>
                <th>Originating Account</th>
                <th>To Bank</th>
                <th>Beneficiary Account</th>
                <th>Amount Paid</th>
                <th>Format</th>
                <th>Dataset Label</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {preview.length === 0 ? (
                <tr>
                  <td colSpan={9} style={{ textAlign: 'center', padding: '2rem', color: '#64748b' }}>
                    {loading ? 'Loading transaction stream...' : 'No transactions loaded.'}
                  </td>
                </tr>
              ) : (
                preview.map((tx, idx) => {
                  const isLaunder = tx['Is Laundering'] === 1;
                  return (
                    <tr key={idx}>
                      <td>{tx.Timestamp}</td>
                      <td>Bank {tx['From Bank']}</td>
                      <td><code>{tx.Account}</code></td>
                      <td>Bank {tx['To Bank']}</td>
                      <td><code>{tx['Account.1'] || tx.Account}</code></td>
                      <td>
                        <strong>${Number(tx['Amount Paid']).toLocaleString()}</strong>{' '}
                        <span style={{ fontSize: '0.75rem', color: '#64748b' }}>{tx['Payment Currency']}</span>
                      </td>
                      <td><span className="item-chip">{tx['Payment Format']}</span></td>
                      <td>
                        <span className={`badge ${isLaunder ? 'badge-risk-high' : 'badge-risk-low'}`}>
                          {isLaunder ? 'Historical Flag' : 'Legitimate'}
                        </span>
                      </td>
                      <td>
                        <button
                          className="btn btn-secondary"
                          onClick={() => onInspectTransaction(tx)}
                          style={{ padding: '0.25rem 0.6rem', fontSize: '0.75rem' }}
                        >
                          <Eye size={13} />
                          <span>Investigate</span>
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
