import React, { useState, useEffect } from 'react';
import { Cpu, Play, AlertTriangle, CheckCircle, ShieldAlert, Sliders, HelpCircle, Activity } from 'lucide-react';
import { api } from '../services/api';

export function RandomForestPage() {
  const [evalData, setEvalData] = useState(null);
  const [evalLoading, setEvalLoading] = useState(false);
  const [trainLoading, setTrainLoading] = useState(false);
  const [evalError, setEvalError] = useState(null);

  const [trainSampleSize, setTrainSampleSize] = useState(50000);
  const [trainSuccess, setTrainSuccess] = useState(null);

  // Live Scoring Form State
  const [fromBank, setFromBank] = useState('10');
  const [toBank, setToBank] = useState('20');
  const [originAccount, setOriginAccount] = useState('8000EBD30');
  const [beneficiaryAccount, setBeneficiaryAccount] = useState('8000F5340');
  const [amountPaid, setAmountPaid] = useState('65000');
  const [amountReceived, setAmountReceived] = useState('65000');
  const [payCurrency, setPayCurrency] = useState('US Dollar');
  const [recvCurrency, setRecvCurrency] = useState('US Dollar');
  const [format, setFormat] = useState('Wire');
  const [threshold, setThreshold] = useState('0.40');

  const [predictLoading, setPredictLoading] = useState(false);
  const [predictResult, setPredictResult] = useState(null);
  const [predictError, setPredictError] = useState(null);

  const loadEvaluation = async () => {
    setEvalLoading(true);
    setEvalError(null);
    try {
      const data = await api.getRFEvaluation();
      setEvalData(data);
    } catch (err) {
      setEvalError(err.message || 'Failed loading model evaluation');
    } finally {
      setEvalLoading(false);
    }
  };

  useEffect(() => {
    loadEvaluation();
  }, []);

  const handleTrain = async () => {
    if (trainLoading) return;
    setTrainLoading(true);
    setEvalError(null);
    setTrainSuccess(null);
    try {
      const res = await api.trainRandomForest({
        sample_size: trainSampleSize,
        n_estimators: 100,
        decision_threshold: parseFloat(threshold),
        run_in_background: false,
      });
      setEvalData(res);
      setTrainSuccess(`Successfully retrained Random Forest on ${Number(res.sample_size || trainSampleSize).toLocaleString()} transactions!`);
    } catch (err) {
      setEvalError(err.message || 'Training failed');
    } finally {
      setTrainLoading(false);
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setPredictLoading(true);
    setPredictError(null);
    try {
      const tx = {
        Timestamp: '2022/09/01 12:00',
        'From Bank': parseInt(fromBank, 10),
        'To Bank': parseInt(toBank, 10),
        Account: originAccount,
        'Account.1': beneficiaryAccount,
        'Amount Received': parseFloat(amountReceived),
        'Amount Paid': parseFloat(amountPaid),
        'Receiving Currency': recvCurrency,
        'Payment Currency': payCurrency,
        'Payment Format': format,
      };

      const res = await api.predictTransactions([tx], parseFloat(threshold));
      if (res && res.predictions && res.predictions.length > 0) {
        setPredictResult(res.predictions[0]);
      }
    } catch (err) {
      setPredictError(err.message || 'Prediction failed');
    } finally {
      setPredictLoading(false);
    }
  };

  const applyPreset = (preset) => {
    if (preset === 'high_wire') {
      setFromBank('10');
      setToBank('20');
      setOriginAccount('8000EBD30');
      setBeneficiaryAccount('8000F5340');
      setAmountPaid('95000');
      setAmountReceived('95000');
      setPayCurrency('US Dollar');
      setRecvCurrency('US Dollar');
      setFormat('Wire');
    } else if (preset === 'routine_ach') {
      setFromBank('10');
      setToBank('10');
      setOriginAccount('8000A1110');
      setBeneficiaryAccount('8000B2220');
      setAmountPaid('1250');
      setAmountReceived('1250');
      setPayCurrency('US Dollar');
      setRecvCurrency('US Dollar');
      setFormat('ACH');
    } else if (preset === 'cross_currency') {
      setFromBank('12');
      setToBank('44');
      setOriginAccount('8000C3330');
      setBeneficiaryAccount('8000D4440');
      setAmountPaid('50000');
      setAmountReceived('53000');
      setPayCurrency('Euro');
      setRecvCurrency('US Dollar');
      setFormat('Wire');
    }
  };

  const testMetrics = evalData?.test_metrics;
  const cm = testMetrics?.confusion_matrix;

  return (
    <div>
      {/* Header */}
      <div className="page-header">
        <h1 className="page-title">Random Forest — Transaction Predictions</h1>
        <p className="page-description">
          Supervised machine learning pipeline evaluated out-of-time (70% train / 15% validation / 15% test).
          Score incoming transactions, inspect operational confusion matrices, and test decision threshold sensitivities.
        </p>
      </div>

      {/* Model Training & Evaluation Panel */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <div className="card-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Cpu size={18} color="#7c3aed" />
            <span style={{ fontWeight: 600, color: '#0f172a' }}>
              Model Status & Out-of-Time Performance Metrics
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <select
              className="form-select"
              value={trainSampleSize}
              onChange={(e) => setTrainSampleSize(parseInt(e.target.value, 10))}
              disabled={trainLoading}
              style={{ padding: '0.35rem 0.6rem', fontSize: '0.8125rem', width: 'auto' }}
            >
              <option value="50000">50,000 rows (Fast Trial)</option>
              <option value="100000">100,000 rows (Recommended, 2 test positives)</option>
              <option value="200000">200,000 rows (Full slice, ROC-AUC &gt; 0.94)</option>
            </select>
            <button className="btn btn-secondary" onClick={handleTrain} disabled={trainLoading} style={{ fontSize: '0.8125rem' }}>
              <Play size={14} className={trainLoading ? 'spinner' : ''} />
              <span>{trainLoading ? 'Training Model...' : `Retrain Pipeline (${(trainSampleSize / 1000).toFixed(0)}k Sample)`}</span>
            </button>
          </div>
        </div>

        <div className="card-body">
          {trainSuccess && (
            <div className="alert-banner" style={{ backgroundColor: '#f0fdf4', borderColor: '#bbf7d0', color: '#166534', marginBottom: '1.25rem' }}>
              <CheckCircle size={18} style={{ flexShrink: 0 }} />
              <div>{trainSuccess}</div>
            </div>
          )}

          {evalError && (
            <div className="alert-banner warning" style={{ marginBottom: '1.25rem' }}>
              <AlertTriangle size={18} style={{ flexShrink: 0 }} />
              <div>{evalError}</div>
            </div>
          )}

          {evalLoading ? (
            <div style={{ padding: '2rem', textAlign: 'center', color: '#64748b' }}>
              Loading model evaluation data...
            </div>
          ) : evalData && evalData.model_available && testMetrics ? (
            <div>
              {/* Optional Protocol Notes for Compliance */}
              {testMetrics.notes && (
                <div className="alert-banner warning" style={{ marginBottom: '1.25rem', padding: '0.75rem 1rem' }}>
                  <AlertTriangle size={18} style={{ flexShrink: 0 }} />
                  <div style={{ fontSize: '0.8125rem' }}>
                    <strong>Evaluation Window Protocol:</strong> {testMetrics.notes}
                  </div>
                </div>
              )}

              {/* Metrics Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '1rem', marginBottom: '1.5rem' }}>
                <div className="stat-card" style={{ padding: '0.875rem' }}>
                  <div className="stat-label">ROC-AUC Score</div>
                  <div className="stat-value" style={{ fontSize: '1.5rem', color: '#0284c7' }}>
                    {testMetrics.roc_auc !== null
                      ? Number(testMetrics.roc_auc).toFixed(4)
                      : (testMetrics.positive_count === 0 ? 'Undefined' : 'N/A')}
                  </div>
                  <div className="stat-subtext">Overall ranking quality</div>
                </div>

                <div className="stat-card" style={{ padding: '0.875rem' }}>
                  <div className="stat-label">Decision Threshold</div>
                  <div className="stat-value" style={{ fontSize: '1.5rem' }}>
                    {testMetrics.threshold}
                  </div>
                  <div className="stat-subtext">Operating decision point</div>
                </div>

                <div className="stat-card" style={{ padding: '0.875rem' }}>
                  <div className="stat-label">Precision</div>
                  <div className="stat-value" style={{ fontSize: '1.5rem' }}>
                    {testMetrics.precision !== null
                      ? `${(testMetrics.precision * 100).toFixed(1)}%`
                      : (testMetrics.positive_count === 0 ? 'Undefined' : 'N/A')}
                  </div>
                  <div className="stat-subtext">Alert accuracy (avoid alert fatigue)</div>
                </div>

                <div className="stat-card" style={{ padding: '0.875rem' }}>
                  <div className="stat-label">Recall</div>
                  <div className="stat-value" style={{ fontSize: '1.5rem' }}>
                    {testMetrics.recall !== null
                      ? `${(testMetrics.recall * 100).toFixed(1)}%`
                      : (testMetrics.positive_count === 0 ? 'Undefined' : 'N/A')}
                  </div>
                  <div className="stat-subtext">Detection rate of true incidents</div>
                </div>

                <div className="stat-card" style={{ padding: '0.875rem' }}>
                  <div className="stat-label">Test Set Volume</div>
                  <div className="stat-value" style={{ fontSize: '1.5rem' }}>
                    {Number(testMetrics.sample_size).toLocaleString()}
                  </div>
                  <div className="stat-subtext" style={{ fontSize: '0.75rem', color: '#64748b' }}>
                    {testMetrics.positive_count} Laundering / {Number(testMetrics.negative_count).toLocaleString()} Legitimate
                  </div>
                </div>
              </div>

              {/* Confusion Matrix Interpretation for Compliance */}
              {cm && (
                <div style={{ backgroundColor: '#f8fafc', padding: '1.25rem', borderRadius: '8px', border: '1px solid #e2e8f0', marginBottom: '1.5rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                    <h4 style={{ fontSize: '0.9rem', fontWeight: 600, color: '#0f172a', margin: 0 }}>
                      Operational Confusion Matrix (Test Window)
                    </h4>
                    <span style={{ fontSize: '0.75rem', color: '#64748b' }}>
                      Ground-Truth: <strong>{testMetrics.positive_count}</strong> actual laundering vs <strong>{Number(testMetrics.negative_count).toLocaleString()}</strong> legitimate
                    </span>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem', fontSize: '0.8125rem' }}>
                    <div style={{ backgroundColor: '#ffffff', padding: '0.75rem', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                      <strong style={{ color: '#166534' }}>True Negatives ({cm.tn.toLocaleString()}):</strong>
                      <div style={{ color: '#64748b', marginTop: '0.2rem' }}>
                        Legitimate transactions correctly processed without interrupting customer activity.
                      </div>
                    </div>
                    <div style={{ backgroundColor: '#ffffff', padding: '0.75rem', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                      <strong style={{ color: '#92400e' }}>False Positives ({cm.fp.toLocaleString()}):</strong>
                      <div style={{ color: '#64748b', marginTop: '0.2rem' }}>
                        Legitimate transactions flagged for review (review workload / false alarms).
                      </div>
                    </div>
                    <div style={{ backgroundColor: '#ffffff', padding: '0.75rem', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                      <strong style={{ color: '#991b1b' }}>False Negatives ({cm.fn.toLocaleString()}):</strong>
                      <div style={{ color: '#64748b', marginTop: '0.2rem' }}>
                        Laundering transactions that bypassed detection (compliance exposure).
                      </div>
                    </div>
                    <div style={{ backgroundColor: '#ffffff', padding: '0.75rem', borderRadius: '6px', border: '1px solid #e2e8f0' }}>
                      <strong style={{ color: '#0284c7' }}>True Positives ({cm.tp.toLocaleString()}):</strong>
                      <div style={{ color: '#64748b', marginTop: '0.2rem' }}>
                        Laundering incidents successfully stopped and forwarded for human review.
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Feature Importance */}
              {evalData.top_features && evalData.top_features.length > 0 && (
                <div>
                  <h4 style={{ fontSize: '0.9rem', fontWeight: 600, color: '#0f172a', marginBottom: '0.5rem' }}>
                    Top Predictive Decision Factors
                  </h4>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {evalData.top_features.slice(0, 7).map((feat, i) => (
                      <span key={i} className="item-chip" style={{ backgroundColor: '#ffffff', borderColor: '#cbd5e1' }}>
                        <strong>{feat.feature}</strong>: {(feat.importance * 100).toFixed(1)}% weight
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '2rem' }}>
              <p style={{ color: '#64748b', marginBottom: '1rem' }}>
                No model evaluation currently available. Click below to train the Random Forest pipeline on real chronological data.
              </p>
              <button className="btn btn-primary" onClick={handleTrain} disabled={trainLoading}>
                <Play size={15} />
                <span>{trainLoading ? 'Training Pipeline...' : 'Train Model on 50,000 Chronological Rows'}</span>
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Live Transaction Scoring Simulator */}
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <Activity size={18} color="#0284c7" />
            <span>Interactive Transaction Risk Scorer</span>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <span style={{ fontSize: '0.8125rem', color: '#64748b', alignSelf: 'center' }}>Presets:</span>
            <button className="btn btn-secondary" onClick={() => applyPreset('high_wire')} style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}>
              High-Dollar Wire
            </button>
            <button className="btn btn-secondary" onClick={() => applyPreset('routine_ach')} style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}>
              Routine ACH
            </button>
            <button className="btn btn-secondary" onClick={() => applyPreset('cross_currency')} style={{ fontSize: '0.75rem', padding: '0.25rem 0.5rem' }}>
              Cross-Currency Wire
            </button>
          </div>
        </div>

        <div className="card-body">
          <form onSubmit={handlePredict}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem', marginBottom: '1.25rem' }}>
              <div className="form-group">
                <label className="form-label">From Bank (Originator)</label>
                <input className="form-input" value={fromBank} onChange={(e) => setFromBank(e.target.value)} required />
              </div>

              <div className="form-group">
                <label className="form-label">To Bank (Beneficiary)</label>
                <input className="form-input" value={toBank} onChange={(e) => setToBank(e.target.value)} required />
              </div>

              <div className="form-group">
                <label className="form-label">Originating Account</label>
                <input className="form-input" value={originAccount} onChange={(e) => setOriginAccount(e.target.value)} required />
              </div>

              <div className="form-group">
                <label className="form-label">Beneficiary Account</label>
                <input className="form-input" value={beneficiaryAccount} onChange={(e) => setBeneficiaryAccount(e.target.value)} required />
              </div>

              <div className="form-group">
                <label className="form-label">Amount Paid ($)</label>
                <input className="form-input" type="number" step="0.01" value={amountPaid} onChange={(e) => setAmountPaid(e.target.value)} required />
              </div>

              <div className="form-group">
                <label className="form-label">Amount Received ($)</label>
                <input className="form-input" type="number" step="0.01" value={amountReceived} onChange={(e) => setAmountReceived(e.target.value)} required />
              </div>

              <div className="form-group">
                <label className="form-label">Payment Format</label>
                <select className="form-select" value={format} onChange={(e) => setFormat(e.target.value)}>
                  <option value="Wire">Wire</option>
                  <option value="ACH">ACH</option>
                  <option value="Cheque">Cheque</option>
                  <option value="Cash">Cash</option>
                  <option value="Bitcoin">Bitcoin</option>
                  <option value="Credit Card">Credit Card</option>
                  <option value="Reinvestment">Reinvestment</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Decision Threshold</label>
                <input className="form-input" type="number" step="0.05" min="0.10" max="0.90" value={threshold} onChange={(e) => setThreshold(e.target.value)} required />
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: '0.8125rem', color: '#64748b' }}>
                Note: 'Is Laundering' is strictly isolated and forbidden from inference payloads to prevent target leakage.
              </span>
              <button className="btn btn-primary" type="submit" disabled={predictLoading}>
                <Cpu size={16} />
                <span>{predictLoading ? 'Scoring...' : 'Score Transaction'}</span>
              </button>
            </div>
          </form>

          {predictError && (
            <div className="alert-banner warning" style={{ marginTop: '1.25rem' }}>
              <AlertTriangle size={18} />
              <div>{predictError}</div>
            </div>
          )}

          {predictResult && (
            <div style={{
              marginTop: '1.5rem',
              padding: '1.25rem',
              backgroundColor: predictResult.is_flagged ? '#fef2f2' : '#f0fdf4',
              border: `1px solid ${predictResult.is_flagged ? '#fecaca' : '#bbf7d0'}`,
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                {predictResult.is_flagged ? (
                  <ShieldAlert size={32} color="#dc2626" />
                ) : (
                  <CheckCircle size={32} color="#16a34a" />
                )}
                <div>
                  <div style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', color: predictResult.is_flagged ? '#991b1b' : '#166534' }}>
                    {predictResult.is_flagged ? 'Flagged for Compliance Investigation' : 'Low Suspicion Indicator (Standard Processing)'}
                  </div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#0f172a' }}>
                    {(predictResult.probability_score * 100).toFixed(1)}% Risk Score (Risk Tier: {predictResult.risk_level.toUpperCase()})
                  </div>
                  <div style={{ fontSize: '0.8125rem', color: '#475569', marginTop: '0.2rem' }}>
                    Evaluated against threshold {threshold}. Score indicates ranking order among trained AML features, not proof of guilt.
                  </div>
                </div>
              </div>

              <div style={{ textAlign: 'right' }}>
                <span className={`badge ${predictResult.is_flagged ? 'badge-risk-high' : 'badge-risk-low'}`} style={{ fontSize: '0.875rem', padding: '0.35rem 0.75rem' }}>
                  {predictResult.is_flagged ? 'Queue for SAR Review' : 'Auto-Cleared'}
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
