import React, { useState } from 'react';
import { X, ShieldAlert, CheckCircle2, AlertTriangle, Info, ArrowRight, Activity } from 'lucide-react';
import { api } from '../services/api';

export function TransactionModal({ transaction, onClose }) {
  const [scoring, setScoring] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [scoreError, setScoreError] = useState(null);

  if (!transaction) return null;

  const handleScoreTransaction = async () => {
    setScoring(true);
    setScoreError(null);
    try {
      const payload = [{
        Timestamp: transaction.Timestamp || '2022/09/01 12:00',
        'From Bank': parseInt(transaction['From Bank'] || 0, 10),
        'To Bank': parseInt(transaction['To Bank'] || 0, 10),
        Account: String(transaction.Account || 'ACC_UNKNOWN'),
        'Account.1': String(transaction['Account.1'] || transaction.Account || 'ACC_UNKNOWN'),
        'Amount Received': parseFloat(transaction['Amount Received'] || 0.0),
        'Amount Paid': parseFloat(transaction['Amount Paid'] || 0.0),
        'Receiving Currency': String(transaction['Receiving Currency'] || 'US Dollar'),
        'Payment Currency': String(transaction['Payment Currency'] || 'US Dollar'),
        'Payment Format': String(transaction['Payment Format'] || 'Wire'),
      }];

      const res = await api.predictTransactions(payload, 0.40);
      if (res && res.predictions && res.predictions.length > 0) {
        setPredictionResult(res.predictions[0]);
      }
    } catch (err) {
      setScoreError(err.message || 'Scoring failed');
    } finally {
      setScoring(false);
    }
  };

  const isCrossCurrency = transaction['Payment Currency'] !== transaction['Receiving Currency'];
  const isInterBank = transaction['From Bank'] !== transaction['To Bank'];
  const amount = parseFloat(transaction['Amount Paid'] || 0);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <Activity size={20} color="#0284c7" />
            <h3 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Compliance Transaction Investigation</h3>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#64748b' }}>
            <X size={20} />
          </button>
        </div>

        <div className="modal-body">
          {/* Top Notice */}
          <div className="alert-banner" style={{ marginBottom: '1.25rem' }}>
            <Info size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
            <div>
              <strong>Review Guidance:</strong> Transactions flagged by models or associated with frequent AML patterns
              represent investigative priorities for compliance officers, <em>not proof of money laundering or criminal intent</em>.
            </div>
          </div>

          {/* Actual Dataset Fields */}
          <div style={{ marginBottom: '1.5rem' }}>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#334155', marginBottom: '0.75rem' }}>
              Actual Transaction Fields (Verified Schema)
            </h4>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '0.75rem',
              backgroundColor: '#f8fafc',
              padding: '1rem',
              borderRadius: '8px',
              border: '1px solid #e2e8f0',
              fontSize: '0.875rem'
            }}>
              <div><span style={{ color: '#64748b' }}>Timestamp:</span> <strong>{transaction.Timestamp}</strong></div>
              <div><span style={{ color: '#64748b' }}>Payment Format:</span> <strong>{transaction['Payment Format']}</strong></div>
              <div><span style={{ color: '#64748b' }}>From Bank:</span> <strong>Bank {transaction['From Bank']}</strong></div>
              <div><span style={{ color: '#64748b' }}>To Bank:</span> <strong>Bank {transaction['To Bank']}</strong></div>
              <div><span style={{ color: '#64748b' }}>Originating Account:</span> <code>{transaction.Account}</code></div>
              <div><span style={{ color: '#64748b' }}>Beneficiary Account:</span> <code>{transaction['Account.1'] || transaction.Account}</code></div>
              <div><span style={{ color: '#64748b' }}>Amount Paid:</span> <strong>${Number(transaction['Amount Paid']).toLocaleString()}</strong> ({transaction['Payment Currency']})</div>
              <div><span style={{ color: '#64748b' }}>Amount Received:</span> <strong>${Number(transaction['Amount Received']).toLocaleString()}</strong> ({transaction['Receiving Currency']})</div>
            </div>
          </div>

          {/* Model Prediction Section */}
          <div style={{ marginBottom: '1.5rem', padding: '1.25rem', backgroundColor: '#ffffff', border: '1px solid #e2e8f0', borderRadius: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
              <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#0f172a' }}>
                Supervised Model Classification (Random Forest)
              </h4>
              {!predictionResult && (
                <button
                  className="btn btn-primary"
                  onClick={handleScoreTransaction}
                  disabled={scoring}
                  style={{ fontSize: '0.8125rem', padding: '0.35rem 0.75rem' }}
                >
                  {scoring ? 'Evaluating...' : 'Score with Random Forest'}
                </button>
              )}
            </div>

            {scoreError && (
              <div style={{ color: '#b91c1c', fontSize: '0.85rem', backgroundColor: '#fee2e2', padding: '0.5rem', borderRadius: '4px' }}>
                {scoreError}
              </div>
            )}

            {predictionResult ? (
              <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                <div style={{
                  padding: '0.75rem 1.25rem',
                  borderRadius: '8px',
                  backgroundColor: predictionResult.is_flagged ? '#fee2e2' : '#dcfce7',
                  border: `1px solid ${predictionResult.is_flagged ? '#fca5a5' : '#86efac'}`,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem'
                }}>
                  {predictionResult.is_flagged ? (
                    <ShieldAlert size={24} color="#991b1b" />
                  ) : (
                    <CheckCircle2 size={24} color="#166534" />
                  )}
                  <div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: predictionResult.is_flagged ? '#991b1b' : '#166534', textTransform: 'uppercase' }}>
                      {predictionResult.is_flagged ? 'Flagged for Review' : 'Low Suspicion Indicator'}
                    </div>
                    <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#0f172a' }}>
                      {(predictionResult.probability_score * 100).toFixed(1)}% Score
                    </div>
                  </div>
                </div>

                <div style={{ fontSize: '0.875rem', color: '#475569', lineHeight: 1.5 }}>
                  Decision Threshold: <strong>0.40</strong>. Model assigns risk rating: <strong>{predictionResult.risk_level.toUpperCase()}</strong>.
                  The score indicates priority order for review against trained AML features.
                </div>
              </div>
            ) : (
              <div style={{ fontSize: '0.85rem', color: '#64748b' }}>
                Click "Score with Random Forest" to evaluate this transaction's risk score against the trained model without target leakage.
              </div>
            )}
          </div>

          {/* Plain Language Summary & Context */}
          <div style={{ marginBottom: '1rem' }}>
            <h4 style={{ fontSize: '0.95rem', fontWeight: 600, color: '#0f172a', marginBottom: '0.5rem' }}>
              Analytical Evidence & Characteristics
            </h4>
            <ul style={{ paddingLeft: '1.25rem', fontSize: '0.875rem', color: '#334155', lineHeight: 1.6 }}>
              <li>
                <strong>Transfer Route:</strong> {isInterBank ? 'Interbank transfer between two separate financial institutions.' : 'Internal book transfer within the same banking institution.'}
              </li>
              <li>
                <strong>Currency Relationship:</strong> {isCrossCurrency ? 'Cross-currency conversion transaction; higher currency exchange monitoring applies.' : 'Domestic/Single-currency transaction.'}
              </li>
              <li>
                <strong>Financial Volume Tier:</strong> {amount >= 50000 ? 'High financial volume exceeding $50,000.' : amount >= 10000 ? 'Moderate financial volume between $10,000 and $50,000.' : 'Standard volume under $10,000.'}
              </li>
            </ul>
          </div>

          {/* Actionable Next Step for Compliance */}
          <div style={{ backgroundColor: '#f1f5f9', borderLeft: '4px solid #0f766e', padding: '0.875rem 1rem', borderRadius: '4px', fontSize: '0.85rem' }}>
            <strong style={{ color: '#0f766e' }}>Recommended Action:</strong>{' '}
            Check originator profile and purpose of transfer. If amount matches known customer monthly activity, mark as reviewed-legitimate. If unexpected, request KYC verification documentation.
          </div>
        </div>

        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={onClose}>
            Close Investigation
          </button>
        </div>
      </div>
    </div>
  );
}
