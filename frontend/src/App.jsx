import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { TransactionModal } from './components/TransactionModal';
import { OverviewPage } from './pages/OverviewPage';
import { AprioriPage } from './pages/AprioriPage';
import { AdaptiveAprioriPage } from './pages/AdaptiveAprioriPage';
import { UtilityMiningPage } from './pages/UtilityMiningPage';
import { RandomForestPage } from './pages/RandomForestPage';
import { api } from './services/api';

export function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedTransaction, setSelectedTransaction] = useState(null);
  const [healthData, setHealthData] = useState(null);

  useEffect(() => {
    api.getHealth()
      .then(data => setHealthData(data))
      .catch(() => setHealthData(null));
  }, []);

  return (
    <div className="app-container">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        healthData={healthData}
      />

      <main className="main-content">
        {activeTab === 'overview' && (
          <OverviewPage
            setActiveTab={setActiveTab}
            onInspectTransaction={(tx) => setSelectedTransaction(tx)}
          />
        )}

        {activeTab === 'apriori' && <AprioriPage />}

        {activeTab === 'adaptive' && <AdaptiveAprioriPage />}

        {activeTab === 'utility' && <UtilityMiningPage />}

        {activeTab === 'prediction' && <RandomForestPage />}
      </main>

      {/* Shared Transaction Investigation Experience */}
      {selectedTransaction && (
        <TransactionModal
          transaction={selectedTransaction}
          onClose={() => setSelectedTransaction(null)}
        />
      )}

      {/* Footer */}
      <footer style={{
        backgroundColor: '#ffffff',
        borderTop: '1px solid #e2e8f0',
        padding: '1.25rem 1.5rem',
        marginTop: 'auto',
        fontSize: '0.8125rem',
        color: '#64748b',
        textAlign: 'center'
      }}>
        FraudLens AML Operations Portal &copy; 2026 &mdash; Built on IBM Synthetic AML HI-Small Data.
        Developed for Academic IT Capstone Demonstration. All prediction scores represent investigation alert priorities, not proof of guilt.
      </footer>
    </div>
  );
}

export default App;
