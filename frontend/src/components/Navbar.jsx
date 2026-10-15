import React from 'react';
import { ShieldCheck, BarChart3, GitFork, Gem, Cpu, Database } from 'lucide-react';

export function Navbar({ activeTab, setActiveTab, healthData }) {
  const navItems = [
    { id: 'overview', label: 'Bank Overview', icon: BarChart3 },
    { id: 'apriori', label: 'Standard Patterns', icon: GitFork },
    { id: 'adaptive', label: 'Adaptive Discovery', icon: ShieldCheck },
    { id: 'utility', label: 'High-Utility Mining', icon: Gem },
    { id: 'prediction', label: 'Transaction Predictions', icon: Cpu },
  ];

  return (
    <header className="navbar">
      <div className="navbar-inner">
        <div className="brand-section">
          <div className="brand-logo">FL</div>
          <div>
            <div className="brand-title">FraudLens Operations</div>
            <div className="brand-subtitle">AML Pattern Mining & Predictive Intelligence</div>
          </div>
        </div>

        <nav className="nav-links">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`nav-tab ${isActive ? 'active' : ''}`}
              >
                <Icon size={16} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          {healthData && (
            <span className="badge badge-info" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <Database size={12} />
              <span>{healthData.dataset_available ? '5.07M Records' : 'No Dataset'}</span>
            </span>
          )}
          <span className="badge badge-risk-low" style={{ fontSize: '0.75rem' }}>
            ● System Active
          </span>
        </div>
      </div>
    </header>
  );
}
