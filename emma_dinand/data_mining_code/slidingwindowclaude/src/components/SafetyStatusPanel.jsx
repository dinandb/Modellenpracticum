// File: src/components/SafetyStatusPanel.jsx
import React from 'react';
import '../styles/SafetyStatusPanel.css';

const SafetyStatusPanel = ({ isSafe, amountToGoForSafe }) => {
  return (
    <div className={`safety-panel ${isSafe ? 'safe' : 'unsafe'}`}>
      <h3 className="safety-title">System Status</h3>
      <div className="safety-indicator">
        <div className={`status-circle ${isSafe ? 'safe' : 'unsafe'}`}></div>
        <span className="status-text">{isSafe ? 'SAFE' : 'UNSAFE'}</span>
      </div>
      <p className="safety-message">
        {isSafe 
          ? "System is operating within safe parameters"
          : `Amount of timesteps to go before a QP: ${amountToGoForSafe}`
        }
      </p>
    </div>
  );
};

export default SafetyStatusPanel;