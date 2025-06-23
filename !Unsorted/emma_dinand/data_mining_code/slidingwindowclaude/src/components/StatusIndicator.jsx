// File: src/components/StatusIndicator.jsx
import React from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import '../styles/StatusIndicator.css';

const StatusIndicator = () => {
  const { connected, connecting, error } = useWebSocket('ws://localhost:8000/ws');

  return (
    <div className="status-indicator">
      <div className={`indicator ${connected ? 'connected' : connecting ? 'connecting' : 'disconnected'}`}></div>
      <span className="status-text">
        {connected 
          ? 'Connected to server' 
          : connecting 
            ? 'Connecting...' 
            : error 
              ? `Disconnected: ${error}` 
              : 'Disconnected'
        }
      </span>
    </div>
  );
};

export default StatusIndicator;