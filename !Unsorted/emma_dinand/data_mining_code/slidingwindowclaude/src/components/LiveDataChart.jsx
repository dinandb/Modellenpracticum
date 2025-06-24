// File: src/components/LiveDataChart.jsx
import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useWebSocket } from '../hooks/useWebSocket';
import SafetyStatusPanel from './SafetyStatusPanel';
import '../styles/LiveDataChart.css';

const LiveDataChart = () => {
  const [data, setData] = useState([]);
  const [windowSize, setWindowSize] = useState(20);
  const [isPaused, setIsPaused] = useState(false);
  const [safetyStatus, setSafetyStatus] = useState({
    isSafe: false,
    amountToGoForSafe: 0
  });
  const { connected, messages, sendMessage } = useWebSocket('ws://localhost:8000/ws');

  useEffect(() => {
    if (messages && messages.length > 0 && !isPaused) {
      const latestMessage = messages[messages.length - 1];
      try {
        const parsedData = JSON.parse(latestMessage);
        
        // Update safety status with latest data
        setSafetyStatus({
          isSafe: parsedData.is_safe || false,
          amountToGoForSafe: parsedData.amount_to_go_for_safe || 0
        });
        
        setData(prevData => {
          const newData = [...prevData, {

            value: parsedData.value,

            is_safe: parsedData.is_safe || false,
            amount_to_go_for_safe: parsedData.amount_to_go_for_safe || 0
          }];
          
          // Keep only the latest 'windowSize' data points
          if (newData.length > windowSize) {
            return newData.slice(newData.length - windowSize);
          }
          return newData;
        });
      } catch (error) {
        console.error('Failed to parse message:', error);
      }
    }
  }, [messages, windowSize, isPaused]);

  const handleWindowSizeChange = (e) => {
    const newSize = parseInt(e.target.value);
    if (!isNaN(newSize) && newSize > 0) {
      setWindowSize(newSize);
      // If the new window size is smaller than current data array, trim the data
      if (newSize < data.length) {
        setData(data.slice(data.length - newSize));
      }
    }
  };

  const handleRequestData = () => {
    if (connected && !isPaused) {
      sendMessage(JSON.stringify({ action: 'request_data' }));
    }
  };

  const togglePause = () => {
    setIsPaused(!isPaused);
  };

  return (
    <div className="chart-container">
      <div className="chart-controls">
        <div className="control-group">
          <label htmlFor="windowSize">Window Size:</label>
          <input
            id="windowSize"
            type="number"
            min="5"
            max="100"
            value={windowSize}
            onChange={handleWindowSizeChange}
          />
        </div>
        <div className="button-group">
          <button 
            className={`pause-btn ${isPaused ? 'paused' : ''}`}
            onClick={togglePause}
          >
            {isPaused ? 'Resume' : 'Pause'}
          </button>
          <button 
            className="request-data-btn"
            onClick={handleRequestData}
            disabled={!connected || isPaused}
          >
            Request Data
          </button>
        </div>
      </div>
      
      {/* Safety Status Panel */}
      <SafetyStatusPanel 
        isSafe={safetyStatus.isSafe} 
        amountToGoForSafe={safetyStatus.amountToGoForSafe} 
      />
      
      <div className="chart-wrapper">
        <ResponsiveContainer width="100%" height={400}>
          <LineChart
            data={data}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line 
              type="linear" 
              dataKey="value" 
              stroke="#8884d8" 
              activeDot={{ r: 8 }}
              isAnimationActive={false}
              dot={{ strokeWidth: 2, r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      <div className="data-stats">
        <p>Data Points: {data.length} / {windowSize}</p>
        {data.length > 0 && (
          <p>Latest Value: {data[data.length - 1].value}</p>
        )}
        <p className={`status-text ${isPaused ? 'paused' : 'active'}`}>
          Status: {isPaused ? 'Paused' : 'Active'}
        </p>
      </div>
    </div>
  );
};

export default LiveDataChart;