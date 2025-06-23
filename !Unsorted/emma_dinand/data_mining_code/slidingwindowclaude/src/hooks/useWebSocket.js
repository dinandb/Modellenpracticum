// File: src/hooks/useWebSocket.js
import { useState, useEffect, useRef, useCallback } from 'react';

export const useWebSocket = (url) => {
  const [connected, setConnected] = useState(false);
  const [connecting, setConnecting] = useState(false);
  const [messages, setMessages] = useState([]);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);

  // Establish connection
  useEffect(() => {
    const connectWebSocket = () => {
      setConnecting(true);
      setError(null);
      
      const ws = new WebSocket(url);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setConnected(true);
        setConnecting(false);
      };
      
      ws.onclose = (event) => {
        console.log('WebSocket disconnected', event);
        setConnected(false);
        setConnecting(false);
        
        // Try to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };
      
      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError('Connection failed');
        setConnecting(false);
        setConnected(false);
      };
      
      ws.onmessage = (event) => {
        console.log('Message received:', event.data);
        setMessages(prev => [...prev, event.data]);
      };
      
      wsRef.current = ws;
      
      return () => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.close();
        }
      };
    };
    
    const cleanup = connectWebSocket();
    
    return cleanup;
  }, [url]);
  
  // Send message function
  const sendMessage = useCallback((message) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(message);
      return true;
    }
    return false;
  }, []);
  
  return { connected, connecting, messages, error, sendMessage };
};