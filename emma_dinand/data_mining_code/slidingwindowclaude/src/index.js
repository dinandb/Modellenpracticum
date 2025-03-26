import React from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App';

// Get the root element from index.html
const container = document.getElementById('root');
const root = createRoot(container);

// Render your App component into the root element
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

console.log("React application has started rendering");