import React from 'react';
import './App.css';
import LiveDataChart from './components/LiveDataChart';
import StatusIndicator from './components/StatusIndicator';

function App() {
  return (
    <div className="App">
      <header>
        <h1>Real-time Data Monitor</h1>
      </header>
      <main>
        <StatusIndicator />
        <LiveDataChart />
      </main>
      <footer>
        <p>React Sliding Window App - Real-time Data Visualization</p>
      </footer>
    </div>
  );
}

export default App;