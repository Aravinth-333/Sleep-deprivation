// src/App.js
import React, { useState } from 'react';
import InputForm from './components/InputForm';
import ResultCard from './components/ResultCard';
import './index.css';

const App = () => {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = async (inputData) => {
    setLoading(true);
    setError(null);
    
    try {
      console.log('Sending data to backend:', inputData);
      
      // Make API call to Flask backend
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputData)
      });

      const data = await response.json();
      console.log('Received response:', data);

      if (response.ok && data.success) {
        // Format the result for the ResultCard component
        const formattedResult = {
          prediction: data.prediction,
          predictionText: data.prediction_text,
          riskLevel: data.risk_level,
          riskDescription: data.risk_description,
          confidence: data.confidence,
          riskPercentage: data.risk_percentage
        };
        
        setPrediction(formattedResult);
      } else {
        throw new Error(data.error || 'Prediction failed');
      }
    } catch (error) {
      console.error('Prediction error:', error);
      setError(error.message || 'Failed to connect to prediction service');
    } finally {
      setLoading(false);
    }
  };

  const resetApp = () => {
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="app">
      <div className="app-container">
        {error && (
          <div className="error-container">
            <div className="error-card">
              <h3>⚠️ Prediction Error</h3>
              <p>{error}</p>
              <button onClick={resetApp} className="btn btn-secondary">
                Try Again
              </button>
            </div>
          </div>
        )}
        
        {!prediction && !loading && (
          <InputForm onPredict={handlePredict} />
        )}
        
        {loading && (
          <div className="loading-container">
            <div className="loading-card">
              <div className="loading-spinner"></div>
              <h3>Analyzing Your Sleep Health Data...</h3>
              <p>Processing your information through our AI model...</p>
            </div>
          </div>
        )}
        
        {prediction && !loading && (
          <div className="result-section">
            <ResultCard result={prediction} />
            <div className="action-buttons">
              <button onClick={resetApp} className="btn btn-primary">
                Take Another Assessment
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;