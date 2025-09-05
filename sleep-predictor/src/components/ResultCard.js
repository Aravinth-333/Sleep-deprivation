// src/components/ResultCard.js
import React from 'react';

const ResultCard = ({ result }) => {
  // Handle case where result might be undefined or not an object
  if (!result || typeof result !== 'object') {
    return (
      <div className="result-container">
        <div className="error-card">
          <h3>Error Displaying Results</h3>
          <p>Invalid result data received. Please try submitting the form again.</p>
        </div>
      </div>
    );
  }

  // Extract data from the result object with fallbacks
  const {
    prediction = 0,
    predictionText = 'No prediction available',
    riskLevel = 'unknown',
    riskDescription = 'No description available',
    confidence = { no_risk: 0, risk: 0 },
    riskPercentage = 0
  } = result;

  // Ensure confidence is properly structured
  const safeConfidence = {
    no_risk: confidence && typeof confidence.no_risk === 'number' ? confidence.no_risk : 0,
    risk: confidence && typeof confidence.risk === 'number' ? confidence.risk : 0
  };

  const getRiskInfo = (level) => {
    const levelStr = String(level).toLowerCase();
    
    switch (levelStr) {
      case 'high':
        return {
          icon: '🚨',
          color: 'high-risk',
          title: 'High Sleep Deprivation Risk',
          description: 'Immediate attention recommended. Consider consulting a healthcare professional.',
          recommendations: [
            'Consult with a sleep specialist or healthcare provider immediately',
            'Implement strict sleep hygiene practices',
            'Consider a comprehensive sleep study evaluation',
            'Review current medications and lifestyle with your doctor',
            'Consider cognitive behavioral therapy for insomnia (CBT-I)'
          ]
        };
      case 'moderate':
        return {
          icon: '⚠️',
          color: 'moderate-risk',
          title: 'Moderate Sleep Deprivation Risk',
          description: 'Some concern identified. Lifestyle modifications recommended.',
          recommendations: [
            'Establish and maintain a consistent sleep schedule',
            'Reduce caffeine intake, especially after 2 PM',
            'Limit screen time 1-2 hours before bedtime',
            'Increase physical activity during daytime hours',
            'Practice stress management and relaxation techniques',
            'Monitor sleep patterns for improvement'
          ]
        };
      case 'low':
        return {
          icon: '✅',
          color: 'low-risk',
          title: 'Low Sleep Deprivation Risk',
          description: 'Good sleep health indicators detected. Continue current healthy habits.',
          recommendations: [
            'Continue maintaining your current sleep habits',
            'Keep monitoring sleep quality regularly',
            'Maintain healthy lifestyle choices',
            'Stay aware of factors that could impact sleep',
            'Consider periodic sleep health assessments'
          ]
        };
      default:
        return {
          icon: '📊',
          color: 'neutral',
          title: 'Sleep Assessment Complete',
          description: 'Analysis completed successfully.',
          recommendations: [
            'Review your sleep patterns regularly',
            'Maintain healthy sleep habits',
            'Consider retaking the assessment periodically'
          ]
        };
    }
  };

  const riskInfo = getRiskInfo(riskLevel);

  // Safe number formatting
  const formatPercentage = (num) => {
    const safeNum = typeof num === 'number' ? num : 0;
    return safeNum.toFixed(1);
  };

  return (
    <div className="result-container">
      <div className={`result-card ${riskInfo.color}`}>
        <div className="result-header">
          <div className="result-icon">{riskInfo.icon}</div>
          <div className="result-title-section">
            <h2 className="result-title">{riskInfo.title}</h2>
            <p className="result-description">{riskInfo.description}</p>
          </div>
        </div>

        <div className="result-content">
          <div className="prediction-details">
            <div className="prediction-main">
              <h4>Assessment Result:</h4>
              <p className="result-value">{predictionText}</p>
              <p className="risk-description">{riskDescription}</p>
            </div>
            
            <div className="confidence-metrics">
              <h4>Confidence Scores:</h4>
              <div className="confidence-bars">
                <div className="confidence-item">
                  <span className="confidence-label">No Risk</span>
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill no-risk" 
                      style={{width: `${safeConfidence.no_risk * 100}%`}}
                    ></div>
                  </div>
                  <span className="confidence-value">{formatPercentage(safeConfidence.no_risk * 100)}%</span>
                </div>
                
                <div className="confidence-item">
                  <span className="confidence-label">Sleep Deprivation Risk</span>
                  <div className="confidence-bar">
                    <div 
                      className="confidence-fill risk" 
                      style={{width: `${safeConfidence.risk * 100}%`}}
                    ></div>
                  </div>
                  <span className="confidence-value">{formatPercentage(safeConfidence.risk * 100)}%</span>
                </div>
              </div>
              
              <div className="risk-percentage">
                <p><strong>Overall Risk Level: {formatPercentage(riskPercentage)}%</strong></p>
              </div>
            </div>
          </div>

          {riskInfo.recommendations && riskInfo.recommendations.length > 0 && (
            <div className="recommendations-section">
              <h4>Personalized Recommendations:</h4>
              <ul className="recommendations-list">
                {riskInfo.recommendations.map((rec, index) => (
                  <li key={index} className="recommendation-item">
                    <span className="rec-bullet">•</span>
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="result-footer">
          <div className="disclaimer">
            <p>
              <strong>Medical Disclaimer:</strong> This AI-powered assessment is for informational purposes only 
              and should not replace professional medical advice. Please consult with a qualified healthcare provider 
              for proper diagnosis and treatment of sleep disorders.
            </p>
          </div>
          
          <div className="next-steps">
            <button 
              className="btn btn-outline" 
              onClick={() => window.print()}
              type="button"
            >
              📋 Print Results
            </button>
            <button 
              className="btn btn-outline" 
              onClick={() => {
                const subject = encodeURIComponent('Sleep Health Assessment Results');
                const body = encodeURIComponent(
                  `My sleep assessment shows: ${predictionText}\nRisk Level: ${riskLevel}\nConfidence: ${formatPercentage(riskPercentage)}%`
                );
                window.open(`mailto:?subject=${subject}&body=${body}`);
              }}
              type="button"
            >
              📧 Email Results
            </button>
            <button 
              className="btn btn-secondary" 
              onClick={() => window.location.reload()}
              type="button"
            >
              🔄 New Assessment
            </button>
          </div>
        </div>
      </div>

      {/* Additional Health Tips */}
      <div className="health-tips-card">
        <h3>💡 Sleep Health Tips</h3>
        <div className="tips-grid">
          <div className="tip-item">
            <span className="tip-icon">🌙</span>
            <div>
              <h5>Consistent Schedule</h5>
              <p>Go to bed and wake up at the same time every day, even weekends</p>
            </div>
          </div>
          <div className="tip-item">
            <span className="tip-icon">🌡️</span>
            <div>
              <h5>Optimal Environment</h5>
              <p>Keep bedroom cool (60-67°F), dark, and quiet</p>
            </div>
          </div>
          <div className="tip-item">
            <span className="tip-icon">📱</span>
            <div>
              <h5>Digital Detox</h5>
              <p>Avoid screens 1-2 hours before bedtime for better sleep quality</p>
            </div>
          </div>
          <div className="tip-item">
            <span className="tip-icon">🏃</span>
            <div>
              <h5>Regular Exercise</h5>
              <p>Daily physical activity improves sleep, but avoid vigorous exercise close to bedtime</p>
            </div>
          </div>
          <div className="tip-item">
            <span className="tip-icon">☕</span>
            <div>
              <h5>Caffeine Timing</h5>
              <p>Avoid caffeine after 2 PM as it can stay in your system 6-8 hours</p>
            </div>
          </div>
          <div className="tip-item">
            <span className="tip-icon">🧘</span>
            <div>
              <h5>Stress Management</h5>
              <p>Practice relaxation techniques like meditation or deep breathing</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultCard;