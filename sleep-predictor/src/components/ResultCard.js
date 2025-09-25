import React from 'react';
import './ResultCard.css';

const ResultCard = ({ result }) => {
  // Handle case where result might be undefined or not an object
  if (!result || typeof result !== 'object') {
    return (
      <div className="result-container">
        <div className="cyber-error-card">
          <div className="error-icon">⚠️</div>
          <h3 className="error-title">Analysis Error</h3>
          <p className="error-message">Invalid result data received. Please try submitting the form again.</p>
          <button 
            className="cyber-retry-btn" 
            onClick={() => window.location.reload()}
          >
            Try Again
          </button>
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
          glowColor: '#ff073a',
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
          glowColor: '#ffff00',
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
          glowColor: '#39ff14',
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
          glowColor: '#00f5ff',
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
    <div className="cyber-result-container">
      {/* Animated Background Elements */}
      <div className="result-stars"></div>
      <div className="result-particles"></div>

      <div className={`cyber-result-card ${riskInfo.color}`}>
        <div className="result-header">
          <div className="result-icon-container">
            <div className="result-icon" style={{ filter: `drop-shadow(0 0 15px ${riskInfo.glowColor})` }}>
              {riskInfo.icon}
            </div>
          </div>
          <div className="result-title-section">
            <h2 className="cyber-result-title">{riskInfo.title}</h2>
            <p className="result-description">{riskInfo.description}</p>
          </div>
        </div>

        <div className="result-content">
          <div className="prediction-details">
            <div className="prediction-main">
              <h4 className="section-heading">Assessment Result:</h4>
              <div className="result-value-container">
                <p className="result-value">{predictionText}</p>
                <p className="risk-description">{riskDescription}</p>
              </div>
            </div>
            
            <div className="confidence-metrics">
              <h4 className="section-heading">Confidence Analysis:</h4>
              <div className="confidence-bars">
                <div className="confidence-item">
                  <span className="confidence-label">No Risk</span>
                  <div className="cyber-confidence-bar">
                    <div 
                      className="confidence-fill no-risk" 
                      style={{width: `${safeConfidence.no_risk * 100}%`}}
                    ></div>
                  </div>
                  <span className="confidence-value">{formatPercentage(safeConfidence.no_risk * 100)}%</span>
                </div>
                
                <div className="confidence-item">
                  <span className="confidence-label">Sleep Deprivation Risk</span>
                  <div className="cyber-confidence-bar">
                    <div 
                      className="confidence-fill risk" 
                      style={{width: `${safeConfidence.risk * 100}%`}}
                    ></div>
                  </div>
                  <span className="confidence-value">{formatPercentage(safeConfidence.risk * 100)}%</span>
                </div>
              </div>
              
              <div className="risk-percentage">
                <div className="overall-risk-container">
                  <span className="risk-label">Overall Risk Level:</span>
                  <span className="risk-value">{formatPercentage(riskPercentage)}%</span>
                </div>
              </div>
            </div>
          </div>

          {riskInfo.recommendations && riskInfo.recommendations.length > 0 && (
            <div className="recommendations-section">
              <h4 className="section-heading">Personalized Recommendations:</h4>
              <ul className="cyber-recommendations-list">
                {riskInfo.recommendations.map((rec, index) => (
                  <li key={index} className="cyber-recommendation-item">
                    <span className="rec-bullet">⚡</span>
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="result-footer">
          <div className="disclaimer">
            <div className="disclaimer-icon">⚕️</div>
            <div className="disclaimer-text">
              <p>
                <strong>Medical Disclaimer:</strong> This AI-powered assessment is for informational purposes only 
                and should not replace professional medical advice. Please consult with a qualified healthcare provider 
                for proper diagnosis and treatment of sleep disorders.
              </p>
            </div>
          </div>
          
          <div className="next-steps">
            <button 
              className="cyber-btn print-btn" 
              onClick={() => window.print()}
              type="button"
            >
              <span className="btn-icon">📋</span>
              Print Results
            </button>
            <button 
              className="cyber-btn email-btn" 
              onClick={() => {
                const subject = encodeURIComponent('Sleep Health Assessment Results');
                const body = encodeURIComponent(
                  `My sleep assessment shows: ${predictionText}\nRisk Level: ${riskLevel}\nConfidence: ${formatPercentage(riskPercentage)}%`
                );
                window.open(`mailto:?subject=${subject}&body=${body}`);
              }}
              type="button"
            >
              <span className="btn-icon">📧</span>
              Email Results
            </button>
            <button 
              className="cyber-btn new-assessment-btn" 
              onClick={() => window.location.reload()}
              type="button"
            >
              <span className="btn-icon">🔄</span>
              New Assessment
            </button>
          </div>
        </div>
      </div>

      {/* Enhanced Health Tips */}
      <div className="cyber-health-tips-card">
        <div className="tips-header">
          <h3 className="tips-title">💡 Sleep Optimization Guide</h3>
          <div className="tips-subtitle">Science-backed strategies for better sleep</div>
        </div>
        <div className="tips-grid">
          <div className="cyber-tip-item">
            <div className="tip-icon-container">
              <span className="tip-icon">🌙</span>
            </div>
            <div className="tip-content">
              <h5>Consistent Schedule</h5>
              <p>Go to bed and wake up at the same time every day, even weekends</p>
            </div>
          </div>
          <div className="cyber-tip-item">
            <div className="tip-icon-container">
              <span className="tip-icon">🌡️</span>
            </div>
            <div className="tip-content">
              <h5>Optimal Environment</h5>
              <p>Keep bedroom cool (60-67°F), dark, and quiet</p>
            </div>
          </div>
          <div className="cyber-tip-item">
            <div className="tip-icon-container">
              <span className="tip-icon">📱</span>
            </div>
            <div className="tip-content">
              <h5>Digital Detox</h5>
              <p>Avoid screens 1-2 hours before bedtime for better sleep quality</p>
            </div>
          </div>
          <div className="cyber-tip-item">
            <div className="tip-icon-container">
              <span className="tip-icon">🏃</span>
            </div>
            <div className="tip-content">
              <h5>Regular Exercise</h5>
              <p>Daily physical activity improves sleep, but avoid vigorous exercise close to bedtime</p>
            </div>
          </div>
          <div className="cyber-tip-item">
            <div className="tip-icon-container">
              <span className="tip-icon">☕</span>
            </div>
            <div className="tip-content">
              <h5>Caffeine Timing</h5>
              <p>Avoid caffeine after 2 PM as it can stay in your system 6-8 hours</p>
            </div>
          </div>
          <div className="cyber-tip-item">
            <div className="tip-icon-container">
              <span className="tip-icon">🧘</span>
            </div>
            <div className="tip-content">
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