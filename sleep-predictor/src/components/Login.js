import React, { useState } from 'react';
import './Login.css';

const Login = ({ onLogin }) => {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    // Simulate API call
    setTimeout(() => {
      if (formData.email && formData.password) {
        // Mock successful login
        if (onLogin) {
          onLogin({
            email: formData.email,
            name: formData.email.split('@')[0],
            id: Date.now()
          });
        }
      } else {
        setError('Please fill in all fields');
      }
      setLoading(false);
    }, 1500);
  };

  const toggleMode = () => {
    setIsLogin(!isLogin);
    setError('');
    setFormData({ email: '', password: '' });
  };

  return (
    <div className="cyber-login-container">
      {/* Animated Background */}
      <div className="login-stars"></div>
      <div className="login-particles"></div>
      <div className="login-moon"></div>

      <div className="login-wrapper">
        {/* Header */}
        <div className="login-header">
          <div className="login-logo">
            <div className="logo-icon">🌙</div>
            <h1 className="logo-text">SLEEP ANALYZER</h1>
          </div>
          <div className="login-tagline">Advanced Sleep Deprivation Detection System</div>
        </div>

        {/* Login Form */}
        <div className="login-form-container">
          <div className="form-header">
            <h2 className="form-title">
              {isLogin ? 'ACCESS TERMINAL' : 'CREATE ACCOUNT'}
            </h2>
            <div className="form-subtitle">
              {isLogin ? 'Enter your credentials to continue' : 'Join the sleep optimization network'}
            </div>
          </div>

          <div onSubmit={handleSubmit} className="cyber-form">
            <div className="input-group">
              <label htmlFor="email" className="cyber-label">
                <span className="label-icon">📧</span>
                Email Address
              </label>
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleInputChange}
                placeholder="user@sleepanalyzer.com"
                className="cyber-input"
                required
              />
            </div>

            <div className="input-group">
              <label htmlFor="password" className="cyber-label">
                <span className="label-icon">🔒</span>
                Password
              </label>
              <input
                type="password"
                id="password"
                name="password"
                value={formData.password}
                onChange={handleInputChange}
                placeholder="Enter secure password"
                className="cyber-input"
                required
              />
            </div>

            {error && (
              <div className="error-message">
                <span className="error-icon">⚠️</span>
                {error}
              </div>
            )}

            <button
              onClick={handleSubmit}
              className={`cyber-submit-btn ${loading ? 'loading' : ''}`}
              disabled={loading}
            >
              {loading ? (
                <div className="loading-spinner"></div>
              ) : (
                <>
                  <span className="btn-text">
                    {isLogin ? 'INITIALIZE ACCESS' : 'CREATE ACCOUNT'}
                  </span>
                  <span className="btn-arrow">→</span>
                </>
              )}
            </button>
          </div>

          <div className="form-footer">
            <div className="mode-toggle">
              <span>
                {isLogin ? "Don't have an account?" : "Already have an account?"}
              </span>
              <button
                type="button"
                onClick={toggleMode}
                className="toggle-btn"
              >
                {isLogin ? 'Create Account' : 'Sign In'}
              </button>
            </div>

            {isLogin && (
              <div className="quick-access">
                <button
                  type="button"
                  onClick={() => {
                    setFormData({ email: 'demo@user.com', password: 'demo123' });
                  }}
                  className="demo-btn"
                >
                  Use Demo Account
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Features */}
        <div className="features-section">
          <div className="feature-item">
            <div className="feature-icon">🧠</div>
            <div className="feature-text">
              <h4>AI-Powered Analysis</h4>
              <p>Advanced machine learning algorithms</p>
            </div>
          </div>
          <div className="feature-item">
            <div className="feature-icon">📊</div>
            <div className="feature-text">
              <h4>Weekly Predictions</h4>
              <p>Track patterns and get insights</p>
            </div>
          </div>
          <div className="feature-item">
            <div className="feature-icon">🔒</div>
            <div className="feature-text">
              <h4>Secure & Private</h4>
              <p>Your data is encrypted and protected</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;