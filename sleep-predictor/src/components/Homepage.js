import React, { useState, useEffect } from 'react';
import './HeadspaceHomepage.css';

const HeadspaceHomepage = () => {
  const [activeFeature, setActiveFeature] = useState(0);
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveFeature((prev) => (prev + 1) % 3);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const features = [
    {
      title: "Sleep Stories",
      description: "Soothing narrations designed to help you fall asleep.",
      icon: "🌙"
    },
    {
      title: "Sleep Sounds",
      description: "Calming sounds to create the perfect sleep environment.",
      icon: "🔊"
    },
    {
      title: "Sleep Meditations",
      description: "Guided practices to quiet your mind before sleep.",
      icon: "🧘"
    }
  ];

  return (
    <div className="headspace-homepage">
      {/* Navigation */}
      <nav className="navbar">
        <div className="nav-container">
          <div className="nav-logo">
            <span className="logo-icon">💤</span>
            <span className="logo-text">Sleep by Headspace</span>
          </div>
          <div className={`nav-menu ${isMenuOpen ? 'active' : ''}`}>
            <a href="#features" className="nav-link">Features</a>
            <a href="#how-it-works" className="nav-link">How It Works</a>
            <a href="#testimonials" className="nav-link">Testimonials</a>
            <a href="#pricing" className="nav-link">Pricing</a>
            <div className="nav-buttons">
              <button className="btn-login">Log In</button>
              <button className="btn-try-free">Try Free</button>
            </div>
          </div>
          <div className="menu-toggle" onClick={() => setIsMenuOpen(!isMenuOpen)}>
            <span className="bar"></span>
            <span className="bar"></span>
            <span className="bar"></span>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero">
        <div className="hero-content">
          <h1>Sleep better. Wake up happier.</h1>
          <p>Fall asleep faster and wake up feeling refreshed with our science-backed sleep solutions.</p>
          <div className="hero-buttons">
            <button className="btn-primary">Start Sleeping Better</button>
            <button className="btn-secondary">Learn More</button>
          </div>
        </div>
        <div className="hero-visual">
          <div className="floating-moon">🌙</div>
          <div className="floating-stars">
            <span className="star">⭐</span>
            <span className="star">⭐</span>
            <span className="star">⭐</span>
          </div>
          <div className="sleeping-person">😴</div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="features">
        <div className="container">
          <h2>Designed for better sleep</h2>
          <p className="section-subtitle">Our tools and techniques are backed by science and designed to help you unwind and sleep more soundly.</p>
          
          <div className="features-grid">
            {features.map((feature, index) => (
              <div 
                key={index} 
                className={`feature-card ${index === activeFeature ? 'active' : ''}`}
                onMouseEnter={() => setActiveFeature(index)}
              >
                <div className="feature-icon">{feature.icon}</div>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="how-it-works">
        <div className="container">
          <h2>How it works</h2>
          <div className="steps">
            <div className="step">
              <div className="step-number">1</div>
              <h3>Download the app</h3>
              <p>Get started by downloading our app on iOS or Android.</p>
            </div>
            <div className="step">
              <div className="step-number">2</div>
              <h3>Create your sleep profile</h3>
              <p>Tell us about your sleep habits and preferences.</p>
            </div>
            <div className="step">
              <div className="step-number">3</div>
              <h3>Start your journey</h3>
              <p>Follow your personalized sleep plan and track your progress.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section id="testimonials" className="testimonials">
        <div className="container">
          <h2>What our users say</h2>
          <div className="testimonial-cards">
            <div className="testimonial-card">
              <div className="testimonial-text">
                "I've struggled with insomnia for years. Sleep by Headspace has been a game-changer for me. I'm finally getting restful sleep!"
              </div>
              <div className="testimonial-author">
                <div className="author-avatar">👩</div>
                <div className="author-details">
                  <h4>Sarah M.</h4>
                  <p>User for 1 year</p>
                </div>
              </div>
            </div>
            <div className="testimonial-card">
              <div className="testimonial-text">
                "The sleep stories are my favorite. I rarely make it to the end before I'm fast asleep."
              </div>
              <div className="testimonial-author">
                <div className="author-avatar">👨</div>
                <div className="author-details">
                  <h4>Michael T.</h4>
                  <p>User for 6 months</p>
                </div>
              </div>
            </div>
            <div className="testimonial-card">
              <div className="testimonial-text">
                "The meditation exercises have not only improved my sleep but also reduced my overall anxiety."
              </div>
              <div className="testimonial-author">
                <div className="author-avatar">👩</div>
                <div className="author-details">
                  <h4>Jessica L.</h4>
                  <p>User for 2 years</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta">
        <div className="container">
          <h2>Ready to transform your sleep?</h2>
          <p>Join millions of users who have improved their sleep with our science-backed approach.</p>
          <button className="btn-primary">Get Started Today</button>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <div className="footer-content">
            <div className="footer-section">
              <div className="footer-logo">
                <span className="logo-icon">💤</span>
                <span className="logo-text">Sleep by Headspace</span>
              </div>
              <p>Helping you achieve better sleep and improved well-being through science-backed techniques.</p>
            </div>
            <div className="footer-section">
              <h4>Product</h4>
              <ul>
                <li><a href="#features">Features</a></li>
                <li><a href="#pricing">Pricing</a></li>
                <li><a href="#testimonials">Testimonials</a></li>
              </ul>
            </div>
            <div className="footer-section">
              <h4>Company</h4>
              <ul>
                <li><a href="#about">About</a></li>
                <li><a href="#careers">Careers</a></li>
                <li><a href="#contact">Contact</a></li>
              </ul>
            </div>
            <div className="footer-section">
              <h4>Connect</h4>
              <div className="social-icons">
                <a href="#facebook"><i className="fab fa-facebook"></i></a>
                <a href="#twitter"><i className="fab fa-twitter"></i></a>
                <a href="#instagram"><i className="fab fa-instagram"></i></a>
              </div>
            </div>
          </div>
          <div className="footer-bottom">
            <p>&copy; 2023 Sleep by Headspace. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default HeadspaceHomepage;