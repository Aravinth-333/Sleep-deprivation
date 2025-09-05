// src/components/InputForm.js
import React, { useState } from 'react';
import './inputgrid.css'

const InputForm = ({ onPredict }) => {
  // Initialize state based on actual model features
  const [formData, setFormData] = useState({
    age: '',
    gender: '',
    sleep_hours: '',
    sleep_quality: '',
    bedtime: '',
    wakeup_time: '',
    sleep_latency: '',
    screen_time_hours: '',
    in_bed_phone_use_percent: '',
    caffeine_intake: '',
    physical_activity_mins: '',
    diet_meal_timing: '',
    water_intake_liters: '',
    stress_level: '',
    day_type: '',
    occupation_type: '',
    study_or_work_hours: '',
    energy_level: '',
    preferred_sleep_time_category: '',
    social_media_hours: '',
    light_exposure_before_bed: '',
    sleep_consistency_score: '',
    daily_commute_time_mins: '',
    afternoon_naps: '',
    time_spent_outdoors_daily: '',
    smoking: '',
    alcohol_habit: '',
    sleep_environment_quality: '',
    medical_conditions: '',
    work_shift_type: ''
  });

  const [isLoading, setIsLoading] = useState(false);

  // Form field definitions with proper labels and types matching the model
  const formFields = [
    // Personal Information
    {
      section: 'Personal Information',
      icon: '👤',
      fields: [
        { name: 'age', label: 'Age (years)', type: 'number', placeholder: '25', min: 18, max: 100 },
        { name: 'gender', label: 'Gender', type: 'select', options: ['Female', 'Male', 'Other'] },
        { name: 'occupation_type', label: 'Occupation', type: 'select', 
          options: ['Healthcare Professional', 'Manufacturing Worker', 'Office Worker', 'Retail Worker', 'Retired', 'Service Worker', 'Student', 'Teacher', 'Technology Professional', 'Unemployed'] },
        { name: 'work_shift_type', label: 'Work Shift', type: 'select', 
          options: ['Day Shift', 'Night Shift', 'Flexible Hours', 'Remote Work', 'Rotating Shift'] }
      ]
    },
    // Sleep Information
    {
      section: 'Sleep Information',
      icon: '😴',
      fields: [
        { name: 'sleep_hours', label: 'Sleep Duration (hours)', type: 'number', placeholder: '7.5', min: 3, max: 12, step: 0.5 },
        { name: 'sleep_quality', label: 'Sleep Quality (1-10)', type: 'number', placeholder: '7', min: 1, max: 10 },
        { name: 'bedtime', label: 'Bedtime', type: 'time', placeholder: '22:30' },
        { name: 'wakeup_time', label: 'Wake Up Time', type: 'time', placeholder: '06:30' },
        { name: 'sleep_latency', label: 'Time to Fall Asleep (minutes)', type: 'number', placeholder: '15', min: 1, max: 120 },
        { name: 'sleep_consistency_score', label: 'Sleep Consistency (1-10)', type: 'number', placeholder: '7', min: 1, max: 10 },
        { name: 'preferred_sleep_time_category', label: 'Sleep Preference', type: 'select',
          options: ['Early Bird', 'Normal Sleeper', 'Night Owl'] },
        { name: 'afternoon_naps', label: 'Afternoon Nap Duration (minutes)', type: 'number', placeholder: '0', min: 0, max: 180 }
      ]
    },
    // Technology & Lifestyle
    {
      section: 'Technology & Digital Habits',
      icon: '📱',
      fields: [
        { name: 'screen_time_hours', label: 'Daily Screen Time (hours)', type: 'number', placeholder: '6', min: 0, max: 18, step: 0.5 },
        { name: 'social_media_hours', label: 'Social Media Time (hours)', type: 'number', placeholder: '2', min: 0, max: 12, step: 0.5 },
        { name: 'in_bed_phone_use_percent', label: 'Phone Use in Bed (%)', type: 'number', placeholder: '30', min: 0, max: 100 },
        { name: 'light_exposure_before_bed', label: 'Light Exposure Before Bed', type: 'select',
          options: ['Low', 'Medium', 'High'] }
      ]
    },
    // Health & Physical Activity
    {
      section: 'Health & Physical Activity',
      icon: '🏃',
      fields: [
        { name: 'physical_activity_mins', label: 'Daily Physical Activity (minutes)', type: 'number', placeholder: '60', min: 0, max: 300 },
        { name: 'time_spent_outdoors_daily', label: 'Time Outdoors (minutes)', type: 'number', placeholder: '90', min: 0, max: 480 },
        { name: 'energy_level', label: 'Energy Level (1-10)', type: 'number', placeholder: '6', min: 1, max: 10 },
        { name: 'stress_level', label: 'Stress Level (1-10)', type: 'number', placeholder: '5', min: 1, max: 10 },
        { name: 'medical_conditions', label: 'Medical Conditions', type: 'select',
          options: ['unknown', 'ADHD', 'Anxiety Disorder', 'Chronic Pain', 'Depression', 'Diabetes', 'Hypertension', 'Insomnia', 'Sleep Apnea'] }
      ]
    },
    // Diet & Substances
    {
      section: 'Diet & Substances',
      icon: '🍽️',
      fields: [
        { name: 'caffeine_intake', label: 'Daily Caffeine Intake (mg)', type: 'number', placeholder: '150', min: 0, max: 1000 },
        { name: 'water_intake_liters', label: 'Daily Water Intake (liters)', type: 'number', placeholder: '2.5', min: 0, max: 6, step: 0.1 },
        { name: 'diet_meal_timing', label: 'Meal Timing Regularity (1-4)', type: 'number', placeholder: '3', min: 1, max: 4 },
        { name: 'smoking', label: 'Smoking Habit', type: 'select',
          options: ['Never', 'Occasionally', 'Regular', 'Heavy'] },
        { name: 'alcohol_habit', label: 'Alcohol Consumption', type: 'select',
          options: ['Never', 'Occasionally', 'Moderate', 'Heavy'] }
      ]
    },
    // Work & Schedule
    {
      section: 'Work & Daily Schedule',
      icon: '💼',
      fields: [
        { name: 'study_or_work_hours', label: 'Work/Study Hours per Day', type: 'number', placeholder: '8', min: 0, max: 16, step: 0.5 },
        { name: 'daily_commute_time_mins', label: 'Daily Commute Time (minutes)', type: 'number', placeholder: '30', min: 0, max: 240 },
        { name: 'day_type', label: 'Day Type', type: 'select', options: ['Weekday', 'Weekend'] }
      ]
    },
    // Environment
    {
      section: 'Sleep Environment',
      icon: '🛏️',
      fields: [
        { name: 'sleep_environment_quality', label: 'Sleep Environment Quality', type: 'select',
          options: ['Poor', 'Fair', 'Good', 'Excellent'] }
      ]
    }
  ];

  const handleChange = (name, value) => {
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    
    try {
      // Convert time fields and prepare data
      const processedData = { ...formData };
      
      // Handle empty fields by setting default values
      Object.keys(processedData).forEach(key => {
        if (processedData[key] === '' || processedData[key] === null) {
          // Set reasonable defaults based on field type
          const field = formFields.flatMap(section => section.fields).find(f => f.name === key);
          if (field) {
            if (field.type === 'number') {
              processedData[key] = parseFloat(field.placeholder) || 0;
            } else if (field.type === 'select' && field.options) {
              processedData[key] = field.options[0];
            } else if (field.type === 'time') {
              processedData[key] = field.placeholder || '22:00';
            }
          }
        }
      });

      // Convert numeric fields to proper types
      ['age', 'sleep_hours', 'sleep_quality', 'sleep_latency', 'screen_time_hours', 
       'in_bed_phone_use_percent', 'caffeine_intake', 'physical_activity_mins', 
       'diet_meal_timing', 'water_intake_liters', 'stress_level', 'study_or_work_hours',
       'energy_level', 'social_media_hours', 'sleep_consistency_score', 
       'daily_commute_time_mins', 'afternoon_naps', 'time_spent_outdoors_daily'].forEach(field => {
        if (processedData[field]) {
          processedData[field] = parseFloat(processedData[field]);
        }
      });

      await onPredict(processedData);
    } catch (error) {
      console.error('Prediction error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const clearForm = () => {
    const clearedData = {};
    Object.keys(formData).forEach(key => {
      clearedData[key] = '';
    });
    setFormData(clearedData);
  };

  return (
    <div className="healthcare-container">
      <div className="header-section">
        <div className="medical-icon">
          <svg viewBox="0 0 24 24" width="48" height="48" fill="currentColor">
            <path d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2M8 12L9.09 18.26L16 19L9.09 19.74L8 26L6.91 19.74L0 19L6.91 18.26L8 12Z"/>
          </svg>
        </div>
        <h1>Sleep Health Assessment</h1>
        <p className="subtitle">Comprehensive Sleep Deprivation Risk Evaluation</p>
      </div>

      <form onSubmit={handleSubmit} className="healthcare-form">
        <div className="form-sections">
          {formFields.map((section, sectionIndex) => (
            <div key={sectionIndex} className="form-section">
              <h3 className="section-title">
                <span className="section-icon">{section.icon}</span>
                {section.section}
              </h3>
              <div className="input-grid">
                {section.fields.map((field, fieldIndex) => (
                  <div key={fieldIndex} className="form-group">
                    <label className="form-label">{field.label}</label>
                    {field.type === 'select' ? (
                      <select
                        value={formData[field.name]}
                        onChange={(e) => handleChange(field.name, e.target.value)}
                        className="form-control"
                        required
                      >
                        <option value="">Select {field.label}</option>
                        {field.options.map((option, optionIndex) => (
                          <option key={optionIndex} value={option}>{option}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type={field.type}
                        step={field.step || (field.type === 'number' ? '1' : undefined)}
                        min={field.min}
                        max={field.max}
                        value={formData[field.name]}
                        onChange={(e) => handleChange(field.name, e.target.value)}
                        className="form-control"
                        placeholder={field.placeholder}
                        required
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="form-actions">
          <button 
            type="button" 
            onClick={clearForm} 
            className="btn btn-secondary"
            disabled={isLoading}
          >
            Clear Form
          </button>
          <button 
            type="submit" 
            className="btn btn-primary"
            disabled={isLoading}
          >
            {isLoading ? (
              <>
                <span className="spinner"></span>
                Analyzing...
              </>
            ) : (
              <>
                <span className="btn-icon">🔍</span>
                Analyze Sleep Risk
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default InputForm;