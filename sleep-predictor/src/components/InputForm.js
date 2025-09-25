import React, { useState, useCallback } from "react";
import "./inputgrid.css";
import ResultCard from './ResultCard';

/* ============================
   Stable, memoized input components
   (declared outside the main component so their type is stable)
   ============================ */

const FormInput = React.memo(function FormInput({
  type = "text",
  name,
  label,
  value,
  onChange,
  children,
  placeholder,
  required,
  min,
  max,
  step,
  icon,
  ...props
}) {
  return (
    <div className="form-group glow-effect">
      <label htmlFor={name} className="neon-label">
        <span className="icon">{icon}</span>
        {label} {required && <span className="required">*</span>}
      </label>

      {type === "select" ? (
        <select
          id={name}
          name={name}
          value={value ?? ""}
          onChange={onChange}
          className="cyber-input"
          required={required}
          {...props}
        >
          {children}
        </select>
      ) : (
        <input
          type={type}
          id={name}
          name={name}
          min={min}
          max={max}
          step={step}
          value={value ?? ""}
          onChange={onChange}
          placeholder={placeholder}
          className="cyber-input"
          required={required}
          {...props}
        />
      )}
    </div>
  );
});

const SliderInput = React.memo(function SliderInput({
  name,
  label,
  min = 1,
  max = 10,
  value,
  onChange,
  leftLabel,
  rightLabel,
  helpText,
  icon,
}) {
  // For range inputs, value can be a string or number; use fallback to min
  const sliderValue = value ?? min;
  return (
    <div className="form-group glow-effect">
      <label htmlFor={name} className="neon-label">
        <span className="icon">{icon}</span>
        {label}
      </label>
      <div className="slider-container">
        <div className="slider-value neon-text">{sliderValue}</div>
        <input
          type="range"
          id={name}
          name={name}
          min={min}
          max={max}
          value={sliderValue}
          onChange={onChange}
          className="cyber-slider"
        />
        <div className="slider-labels">
          <span>{leftLabel}</span>
          <span>{rightLabel}</span>
        </div>
      </div>
      {helpText && <div className="help-text">{helpText}</div>}
    </div>
  );
});

/* ============================
   Main form
   ============================ */

const SleepDataForm = () => {
  const [formData, setFormData] = useState({
    age: "",
    gender: "",
    bedtime: "",
    wakeup_time: "",
    sleep_latency: "",
    screen_time_hours: "",
    caffeine_intake: "",
    physical_activity_mins: "",
    water_intake_liters: "",
    stress_level: "5", // keep as string
    day_type: "",
    occupation_type: "",
    study_or_work_hours: "",
    social_media_hours: "",
    light_exposure_before_bed: "",
    sleep_consistency_score: "5", // keep as string
    daily_commute_time_mins: "",
    afternoon_naps: "",
    time_spent_outdoors_daily: "",
    smoking: "",
    alcohol_habit: "",
    medical_conditions: "",
    work_shift_type: "",
    preferred_sleep_time_category: "",
    diet_meal_timing: "",           // number of meals per day
    in_bed_phone_use_percent: "", 
  });

  const [showResults, setShowResults] = useState(false);
  const [currentSection, setCurrentSection] = useState(0);
  const [hasUserInteracted, setHasUserInteracted] = useState(false);
  const[prediction,setPrediction]=useState(null);
  // stable callback so its reference doesn't change every render
  const handleInputChange = useCallback((e) => {
    const { name, value } = e.target;

    setFormData((prev) => ({
      ...prev,
      [name]: value // store as string while typing
    }));

    if (!hasUserInteracted) {
      setHasUserInteracted(true);
    }
  }, [hasUserInteracted]);

 const handleSubmit = async () => {
  const requiredFields = ["age", "gender", "bedtime", "wakeup_time", "sleep_latency"];
  const missingFields = requiredFields.filter((field) => !formData[field]);

  if (missingFields.length > 0) {
    alert("Please fill in all required fields: " + missingFields.join(", "));
    return;
  }

  // Prepare processedData (convert to numbers where needed)
  const processedData = {
    ...formData,
    age: formData.age ? parseInt(formData.age, 10) : null,
    sleep_latency: formData.sleep_latency ? parseInt(formData.sleep_latency, 10) : null,
    screen_time_hours: formData.screen_time_hours ? parseFloat(formData.screen_time_hours) : null,
    caffeine_intake: formData.caffeine_intake ? parseFloat(formData.caffeine_intake) : null,
    physical_activity_mins: formData.physical_activity_mins ? parseInt(formData.physical_activity_mins, 10) : null,
    water_intake_liters: formData.water_intake_liters ? parseFloat(formData.water_intake_liters) : null,
    stress_level: formData.stress_level ? parseInt(formData.stress_level, 10) : null,
    sleep_consistency_score: formData.sleep_consistency_score ? parseInt(formData.sleep_consistency_score, 10) : null,
    daily_commute_time_mins: formData.daily_commute_time_mins ? parseInt(formData.daily_commute_time_mins, 10) : null,
    afternoon_naps: formData.afternoon_naps ? parseInt(formData.afternoon_naps, 10) : null,
    time_spent_outdoors_daily: formData.time_spent_outdoors_daily ? parseInt(formData.time_spent_outdoors_daily, 10) : null,
    social_media_hours: formData.social_media_hours ? parseFloat(formData.social_media_hours) : null,
    study_or_work_hours: formData.study_or_work_hours ? parseFloat(formData.study_or_work_hours) : null,
    diet_meal_timing: parseInt(formData.diet_meal_timing) || 0,            // ✅ make sure it’s number
    in_bed_phone_use_percent: parseInt(formData.in_bed_phone_use_percent) || 0, // ✅ make sure it’s number

  };

  try {
    console.log("sending to backend:",processedData)
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(processedData),
    });

    if (!response.ok) {
      throw new Error("Failed to fetch prediction");
    }

    const result = await response.json();
    console.log("Prediction result:", result);

    setPrediction(result);
    setShowResults(true);

  } catch (error) {
    console.error("Error during prediction:", error);
    alert("Something went wrong while analyzing. Please try again.");
  }
};


  const sections = [
    "Personal Info",
    "Sleep Schedule",
    "Technology",
    "Health & Lifestyle",
    "Work & Study",
    "Medical"
  ];

  return (
    <div className="sleep-deprivation-app">
      <div className="stars" />
      <div className="moon" />
      <div className="floating-particles" />

      <div className="main-container">
        <div className="cyber-header">
          <div className="glitch-text">
            <h1 data-text="">🌙 SLEEP DEPRIVATION ANALYZER</h1>
          </div>
          <p className="subtitle">Decode your sleep patterns • Unlock better rest</p>

          <div className="progress-container">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${((currentSection + 1) / sections.length) * 100}%` }}
              />
            </div>
            <div className="section-indicators">
              {sections.map((section, index) => (
                <div
                  key={index}
                  className={`indicator ${index === currentSection ? "active current" : index < currentSection ? "completed" : ""}`}
                  onClick={() => setCurrentSection(index)}
                >
                  <span className="indicator-number">{index + 1}</span>
                  <span className="indicator-text">{section}</span>
                  {index < currentSection && <span className="indicator-check">✓</span>}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="form-container">
          {/* Personal Information */}
          <div className={`section ${currentSection === 0 ? "active" : ""}`} data-section="personal">
            <h2 className="section-title">
              <span className="section-icon">👤</span> Personal Information
              <div className="title-glow" />
            </h2>
            <div className="form-grid">
              <FormInput
                type="number"
                name="age"
                label="Age"
                value={formData.age}
                onChange={handleInputChange}
                placeholder="Enter your age"
                min={13}
                max={100}
                required
                icon="🎂"
              />
              <FormInput
                type="select"
                name="gender"
                label="Gender"
                value={formData.gender}
                onChange={handleInputChange}
                required
                icon="⚧"
              >
                <option value="">Select Gender</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </FormInput>
            </div>
          </div>

          {/* Sleep Schedule */}
          <div className={`section ${currentSection === 1 ? "active" : ""}`} data-section="sleep">
            <h2 className="section-title">
              <span className="section-icon">😴</span> Sleep Schedule
              <div className="title-glow" />
            </h2>
            <div className="form-grid">
              <FormInput
                type="time"
                name="bedtime"
                label="Bedtime"
                value={formData.bedtime}
                onChange={handleInputChange}
                required
                icon="🌜"
              />
              <FormInput
                type="time"
                name="wakeup_time"
                label="Wake Up Time"
                value={formData.wakeup_time}
                onChange={handleInputChange}
                required
                icon="🌅"
              />
              <FormInput
                type="number"
                name="sleep_latency"
                label="Sleep Latency (minutes)"
                value={formData.sleep_latency}
                onChange={handleInputChange}
                placeholder="e.g., 15"
                min={0}
                max={180}
                required
                icon="⏱️"
              />
              <FormInput
                type="number"
                name="afternoon_naps"
                label="Afternoon Naps (minutes)"
                value={formData.afternoon_naps}
                onChange={handleInputChange}
                placeholder="e.g., 30"
                min={0}
                max={240}
                icon="💤"
              />
            </div>

            <div className="slider-grid">
              <SliderInput
                name="sleep_consistency_score"
                label="Sleep Consistency"
                min={1}
                max={10}
                value={formData.sleep_consistency_score}
                onChange={handleInputChange}
                leftLabel="Chaotic"
                rightLabel="Perfect Routine"
                icon="📊"
              />

              <FormInput
                type="select"
                name="preferred_sleep_time_category"
                label="Sleep Chronotype"
                value={formData.preferred_sleep_time_category}
                onChange={handleInputChange}
                icon="🦉"
              >
                <option value="">Select Type</option>
                <option value="Early Bird">Early Bird</option>
                <option value="Normal Sleeper">Normal Sleeper</option>
                <option value="Night Owl">Night Owl</option>
              </FormInput>
            </div>
          </div>

          {/* Technology */}
          <div className={`section ${currentSection === 2 ? "active" : ""}`} data-section="tech">
            <h2 className="section-title">
              <span className="section-icon">📱</span> Digital Exposure
              <div className="title-glow" />
            </h2>
            <div className="form-grid">
              <FormInput
                type="number"
                name="screen_time_hours"
                label="Screen Time (hours/day)"
                value={formData.screen_time_hours}
                onChange={handleInputChange}
                placeholder="e.g., 8"
                min={0}
                max={24}
                step={0.5}
                icon="📺"
              />
              <FormInput
                type="number"
                name="social_media_hours"
                label="Social Media (hours/day)"
                value={formData.social_media_hours}
                onChange={handleInputChange}
                placeholder="e.g., 3"
                min={0}
                max={24}
                step={0.5}
                icon="📲"
              />
              <FormInput
                type="select"
                name="light_exposure_before_bed"
                label="Blue Light Exposure Before Bed"
                value={formData.light_exposure_before_bed}
                onChange={handleInputChange}
                icon="💡"
              >
                <option value="">Select Level</option>
                <option value="Low">Low - Dark room, no screens</option>
                <option value="Medium">Medium - Dim lighting, some screens</option>
                <option value="High">High - Bright lights, heavy screen use</option>
              </FormInput>
              <FormInput
  type="number"
  name="in_bed_phone_use_percent"
  label="Phone Usage in Bed (%)"
  value={formData.in_bed_phone_use_percent}
  onChange={handleInputChange}
  placeholder="0-100"
  min={0}
  max={100}
  icon="📱"
/>
            </div>
          </div>

          {/* Health & Lifestyle */}
          <div className={`section ${currentSection === 3 ? "active" : ""}`} data-section="health">
            <h2 className="section-title">
              <span className="section-icon">💪</span> Health & Lifestyle
              <div className="title-glow" />
            </h2>
            <div className="form-grid">
              <FormInput
                type="number"
                name="caffeine_intake"
                label="Caffeine Intake (mg/day)"
                value={formData.caffeine_intake}
                onChange={handleInputChange}
                placeholder="e.g., 200"
                min={0}
                max={1000}
                icon="☕"
              />
              <FormInput
  type="number"
  name="diet_meal_timing"
  label="Number of Meals per Day"
  value={formData.diet_meal_timing}
  onChange={handleInputChange}
  placeholder="e.g., 3"
  min={0}
  max={10}
  icon="🍽️"
/>
              <FormInput
                type="number"
                name="water_intake_liters"
                label="Water Intake (L/day)"
                value={formData.water_intake_liters}
                onChange={handleInputChange}
                placeholder="e.g., 2.5"
                min={0}
                max={10}
                step={0.1}
                icon="💧"
              />
              <FormInput
                type="number"
                name="physical_activity_mins"
                label="Exercise (minutes/day)"
                value={formData.physical_activity_mins}
                onChange={handleInputChange}
                placeholder="e.g., 60"
                min={0}
                max={480}
                icon="🏃"
              />
              <FormInput
                type="number"
                name="time_spent_outdoors_daily"
                label="Outdoor Time (minutes/day)"
                value={formData.time_spent_outdoors_daily}
                onChange={handleInputChange}
                placeholder="e.g., 120"
                min={0}
                max={720}
                icon="🌳"
              />
              <FormInput
                type="select"
                name="smoking"
                label="Smoking Habit"
                value={formData.smoking}
                onChange={handleInputChange}
                icon="🚬"
              >
                <option value="">Select Option</option>
                <option value="Never">Never</option>
                <option value="Occasionally">Occasionally</option>
                <option value="Regular">Regular</option>
                <option value="Heavy">Heavy</option>
              </FormInput>
              <FormInput
                type="select"
                name="alcohol_habit"
                label="Alcohol Consumption"
                value={formData.alcohol_habit}
                onChange={handleInputChange}
                icon="🍷"
              >
                <option value="">Select Option</option>
                <option value="Never">Never</option>
                <option value="Occasionally">Occasionally</option>
                <option value="Moderate">Moderate</option>
                <option value="Heavy">Heavy</option>
              </FormInput>
            </div>

            <div className="slider-grid">
              <SliderInput
                name="stress_level"
                label="Stress Level"
                min={1}
                max={10}
                value={formData.stress_level}
                onChange={handleInputChange}
                leftLabel="Zen Mode"
                rightLabel="Burnout"
                icon="😰"
              />
            </div>
          </div>

          {/* Work & Study */}
          <div className={`section ${currentSection === 4 ? "active" : ""}`} data-section="work">
            <h2 className="section-title">
              <span className="section-icon">💼</span> Work & Study
              <div className="title-glow" />
            </h2>
            <div className="form-grid">
              <FormInput
                type="select"
                name="occupation_type"
                label="Occupation"
                value={formData.occupation_type}
                onChange={handleInputChange}
                icon="👔"
              >
                <option value="">Select Occupation</option>
                <option value="Teacher">Teacher</option>
                <option value="Retail Worker">Retail Worker</option>
                <option value="Service Worker">Service Worker</option>
                <option value="Office Worker">Office Worker</option>
                <option value="Technology Professional">Technology Professional</option>
                <option value="Healthcare Professional">Healthcare Professional</option>
                <option value="Student">Student</option>
                <option value="Unemployed">Unemployed</option>
                <option value="Manufacturing Worker">Manufacturing Worker</option>
                <option value="Retired">Retired</option>
              </FormInput>

              <FormInput
                type="select"
                name="work_shift_type"
                label="Work Schedule"
                value={formData.work_shift_type}
                onChange={handleInputChange}
                icon="🕘"
              >
                <option value="">Select Schedule</option>
                <option value="Day Shift">Day Shift</option>
                <option value="Night Shift">Night Shift</option>
                <option value="Rotating Shift">Rotating Shift</option>
                <option value="Remote Work">Remote Work</option>
                <option value="Flexible Hours">Flexible Hours</option>
              </FormInput>

              <FormInput
                type="number"
                name="study_or_work_hours"
                label="Work/Study Hours (per day)"
                value={formData.study_or_work_hours}
                onChange={handleInputChange}
                placeholder="e.g., 8"
                min={0}
                max={24}
                icon="📚"
              />

              <FormInput
                type="number"
                name="daily_commute_time_mins"
                label="Commute Time (minutes)"
                value={formData.daily_commute_time_mins}
                onChange={handleInputChange}
                placeholder="e.g., 45"
                min={0}
                max={480}
                icon="🚗"
              />

              <FormInput
                type="select"
                name="day_type"
                label="Day Type"
                value={formData.day_type}
                onChange={handleInputChange}
                icon="📅"
              >
                <option value="">Select Day Type</option>
                <option value="Weekday">Weekday</option>
                <option value="Weekend">Weekend</option>
              </FormInput>
            </div>
          </div>

          {/* Medical */}
          <div className={`section ${currentSection === 5 ? "active" : ""}`} data-section="medical">
            <h2 className="section-title">
              <span className="section-icon">🏥</span> Medical Assessment
              <div className="title-glow" />
            </h2>
            <div className="form-grid">
              <FormInput
                type="select"
                name="medical_conditions"
                label="Medical Conditions"
                value={formData.medical_conditions}
                onChange={handleInputChange}
                icon="🩺"
              >
                <option value="">Select Condition</option>
                <option value="None">None</option>
                <option value="Hypertension">Hypertension</option>
                <option value="Depression">Depression</option>
                <option value="Anxiety Disorder">Anxiety Disorder</option>
                <option value="ADHD">ADHD</option>
                <option value="Insomnia">Insomnia</option>
                <option value="Diabetes">Diabetes</option>
                <option value="Chronic Pain">Chronic Pain</option>
                <option value="Sleep Apnea">Sleep Apnea</option>
              </FormInput>
            </div>
          </div>

          {/* Navigation */}
          <div className="navigation">
            <button
              className="nav-btn prev"
              onClick={() => setCurrentSection(Math.max(0, currentSection - 1))}
              disabled={currentSection === 0}
            >
              ← Previous
            </button>

            {currentSection < sections.length - 1 ? (
              <button className="nav-btn next" onClick={() => setCurrentSection(currentSection + 1)}>
                Next →
              </button>
            ) : (
              <button onClick={handleSubmit} className="cyber-submit">
                <span className="btn-text">Analyze Sleep Data</span>
                <div className="btn-glow" />
              </button>
            )}
          </div>

          {/* Results */}
      {showResults && prediction && (
  <ResultCard result={prediction} />
)}

        </div>
      </div>
    </div>
  );
};

export default SleepDataForm;
