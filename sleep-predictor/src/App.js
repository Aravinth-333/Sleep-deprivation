// src/App.js
import React, { useState } from "react";
import SleepDataForm from "./components/InputForm";
import ResultCard from "./components/ResultCard";

function App() {
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handlePredict = async (formData) => {
    try {
      setError("");

      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch prediction");
      }

      const data = await response.json();
      console.log("Prediction result:", data);

      // ✅ Directly set the backend response (like InputForm.js does)
      setResult(data);
    } catch (err) {
      console.error("Error:", err);
      setError("Server not reachable or prediction failed.");
      setResult(null);
    }
  };

  return (
    <div className="app-container">
      {!result ? (
        <SleepDataForm onPredict={handlePredict} />
      ) : (
        <ResultCard result={result} />
      )}

      {error && (
        <div className="error-message">
          <p>⚠️ {error}</p>
        </div>
      )}
    </div>
  );
}

export default App;
