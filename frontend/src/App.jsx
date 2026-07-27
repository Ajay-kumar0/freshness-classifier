import { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFile = (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Prediction failed");
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Freshness Classifier</h1>

      <div
        className="dropzone"
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onClick={() => document.getElementById("fileInput").click()}
      >
        {preview ? (
          <img src={preview} alt="preview" className="preview" />
        ) : (
          <p>Drag & drop an image, or click to upload</p>
        )}
        <input
          id="fileInput"
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>

      <button onClick={handlePredict} disabled={!file || loading}>
        {loading ? "Predicting..." : "Predict"}
      </button>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result">
          <h2>{result.label}</h2>
          <p>Confidence: {(result.confidence * 100).toFixed(1)}%</p>
          <div className="probs">
            {Object.entries(result.all_probs).map(([label, prob]) => (
              <div key={label} className="prob-row">
                <span>{label}</span>
                <div className="bar-bg">
                  <div className="bar-fill" style={{ width: `${prob * 100}%` }} />
                </div>
                <span>{(prob * 100).toFixed(1)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;