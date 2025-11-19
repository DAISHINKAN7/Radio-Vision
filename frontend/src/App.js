import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:8000';

function App() {
  // State
  const [currentView, setCurrentView] = useState('upload'); // upload, dataset, results
  const [uploadedSignalId, setUploadedSignalId] = useState(null);
  const [classification, setClassification] = useState(null);
  const [radioImage, setRadioImage] = useState(null);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  
  // Dataset browse
  const [datasetSamples, setDatasetSamples] = useState([]);
  const [currentSample, setCurrentSample] = useState(null);
  const [datasetStats, setDatasetStats] = useState(null);
  
  // Drag and drop
  const [dragActive, setDragActive] = useState(false);
  
  // Manual entry
  const [manualSignal, setManualSignal] = useState('');
  
  // Batch processing
  const [batchSignals, setBatchSignals] = useState([]);
  const [batchResults, setBatchResults] = useState([]);

  // Load dataset info on mount
  useEffect(() => {
    loadDatasetInfo();
  }, []);

  const loadDatasetInfo = async () => {
    try {
      const response = await axios.get(`${API_BASE}/dataset/info`);
      setDatasetStats(response.data);
    } catch (error) {
      console.error('Failed to load dataset info:', error);
    }
  };

  // ============================================================================
  // FILE UPLOAD
  // ============================================================================

  const handleFileUpload = async (file) => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE}/api/upload/signal`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      if (response.data.success) {
        setUploadedSignalId(response.data.signal_id);
        setClassification(response.data.classification);
        setRadioImage(response.data.radio_image);
        setCurrentView('results');
      }
    } catch (error) {
      alert(`Upload failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const onFileChange = (e) => {
    const file = e.target.files[0];
    if (file) handleFileUpload(file);
  };

  // Drag and drop handlers
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  // ============================================================================
  // MANUAL SIGNAL ENTRY
  // ============================================================================

  const handleManualSubmit = async () => {
    if (!manualSignal.trim()) {
      alert('Please enter signal data');
      return;
    }

    setLoading(true);

    try {
      // Parse signal (expecting comma or space separated values)
      const values = manualSignal.trim().split(/[\s,]+/).map(parseFloat);
      
      if (values.some(isNaN)) {
        throw new Error('Invalid number format');
      }

      const response = await axios.post(`${API_BASE}/api/upload/manual`, {
        signal: values
      });

      if (response.data.success) {
        setUploadedSignalId(response.data.signal_id);
        setClassification(response.data.classification);
        setRadioImage(response.data.radio_image);
        setCurrentView('results');
      }
    } catch (error) {
      alert(`Failed to process signal: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // ============================================================================
  // DATASET BROWSING
  // ============================================================================

  const loadDatasetSamples = async (objectType = null) => {
    try {
      const response = await axios.post(`${API_BASE}/dataset/query`, {
        object_type: objectType,
        limit: 50,
        offset: 0
      });

      setDatasetSamples(response.data.results);
    } catch (error) {
      console.error('Failed to load samples:', error);
    }
  };

  const selectSample = async (sampleId) => {
    try {
      const opticalResponse = await axios.get(
        `${API_BASE}/dataset/sample/${sampleId}/images/optical`,
        { responseType: 'blob' }
      );
      
      const radioResponse = await axios.get(
        `${API_BASE}/dataset/sample/${sampleId}/images/radio`,
        { responseType: 'blob' }
      );

      const opticalUrl = URL.createObjectURL(opticalResponse.data);
      const radioUrl = URL.createObjectURL(radioResponse.data);

      setCurrentSample({
        sample_id: sampleId,
        optical_url: opticalUrl,
        radio_url: radioUrl,
        ...datasetSamples.find(s => s.sample_id === sampleId)
      });

      setRadioImage(radioUrl);
      setCurrentView('results');
    } catch (error) {
      console.error('Failed to load sample:', error);
    }
  };

  // ============================================================================
  // GENERATION
  // ============================================================================

  const generateImage = async () => {
    if (!uploadedSignalId && !currentSample) {
      alert('Please upload a signal or select a sample first');
      return;
    }

    setGenerating(true);
    setGeneratedImage(null);
    setMetrics(null);

    try {
      const response = await axios.post(`${API_BASE}/api/generate`, {
        signal_id: uploadedSignalId,
        sample_id: currentSample?.sample_id,
        model: 'pix2pix'
      });

      if (response.data.success) {
        setGeneratedImage(response.data.generated_image);
        setMetrics(response.data.metrics);
      }
    } catch (error) {
      alert(`Generation failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setGenerating(false);
    }
  };

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <div className="App">
      {/* Header */}
      <header className="app-header">
        <h1>🌌 Radio Vision - Production System</h1>
        <nav className="nav-tabs">
          <button
            className={currentView === 'upload' ? 'active' : ''}
            onClick={() => setCurrentView('upload')}
          >
            📤 Upload
          </button>
          <button
            className={currentView === 'dataset' ? 'active' : ''}
            onClick={() => {
              setCurrentView('dataset');
              if (datasetSamples.length === 0) loadDatasetSamples();
            }}
          >
            📊 Dataset
          </button>
          <button
            className={currentView === 'results' ? 'active' : ''}
            onClick={() => setCurrentView('results')}
            disabled={!uploadedSignalId && !currentSample}
          >
            🎨 Results
          </button>
        </nav>
      </header>

      <main className="app-main">
        {/* UPLOAD VIEW */}
        {currentView === 'upload' && (
          <div className="upload-section">
            <h2>Upload Signal</h2>
            
            {/* File Upload */}
            <div
              className={`drop-zone ${dragActive ? 'active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              <input
                type="file"
                id="file-upload"
                accept=".csv,.npy,.txt,.png,.jpg,.jpeg"
                onChange={onFileChange}
                style={{ display: 'none' }}
              />
              <label htmlFor="file-upload" className="upload-label">
                <div className="upload-icon">📁</div>
                <p>Drag & drop signal file here</p>
                <p className="upload-hint">or click to browse</p>
                <p className="upload-formats">
                  Supported: CSV, NPY, TXT, PNG, JPG
                </p>
              </label>
            </div>

            {/* Manual Entry */}
            <div className="manual-entry">
              <h3>Or Enter Signal Manually</h3>
              <textarea
                className="signal-input"
                placeholder="Enter signal values (comma or space separated)&#10;Example: 0.1, 0.2, 0.3, 0.4, ..."
                value={manualSignal}
                onChange={(e) => setManualSignal(e.target.value)}
                rows={10}
              />
              <button
                className="btn btn-primary"
                onClick={handleManualSubmit}
                disabled={loading}
              >
                {loading ? 'Processing...' : 'Submit Signal'}
              </button>
            </div>

            {loading && (
              <div className="loading">
                <div className="spinner"></div>
                <p>Processing signal...</p>
              </div>
            )}
          </div>
        )}

        {/* DATASET VIEW */}
        {currentView === 'dataset' && (
          <div className="dataset-section">
            <h2>Browse Dataset</h2>
            
            {datasetStats && (
              <div className="dataset-stats">
                <p>Total Samples: {datasetStats.total_samples}</p>
                <div className="object-types">
                  <button onClick={() => loadDatasetSamples()}>All</button>
                  <button onClick={() => loadDatasetSamples('spiral_galaxy')}>
                    🌀 Spiral Galaxies ({datasetStats.object_types.spiral_galaxy})
                  </button>
                  <button onClick={() => loadDatasetSamples('emission_nebula')}>
                    ☁️ Nebulae ({datasetStats.object_types.emission_nebula})
                  </button>
                  <button onClick={() => loadDatasetSamples('quasar')}>
                    💫 Quasars ({datasetStats.object_types.quasar})
                  </button>
                  <button onClick={() => loadDatasetSamples('pulsar')}>
                    ⚡ Pulsars ({datasetStats.object_types.pulsar})
                  </button>
                </div>
              </div>
            )}

            <div className="samples-grid">
              {datasetSamples.map((sample) => (
                <div
                  key={sample.sample_id}
                  className="sample-card"
                  onClick={() => selectSample(sample.sample_id)}
                >
                  <div className="sample-id">#{sample.sample_id}</div>
                  <div className="sample-type">{sample.object_type.replace('_', ' ')}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* RESULTS VIEW */}
        {currentView === 'results' && (
          <div className="results-section">
            <h2>Results</h2>

            {/* Classification */}
            {classification && (
              <div className="classification-panel">
                <h3>Classification</h3>
                <div className="classification-result">
                  <div className="predicted-class">
                    {classification.predicted_class.replace('_', ' ')}
                  </div>
                  <div className="confidence">
                    Confidence: {(classification.confidence * 100).toFixed(1)}%
                  </div>
                </div>
                
                {classification.probabilities && (
                  <div className="probabilities">
                    {Object.entries(classification.probabilities).map(([cls, prob]) => (
                      <div key={cls} className="prob-bar">
                        <span className="prob-label">{cls.replace('_', ' ')}</span>
                        <div className="prob-track">
                          <div
                            className="prob-fill"
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                        <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Images */}
            <div className="images-panel">
              <div className="image-container">
                <h4>Radio Input</h4>
                {radioImage && (
                  <img
                    src={radioImage}
                    alt="Radio"
                    className="result-image"
                  />
                )}
              </div>

              <div className="generation-controls">
                <button
                  className="btn btn-primary btn-large"
                  onClick={generateImage}
                  disabled={generating}
                >
                  {generating ? 'Generating...' : '▶️ Generate Optical Image'}
                </button>
              </div>

              {generatedImage && (
                <div className="image-container">
                  <h4>Generated Optical</h4>
                  <img
                    src={generatedImage}
                    alt="Generated"
                    className="result-image"
                  />
                </div>
              )}

              {currentSample && (
                <div className="image-container">
                  <h4>Ground Truth</h4>
                  <img
                    src={currentSample.optical_url}
                    alt="Ground Truth"
                    className="result-image"
                  />
                </div>
              )}
            </div>

            {/* Metrics */}
            {metrics && Object.keys(metrics).length > 0 && (
              <div className="metrics-panel">
                <h3>Quality Metrics</h3>
                <div className="metrics-grid">
                  <div className="metric">
                    <div className="metric-label">PSNR</div>
                    <div className="metric-value">{metrics.psnr?.toFixed(2)} dB</div>
                  </div>
                  <div className="metric">
                    <div className="metric-label">SSIM</div>
                    <div className="metric-value">{metrics.ssim?.toFixed(3)}</div>
                  </div>
                  <div className="metric">
                    <div className="metric-label">MSE</div>
                    <div className="metric-value">{metrics.mse?.toFixed(4)}</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;