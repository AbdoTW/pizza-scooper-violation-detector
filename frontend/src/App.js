import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Upload, Download, Wifi, WifiOff, AlertCircle, CheckCircle, Clock, AlertTriangle } from 'lucide-react';
import './App.css';

const App = () => {
  // State management
  const [uploadStatus, setUploadStatus] = useState('idle'); // idle, uploaded, processing, completed, error
  const [isConnected, setIsConnected] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [stats, setStats] = useState({
    processing_status: 'idle',
    current_video: null,
    // Hygiene monitoring stats
    violation_detected: false,
    total_violations: 0,
    roi1_hand_count: 0,
    roi2_hand_count: 0,
    roi1_scooper_count: 0,
    roi2_scooper_count: 0,
    current_state: 'No hands detected',
    violation_reason: 'No violation detected',
    stabilization_remaining: 0
  });
  const [uploadedFile, setUploadedFile] = useState(null);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [violationHistory, setViolationHistory] = useState([]);
  const [selectedRoi, setSelectedRoi] = useState('video1');

  // ROI Configuration Data
  const roiConfigurations = {
    video1: {
      label: "Video Configuration 1",
      roi1: [[480, 273], [384, 688], [482, 708], [548, 284]],
      roi2: [[484, 710], [550, 286], [653, 294], [615, 733]]
    },
    video2: {
      label: "Video Configuration 2", 
      roi1: [[402, 264], [289, 678], [396, 699], [478, 271]],
      roi2: [[398, 701], [480, 273], [605, 277], [561, 718]]
    },
    video3: {
      label: "Video Configuration 3",
      roi1: [[468, 267], [378, 678], [469, 703], [546, 274]],
      roi2: [[470, 704], [546, 274], [639, 282], [610, 723]]
    }
  };

  // Refs
  const fileInputRef = useRef(null);
  const wsRef = useRef(null);
  const statsIntervalRef = useRef(null);

  // Configuration
  const API_BASE = 'http://localhost:8003';
  const WS_URL = 'ws://localhost:8003/ws';
  const SUPPORTED_FORMATS = ['mp4', 'avi', 'mov', 'mkv', 'webm'];

  // Utility functions
  const showError = (message) => {
    setError(message);
    setTimeout(() => setError(''), 5000);
  };

  const showSuccess = (message) => {
    setSuccess(message);
    setTimeout(() => setSuccess(''), 3000);
  };

  const validateFile = (file) => {
    if (!file) return 'No file selected';
    
    const fileExtension = file.name.split('.').pop().toLowerCase();
    if (!SUPPORTED_FORMATS.includes(fileExtension)) {
      return `Unsupported format. Please use: ${SUPPORTED_FORMATS.join(', ').toUpperCase()}`;
    }
    
    // Check file size (limit to 500MB)
    const maxSize = 500 * 1024 * 1024; // 500MB in bytes
    if (file.size > maxSize) {
      return 'File size too large. Maximum size is 500MB';
    }
    
    return null;
  };

  // WebSocket functions
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      wsRef.current = new WebSocket(WS_URL);
      
      wsRef.current.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
      };

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.type === 'frame_update') {
            setCurrentFrame(`data:image/jpeg;base64,${data.frame_data}`);
            
            // Update hygiene monitoring stats from WebSocket data
            if (data.hygiene_stats) {
              setStats(prev => ({
                ...prev,
                violation_detected: data.hygiene_stats.violation_detected || false,
                roi1_hand_count: data.hygiene_stats.roi1_hand_count || 0,
                roi2_hand_count: data.hygiene_stats.roi2_hand_count || 0,
                roi1_scooper_count: data.hygiene_stats.roi1_scooper_count || 0,
                roi2_scooper_count: data.hygiene_stats.roi2_scooper_count || 0,
                current_state: data.hygiene_stats.current_state || 'No hands detected',
                violation_reason: data.hygiene_stats.violation_reason || 'No violation detected',
                stabilization_remaining: data.hygiene_stats.stabilization_remaining || 0,
                // FIX: Update total violations count from WebSocket data
                total_violations: data.hygiene_stats.total_violations || prev.total_violations || 0
              }));

              // Add new violations to history
              if (data.hygiene_stats.new_violation) {
                setViolationHistory(prev => [data.hygiene_stats.new_violation, ...prev].slice(0, 10));
                console.log('New violation detected:', data.hygiene_stats.new_violation);
                console.log('Total violations now:', data.hygiene_stats.total_violations);
              }
            }
          }
          
          // Handle processing completion
          if (data.type === 'processing_complete') {
            console.log('Video processing completed!', data);
            setUploadStatus('completed');
            showSuccess('Video processing completed! You can now download the processed video.');
            stopStatsPolling();
            
            setStats(prev => ({
              ...prev,
              processing_status: 'completed'
            }));
            
            if (wsRef.current) {
              wsRef.current.close();
            }
          }

        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      wsRef.current.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        
        if (uploadStatus === 'processing') {
          setTimeout(connectWebSocket, 2000);
        }
      };

      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };

    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      setIsConnected(false);
    }
  }, [uploadStatus]);

  // API functions
  const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Upload failed' }));
        throw new Error(errorData.detail || 'Upload failed');
      }

      const result = await response.json();
      setUploadedFile({
        filename: result.filename,
        originalName: file.name
      });
      setUploadStatus('uploaded');
      showSuccess('File uploaded successfully!');
      
      await startProcessing(result.filename);
      
    } catch (error) {
      console.error('Upload error:', error);
      showError(error.message || 'Upload failed');
      setUploadStatus('error');
    }
  };

  const startProcessing = async (filename) => {
    try {
      const response = await fetch(`${API_BASE}/api/start-stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          filename: filename,
          roi_config: roiConfigurations[selectedRoi]  // Send selected ROI configuration
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Processing failed to start' }));
        throw new Error(errorData.detail || 'Processing failed to start');
      }

      setUploadStatus('processing');
      connectWebSocket();
      startStatsPolling();
      showSuccess('Processing started!');

    } catch (error) {
      console.error('Start processing error:', error);
      showError(error.message || 'Failed to start processing');
      setUploadStatus('error');
    }
  };

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/stats`);
      if (response.ok) {
        const data = await response.json();
        console.log('Stats update from API:', data);
        
        setStats(prev => {
          // Preserve the higher violation count between WebSocket data and API data
          const updatedStats = {
            ...prev,
            ...data,
            // Keep the higher violation count - WebSocket data is more real-time
            total_violations: Math.max(data.total_violations || 0, prev.total_violations || 0)
          };
          
          console.log('Final stats after merge:', updatedStats);
          return updatedStats;
        });

        if (data.processing_status === 'completed' && uploadStatus === 'processing') {
          setUploadStatus('completed');
          showSuccess('Video processing completed!');
          stopStatsPolling();
        } else if (data.processing_status === 'processing') {
          setUploadStatus('processing');
        } else if (data.processing_status === 'error') {
          setUploadStatus('error');
          stopStatsPolling();
        }
      }
    } catch (error) {
      console.error('Error fetching stats:', error);
    }
  };

  const downloadProcessedVideo = async () => {
    if (!uploadedFile?.filename || uploadStatus !== 'completed') return;

    try {
      const statusResponse = await fetch(`${API_BASE}/api/video-status/${uploadedFile.filename}`);
      const statusData = await statusResponse.json();
      
      if (!statusData.ready) {
        showError('Processed video is not ready yet or file is empty');
        return;
      }
      
      console.log(`Downloading video: ${statusData.file_size_mb}MB`);
      
      const response = await fetch(`${API_BASE}/api/download/${uploadedFile.filename}`);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Download failed' }));
        throw new Error(errorData.detail || 'Download failed');
      }
      
      const blob = await response.blob();
      
      if (blob.size === 0) {
        throw new Error('Downloaded file is empty');
      }
      
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `processed_${uploadedFile.originalName}`;
      
      document.body.appendChild(a);
      a.click();
      
      setTimeout(() => {
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }, 100);
      
      showSuccess(`Download started! (${statusData.file_size_mb}MB)`);

    } catch (error) {
      console.error('Download error:', error);
      showError(error.message || 'Failed to download processed video');
    }
  };

  // Polling functions
  const startStatsPolling = () => {
    stopStatsPolling();
    fetchStats();
    statsIntervalRef.current = setInterval(fetchStats, 2000);
  };

  const stopStatsPolling = () => {
    if (statsIntervalRef.current) {
      clearInterval(statsIntervalRef.current);
      statsIntervalRef.current = null;
    }
  };

  // Event handlers
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const validationError = validateFile(file);
    if (validationError) {
      showError(validationError);
      return;
    }

    uploadFile(file);
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const handleDrop = (event) => {
    event.preventDefault();
    event.stopPropagation();
    
    const files = event.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      const validationError = validateFile(file);
      if (validationError) {
        showError(validationError);
        return;
      }

      uploadFile(file);
    }
  };

  const handleRoiChange = (event) => {
    const newRoi = event.target.value;
    setSelectedRoi(newRoi);
    console.log('ROI Configuration changed to:', roiConfigurations[newRoi]);
  };

  // Status indicator component
  const StatusIndicator = () => (
    <div className="flex items-center gap-2">
      {uploadStatus === 'idle' && <Clock className="w-4 h-4 text-gray-400" />}
      {uploadStatus === 'uploaded' && <CheckCircle className="w-4 h-4 text-green-500" />}
      {uploadStatus === 'processing' && (
        <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
      )}
      {uploadStatus === 'completed' && <CheckCircle className="w-4 h-4 text-green-600" />}
      {uploadStatus === 'error' && <AlertCircle className="w-4 h-4 text-red-500" />}
      <span className="font-medium capitalize text-sm">{uploadStatus}</span>
    </div>
  );

  const ConnectionStatus = () => (
    <div className="flex items-center gap-2">
      {isConnected ? (
        <>
          <Wifi className="w-4 h-4 text-green-500" />
          <span className="text-green-600 text-sm">Connected</span>
        </>
      ) : (
        <>
          <WifiOff className="w-4 h-4 text-red-500" />
          <span className="text-red-600 text-sm">Disconnected</span>
        </>
      )}
    </div>
  );

  // Hygiene Status Component - Restored to larger size
  const HygieneStatus = () => {
    const getStatusConfig = () => {
      if (stats.violation_detected) {
        return {
          icon: <AlertTriangle className="w-6 h-6 text-red-500" />,
          text: "VIOLATION DETECTED",
          bgClass: "bg-red-50 border-red-200",
          textClass: "text-red-700"
        };
      } else if (stats.stabilization_remaining > 0) {
        return {
          icon: <Clock className="w-6 h-6 text-yellow-500" />,
          text: `CHECKING... ${stats.stabilization_remaining.toFixed(1)}s`,
          bgClass: "bg-yellow-50 border-yellow-200", 
          textClass: "text-yellow-700"
        };
      } else {
        return {
          icon: <CheckCircle className="w-6 h-6 text-green-500" />,
          text: "COMPLIANT",
          bgClass: "bg-green-50 border-green-200",
          textClass: "text-green-700"
        };
      }
    };

    const statusConfig = getStatusConfig();

    return (
      <div className={`p-4 rounded-lg border ${statusConfig.bgClass}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            {statusConfig.icon}
            <div>
              <h3 className={`font-bold text-xl ${statusConfig.textClass}`}>
                {statusConfig.text}
              </h3>
              <p className="text-sm text-gray-600">State: {stats.current_state}</p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-red-600">
              {stats.total_violations}
            </div>
            <div className="text-sm text-gray-500">Violations</div>
          </div>
        </div>
      </div>
    );
  };

  // Compact Analysis Component
  const AnalysisPanel = () => (
    <div className="bg-white rounded-lg border p-3">
      <h4 className="font-semibold mb-2 text-sm text-gray-800">Current Analysis</h4>
      <div className="space-y-2">
        {/* Compact metrics in 2 rows */}
        <div className="flex justify-between text-xs">
          <span>ROI-1 H: <span className="font-medium text-blue-600">{stats.roi1_hand_count}</span></span>
          <span>ROI-2 H: <span className="font-medium text-red-600">{stats.roi2_hand_count}</span></span>
          <span>ROI-1 S: <span className="font-medium text-green-600">{stats.roi1_scooper_count}</span></span>
          <span>ROI-2 S: <span className="font-medium text-purple-600">{stats.roi2_scooper_count}</span></span>
        </div>
        <div className="pt-1 border-t text-xs">
          <p className="text-gray-700 truncate" title={stats.violation_reason}>
            <span className="font-medium">Reason:</span> {stats.violation_reason}
          </p>
          {stats.stabilization_remaining > 0 && (
            <p className="text-orange-600">
              Stabilizing: {stats.stabilization_remaining.toFixed(1)}s
            </p>
          )}
        </div>
      </div>
    </div>
  );

  // Cleanup on component unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      stopStatsPolling();
    };
  }, []);

  useEffect(() => {
    connectWebSocket();
    startStatsPolling();
  }, [connectWebSocket]);

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-full mx-auto">
        <header className="mb-4 px-6">
          <h1 className="text-2xl font-bold text-gray-900 mb-1">
            Pizza Store Hygiene Monitoring
          </h1>
          <p className="text-sm text-gray-600">
            Real-time hand hygiene compliance tracking with live video analysis
          </p>
        </header>

        {/* Error/Success Messages */}
        <div className="px-6">
          {error && (
            <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-red-500" />
              <span className="text-red-700">{error}</span>
            </div>
          )}

          {success && (
            <div className="mb-4 p-4 bg-green-50 border border-green-200 rounded-lg flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-green-500" />
              <span className="text-green-700">{success}</span>
            </div>
          )}
        </div>

        <div className="bg-white shadow-sm border-t border-gray-200">
          {/* Control Panel */}
          <div className="bg-white border-b border-gray-200 p-4">
            <div className="flex flex-wrap items-center justify-between gap-4 max-w-7xl mx-auto px-6">
              <div className="flex items-center gap-4">
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileSelect}
                  accept={SUPPORTED_FORMATS.map(format => `.${format}`).join(',')}
                  style={{ display: 'none' }}
                />
                <button 
                  onClick={handleUploadClick}
                  disabled={uploadStatus === 'processing'}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                  <Upload className="w-4 h-4 inline mr-2" />
                  Upload
                </button>

                {/* ROI Configuration Selector */}
                <div className="flex items-center gap-2">
                  <label htmlFor="roi-select" className="text-sm font-medium text-gray-700">
                    ROI Config:
                  </label>
                  <select 
                    id="roi-select"
                    value={selectedRoi}
                    onChange={handleRoiChange}
                    disabled={uploadStatus === 'processing'}
                    className="px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
                  >
                    {Object.entries(roiConfigurations).map(([key, config]) => (
                      <option key={key} value={key}>
                        {config.label}
                      </option>
                    ))}
                  </select>
                </div>

                <StatusIndicator />
                <ConnectionStatus />
              </div>
              
              <button 
                onClick={downloadProcessedVideo}
                disabled={uploadStatus !== 'completed' || !uploadedFile}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  uploadStatus === 'completed' && uploadedFile
                    ? 'bg-green-600 hover:bg-green-700 text-white' 
                    : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                }`}
              >
                <Download className="w-4 h-4 inline mr-2" />
                {uploadStatus === 'completed' && uploadedFile ? 'Download' : 
                 uploadStatus === 'processing' ? 'Processing...' :
                 'Download'}
              </button>
            </div>
          </div>

          {/* Compact Hygiene Status Panel */}
          <div className="p-3 bg-gray-50">
            <div className="max-w-7xl mx-auto">
              {/* Single row layout with compact components */}
              <div className="flex gap-4 items-start">
                {/* Left side - Status (takes up more space) */}
                <div className="flex-1">
                  <HygieneStatus />
                </div>
                {/* Right side - Analysis (compact) */}
                <div className="w-80">
                  <AnalysisPanel />
                </div>
              </div>
            </div>
          </div>

          {/* Video Display - Adjusted height for compact layout */}
          <div 
            className="bg-black flex items-center justify-center relative video-container-compact w-full"
            onDragOver={handleDragOver}
            onDrop={handleDrop}
          >
            {currentFrame ? (
              <img 
                src={currentFrame} 
                alt="Processed frame with hygiene monitoring" 
                className="max-w-full max-h-full object-contain"
              />
            ) : (
              <div className="text-white text-center">
                <Upload className="w-16 h-16 mx-auto mb-4 opacity-50" />
                <p className="text-lg">Drop video file here or click Upload</p>
                <p className="text-sm opacity-75">
                  Supports: {SUPPORTED_FORMATS.join(', ').toUpperCase()}
                </p>
                <p className="text-xs opacity-50 mt-2">
                  Real-time hygiene monitoring with ROI detection will appear here
                </p>
              </div>
            )}
            {uploadStatus === 'processing' && (
              <div className="absolute top-4 left-4 bg-red-600 text-white px-3 py-1 rounded-full text-sm font-medium">
                LIVE
              </div>
            )}
            {stats.current_video && (
              <div className="absolute bottom-4 left-4 bg-black bg-opacity-75 text-white px-3 py-1 rounded text-sm">
                {stats.current_video}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;