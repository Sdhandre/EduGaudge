// EduGauge - Classroom Attention Monitoring System
// Complete Application with Session History, Dark Mode, and Enhanced Features

(() => {
  'use strict';

  // Configuration
  const CONFIG = {
    WS_URL: "ws://localhost:8000/ws",
    API_BASE: "http://localhost:8000/api",
    UPDATE_INTERVAL: 500,
    CHART_UPDATE_FREQUENCY: 10,
    ACTIVITY_FEED_MAX: 50,
    STORAGE_KEY: 'eduGauge_sessions',
    SETTINGS_KEY: 'eduGauge_settings',
    THEME_KEY: 'eduGauge_theme'
  };

  // Local Storage Manager
  // MongoDB API Storage Manager
  // MongoDB API Storage Manager - REPLACE YOUR EXISTING Storage OBJECT
  const Storage = {
    save: async (key, data) => {
      try {
        if (key === CONFIG.STORAGE_KEY) {
          // For sessions, save the most recent one
          if (Array.isArray(data) && data.length > 0) {
            const session = data[0]; // Most recent session
            const response = await fetch('http://localhost:8000/api/sessions', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(session)
            });
            const result = await response.json();
            console.log('Session saved to MongoDB:', result);
          }
        } else if (key === CONFIG.SETTINGS_KEY) {
          // Save settings to API (if you have settings endpoint)
          localStorage.setItem(key, JSON.stringify(data));
        } else {
          // Other keys use localStorage
          localStorage.setItem(key, JSON.stringify(data));
        }
      } catch (error) {
        console.error('Failed to save to API:', error);
        // Fallback to localStorage
        localStorage.setItem(key, JSON.stringify(data));
      }
    },

    load: async (key, defaultValue = null) => {
      try {
        if (key === CONFIG.STORAGE_KEY) {
          const response = await fetch('http://localhost:8000/api/sessions');
          const result = await response.json();
          if (result.success) {
            console.log('Loaded sessions from MongoDB:', result.sessions.length);
            return result.sessions;
          } else {
            console.error('Failed to load sessions:', result.message);
            return defaultValue;
          }
        } else {
          // Other keys use localStorage
          const data = localStorage.getItem(key);
          return data ? JSON.parse(data) : defaultValue;
        }
      } catch (error) {
        console.error('Failed to load from API:', error);
        // Fallback to localStorage
        const data = localStorage.getItem(key);
        return data ? JSON.parse(data) : defaultValue;
      }
    },

    remove: async (key) => {
      try {
        if (key === CONFIG.STORAGE_KEY) {
          await fetch(`${CONFIG.API_BASE}/sessions`, { method: 'DELETE' });
        }
        localStorage.removeItem(key);
      } catch (error) {
        console.error('Failed to remove from MongoDB:', error);
        localStorage.removeItem(key);
      }
    },

    clear: async () => {
      try {
        await fetch(`${CONFIG.API_BASE}/sessions`, { method: 'DELETE' });
        localStorage.clear();
      } catch (error) {
        console.error('Failed to clear MongoDB:', error);
        localStorage.clear();
      }
    }
  };


  // DOM Elements
  const elements = {
    // Navigation
    navButtons: document.querySelectorAll('.nav-btn'),
    sections: document.querySelectorAll('.content-section'),

    // Theme
    darkModeToggle: document.getElementById('darkModeToggle'),

    // Session Controls
    startBtn: document.getElementById('startBtn'),
    endSessionBtn: document.getElementById('endSessionBtn'),
    pauseBtn: document.getElementById('pauseBtn'),
    screenshotBtn: document.getElementById('screenshotBtn'),
    recordBtn: document.getElementById('recordBtn'),
    className: document.getElementById('className'),

    // Video Elements
    video: document.getElementById('previewVideo'),
    overlayCanvas: document.getElementById('overlayCanvas'),
    captureCanvas: document.getElementById('captureCanvas'),

    // Live Session Elements
    sessionTimer: document.getElementById('sessionTimer'),
    overallScore: document.getElementById('overallScore'),
    attentionBar: document.getElementById('attentionBar'),
    currentClassInfo: document.getElementById('currentClassInfo'),
    liveStudentCount: document.getElementById('liveStudentCount'),
    liveAttentive: document.getElementById('liveAttentive'),
    liveDistracted: document.getElementById('liveDistracted'),
    livePhoneUsers: document.getElementById('livePhoneUsers'),
    activityFeed: document.getElementById('activityFeed'),
    clearActivityBtn: document.getElementById('clearActivityBtn'),

    // Home Elements
    recentSessions: document.getElementById('recentSessions'),
    viewAllSessionsBtn: document.getElementById('viewAllSessionsBtn'),
    totalSessionTime: document.getElementById('totalSessionTime'),
    avgEngagement: document.getElementById('avgEngagement'),
    totalClasses: document.getElementById('totalClasses'),
    totalInsights: document.getElementById('totalInsights'),
    refreshStatsBtn: document.getElementById('refreshStatsBtn'),
    notificationBtn: document.getElementById('notificationBtn'),
    notificationBadge: document.getElementById('notificationBadge'),

    // Analytics Elements
    analyticsClassInfo: document.getElementById('analyticsClassInfo'),
    analyticsDuration: document.getElementById('analyticsDuration'),
    analyticsAvgScore: document.getElementById('analyticsAvgScore'),
    analyticsMaxStudents: document.getElementById('analyticsMaxStudents'),
    analyticsEngagement: document.getElementById('analyticsEngagement'),
    attentionChart: document.getElementById('attentionChart'),
    keyInsights: document.getElementById('keyInsights'),
    attentionPatterns: document.getElementById('attentionPatterns'),
    recommendations: document.getElementById('recommendations'),

    // ADD THESE LINES FOR AI ANALYSIS:
    generateAiAnalysis: document.getElementById('generateAiAnalysis'),
    aiAnalysisContainer: document.getElementById('aiAnalysisContainer'),
    aiAnalysisLoading: document.getElementById('aiAnalysisLoading'),
    aiAnalysisContent: document.getElementById('aiAnalysisContent'),

    // History Elements
    sessionHistoryContainer: document.getElementById('sessionHistoryContainer'),
    filterSessionsBtn: document.getElementById('filterSessionsBtn'),
    sessionFilter: document.getElementById('sessionFilter'),
    filterPeriod: document.getElementById('filterPeriod'),
    applyFilterBtn: document.getElementById('applyFilterBtn'),
    exportAllBtn: document.getElementById('exportAllBtn'),

    // Settings Elements
    autoSaveSetting: document.getElementById('autoSaveSetting'),
    lowAttentionAlerts: document.getElementById('lowAttentionAlerts'),
    sessionReports: document.getElementById('sessionReports'),
    dataRetention: document.getElementById('dataRetention'),
    sensitivityRange: document.getElementById('sensitivityRange'),
    updateFrequency: document.getElementById('updateFrequency'),
    privacyMode: document.getElementById('privacyMode'),
    cameraQuality: document.getElementById('cameraQuality'),
    resetSettingsBtn: document.getElementById('resetSettingsBtn'),
    saveSettingsBtn: document.getElementById('saveSettingsBtn'),

    // Footer Elements
    clearDataBtn: document.getElementById('clearDataBtn'),
    helpBtn: document.getElementById('helpBtn'),

    // Export Elements
    shareReportBtn: document.getElementById('shareReportBtn'),
    exportReportBtn: document.getElementById('exportReportBtn')
  };

  // Application State
  let state = {
    ws: null,
    currentStream: null,
    overlayCtx: null,
    frameInterval: null,
    timerInterval: null,
    attentionChart: null,
    sessionStartTime: null,
    currentSession: null,
    isMonitoring: false,
    isPaused: false,
    isDarkMode: false,
    notifications: [],
    isRecording: false
  };

  // Session Data (Generalized)
  let sessionData = {
    id: null,
    className: '',
    startTime: null,
    endTime: null,
    duration: 0,
    overallScores: [],
    classMetrics: [],
    timelineData: [],
    totalFrames: 0,
    maxStudentsDetected: 0,
    distractionEvents: [],
    activityLog: [],
    screenshots: []
  };

  // Initialize Application
  function init() {
    console.log('ðŸŽ“ EduGauge - Initializing...');

    initTheme();
    initNavigation();
    initEventListeners();
    loadSettings();
    updateHomeStats();
    loadRecentSessions();
    loadSessionHistory();

    // Load saved sessions
    const savedSessions = Storage.load(CONFIG.STORAGE_KEY, []);
    console.log(` Loaded ${savedSessions.length} saved sessions`);

    console.log(' EduGauge initialized successfully');
  }

  // Theme Management
  function initTheme() {
    const savedTheme = Storage.load(CONFIG.THEME_KEY, 'light');
    state.isDarkMode = savedTheme === 'dark';

    if (state.isDarkMode) {
      document.documentElement.classList.add('dark');
      elements.darkModeToggle.querySelector('.material-icons').textContent = 'light_mode';
    } else {
      document.documentElement.classList.remove('dark');
      elements.darkModeToggle.querySelector('.material-icons').textContent = 'dark_mode';
    }
  }

  function toggleDarkMode() {
    state.isDarkMode = !state.isDarkMode;

    if (state.isDarkMode) {
      document.documentElement.classList.add('dark');
      elements.darkModeToggle.querySelector('.material-icons').textContent = 'light_mode';
    } else {
      document.documentElement.classList.remove('dark');
      elements.darkModeToggle.querySelector('.material-icons').textContent = 'dark_mode';
    }

    Storage.save(CONFIG.THEME_KEY, state.isDarkMode ? 'dark' : 'light');
    showNotification('Theme changed successfully', 'success');
  }

  // Navigation System
  function initNavigation() {
    elements.navButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const targetSection = btn.dataset.section;
        switchSection(targetSection);
        updateNavState(btn);
      });
    });
  }

  function switchSection(sectionId) {
    elements.sections.forEach(section => {
      section.classList.add('section-hidden');
    });

    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
      targetSection.classList.remove('section-hidden');
    }

    // Special handling for analytics section
    if (sectionId === 'analytics' && state.attentionChart) {
      setTimeout(() => {
        state.attentionChart.resize();
      }, 100);
    }
  }

  function updateNavState(activeBtn) {
    elements.navButtons.forEach(btn => {
      btn.classList.remove('text-indigo-600', 'bg-indigo-50', 'dark:bg-indigo-900/50');
      btn.classList.add('text-gray-600', 'dark:text-gray-300');
    });

    activeBtn.classList.remove('text-gray-600', 'dark:text-gray-300');
    activeBtn.classList.add('text-indigo-600', 'bg-indigo-50', 'dark:bg-indigo-900/50');
  }

  // Session Management
  async function startSession() {
    try {
      const className = elements.className.value.trim();

      if (!className) {
        showNotification('Please enter a class name', 'warning');
        return;
      }

      // Generate session ID
      const sessionId = 'session_' + Date.now();

      // Initialize session data
      sessionData = {
        id: sessionId,
        className: className,
        startTime: Date.now(),
        endTime: null,
        duration: 0,
        overallScores: [],
        classMetrics: [],
        timelineData: [],
        totalFrames: 0,
        maxStudentsDetected: 0,
        distractionEvents: [],
        activityLog: [],
        screenshots: []
      };

      // Start camera
      await startCamera();

      // Initialize overlay
      initOverlay();

      // Connect WebSocket
      connectWebSocket();

      // Start session timer
      startSessionTimer();

      // Update UI
      elements.currentClassInfo.textContent = className;
      elements.startBtn.disabled = true;
      elements.endSessionBtn.disabled = false;
      elements.pauseBtn.disabled = false;
      elements.screenshotBtn.disabled = false;
      elements.recordBtn.disabled = false;

      state.isMonitoring = true;
      state.currentSession = sessionData;

      // Switch to live session view
      switchSection('live-session');
      updateNavState(document.querySelector('[data-section="live-session"]'));

      addActivityLog(`Session started: ${className}`, 'success');
      showNotification(`Session "${className}" started successfully`, 'success');

    } catch (error) {
      console.error('Failed to start session:', error);
      showNotification('Failed to start session. Please check camera permissions.', 'error');
    }
  }

  async function endSession() {
    try {
      if (!state.isMonitoring) return;

      // Stop monitoring
      state.isMonitoring = false;
      state.isPaused = false;

      // Stop intervals
      if (state.frameInterval) {
        clearInterval(state.frameInterval);
        state.frameInterval = null;
      }

      stopSessionTimer();

      // Close WebSocket
      if (state.ws) {
        state.ws.close();
        state.ws = null;
      }

      // Stop camera
      stopCamera();

      // Clear overlays
      if (state.overlayCtx) {
        state.overlayCtx.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);
      }

      // Update session end time
      sessionData.endTime = Date.now();
      sessionData.duration = Math.floor((sessionData.endTime - sessionData.startTime) / 1000);

      // Save session to local storage
      saveSession(sessionData);

      // Update UI
      elements.startBtn.disabled = false;
      elements.endSessionBtn.disabled = true;
      elements.pauseBtn.disabled = true;
      elements.screenshotBtn.disabled = true;
      elements.recordBtn.disabled = true;
      elements.overallScore.textContent = 'â€”';
      elements.attentionBar.style.width = '0%';

      // Reset live metrics
      elements.liveStudentCount.textContent = '0';
      elements.liveAttentive.textContent = '0';
      elements.liveDistracted.textContent = '0';
      elements.livePhoneUsers.textContent = '0';

      addActivityLog('Session completed successfully', 'success');
      showNotification(`Session "${sessionData.className}" completed`, 'success');

      // Update home stats
      updateHomeStats();
      loadRecentSessions();

      // Generate analytics
      generateAnalytics();

      // Switch to analytics view
      switchSection('analytics');
      updateNavState(document.querySelector('[data-section="analytics"]'));

    } catch (error) {
      console.error('Error ending session:', error);
      showNotification('Error ending session', 'error');
    }
  }

  function pauseSession() {
    state.isPaused = !state.isPaused;

    if (state.isPaused) {
      elements.pauseBtn.innerHTML = '<span class="material-icons">play_arrow</span> Resume';
      addActivityLog('Session paused', 'warning');
    } else {
      elements.pauseBtn.innerHTML = '<span class="material-icons">pause</span> Pause';
      addActivityLog('Session resumed', 'success');
    }
  }

  function takeScreenshot() {
    if (!elements.video || !elements.captureCanvas) return;

    try {
      const ctx = elements.captureCanvas.getContext('2d');
      ctx.drawImage(elements.video, 0, 0, elements.captureCanvas.width, elements.captureCanvas.height);

      elements.captureCanvas.toBlob(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `EduGauge_Screenshot_${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
        a.click();
        URL.revokeObjectURL(url);

        // Store screenshot reference
        sessionData.screenshots.push({
          timestamp: Date.now(),
          filename: a.download
        });

        addActivityLog('ðŸ“¸ Screenshot captured', 'info');
        showNotification('Screenshot saved successfully', 'success');
      }, 'image/png');

    } catch (error) {
      console.error('Screenshot error:', error);
      showNotification('Failed to capture screenshot', 'error');
    }
  }

  function toggleRecording() {
    state.isRecording = !state.isRecording;

    if (state.isRecording) {
      elements.recordBtn.innerHTML = '<span class="material-icons animate-pulse text-red-500">stop</span> Stop Recording';
      elements.recordBtn.classList.add('bg-red-100', 'dark:bg-red-900/50');
      addActivityLog('ðŸŽ¥ Recording started', 'info');
      showNotification('Recording started', 'success');
    } else {
      elements.recordBtn.innerHTML = '<span class="material-icons">fiber_manual_record</span> Record';
      elements.recordBtn.classList.remove('bg-red-100', 'dark:bg-red-900/50');
      addActivityLog('Recording stopped', 'info');
      showNotification('Recording stopped', 'success');
    }
  }

  // Camera Management
  async function startCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false
    });

    elements.video.srcObject = stream;
    state.currentStream = stream;

    return new Promise(resolve => {
      elements.video.onloadedmetadata = resolve;
    });
  }

  function stopCamera() {
    if (state.currentStream) {
      state.currentStream.getTracks().forEach(track => track.stop());
      state.currentStream = null;
    }
    elements.video.srcObject = null;
  }

  function initOverlay() {
    state.overlayCtx = elements.overlayCanvas.getContext('2d');

    const resizeOverlay = () => {
      if (!elements.video.videoWidth || !elements.video.videoHeight) return;

      const rect = elements.video.getBoundingClientRect();
      elements.overlayCanvas.width = rect.width;
      elements.overlayCanvas.height = rect.height;
      elements.overlayCanvas.style.width = rect.width + 'px';
      elements.overlayCanvas.style.height = rect.height + 'px';
    };

    elements.video.addEventListener('loadedmetadata', resizeOverlay);
    elements.video.addEventListener('resize', resizeOverlay);
    window.addEventListener('resize', resizeOverlay);

    setTimeout(resizeOverlay, 500);
  }

  // WebSocket Communication
  function connectWebSocket() {
    state.ws = new WebSocket(CONFIG.WS_URL);
    state.ws.binaryType = "arraybuffer";

    state.ws.onopen = () => {
      console.log("ðŸ”Œ WebSocket connected");
      startFrameCapture();
      addActivityLog('Connected to monitoring server', 'success');
    };

    state.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        processWebSocketData(data);
      } catch (e) {
        console.error("WebSocket message error:", e);
      }
    };

    state.ws.onclose = () => {
      console.log("ðŸ”Œ WebSocket closed");
      addActivityLog('Disconnected from monitoring server', 'warning');
    };

    state.ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      addActivityLog('Connection error occurred', 'error');
    };
  }

  function startFrameCapture() {
    const ctx = elements.captureCanvas.getContext('2d');

    state.frameInterval = setInterval(() => {
      if (state.ws && state.ws.readyState === WebSocket.OPEN &&
        elements.video.readyState === elements.video.HAVE_ENOUGH_DATA &&
        !state.isPaused) {

        ctx.drawImage(elements.video, 0, 0, elements.captureCanvas.width, elements.captureCanvas.height);
        elements.captureCanvas.toBlob(blob => {
          if (blob) {
            blob.arrayBuffer().then(buf => state.ws.send(buf));
          }
        }, "image/jpeg", 0.7);
      }
    }, CONFIG.UPDATE_INTERVAL);
  }

  // Data Processing
  function processWebSocketData(data) {
    if (!data || !state.isMonitoring) return;

    const timestamp = Date.now();
    const relativeTime = Math.floor((timestamp - sessionData.startTime) / 1000);

    // Process overall attention score
    if (typeof data.overall_score === 'number') {
      sessionData.overallScores.push({
        time: relativeTime,
        score: data.overall_score,
        timestamp: timestamp
      });

      updateLiveScore(data.overall_score);
    }

    // Process class metrics
    if (Array.isArray(data.persons)) {
      const classMetrics = analyzeClassData(data.persons);
      sessionData.classMetrics.push({
        time: relativeTime,
        ...classMetrics
      });

      updateLiveMetrics(classMetrics);
      drawGeneralizedOverlay(data.persons);

      // Check for significant events
      checkForClassEvents(classMetrics);
    }

    sessionData.totalFrames++;

    // Update timeline data for chart
    if (sessionData.totalFrames % CONFIG.CHART_UPDATE_FREQUENCY === 0) {
      updateTimelineChart();
    }
  }

  function analyzeClassData(persons) {
    const total = persons.length;
    let attentive = 0;
    let distracted = 0;
    let phoneUsers = 0;

    persons.forEach(person => {
      switch (person.status) {
        case 'ATTENTIVE':
          attentive++;
          break;
        case 'DROWSY':
          distracted++;
          break;
        case 'USING PHONE':
          phoneUsers++;
          break;
      }
    });

    sessionData.maxStudentsDetected = Math.max(sessionData.maxStudentsDetected, total);

    return {
      total,
      attentive,
      distracted,
      phoneUsers,
      attentivePercent: total > 0 ? Math.round((attentive / total) * 100) : 0,
      distractedPercent: total > 0 ? Math.round((distracted / total) * 100) : 0,
      phonePercent: total > 0 ? Math.round((phoneUsers / total) * 100) : 0
    };
  }

  function checkForClassEvents(metrics) {
    const now = Date.now();

    // High distraction alert
    if (metrics.distractedPercent > 50 && metrics.total >= 5) {
      const lastEvent = sessionData.distractionEvents[sessionData.distractionEvents.length - 1];
      if (!lastEvent || (now - lastEvent.timestamp) > 30000) {
        sessionData.distractionEvents.push({
          timestamp: now,
          type: 'high_distraction',
          message: `High distraction detected: ${metrics.distractedPercent}% of class`
        });
        addActivityLog(`âš ï¸ High distraction level (${metrics.distractedPercent}% of class)`, 'warning');

        // Show notification for low attention alerts setting
        if (Storage.load(CONFIG.SETTINGS_KEY, {}).lowAttentionAlerts !== false) {
          showNotification(`High distraction detected: ${metrics.distractedPercent}% of class`, 'warning');
        }
      }
    }

    // Phone usage spike
    if (metrics.phonePercent > 30 && metrics.total >= 3) {
      addActivityLog(`ðŸ“± Phone usage spike (${metrics.phoneUsers} students)`, 'error');
    }

    // Excellent engagement
    if (metrics.attentivePercent >= 90 && metrics.total >= 5) {
      addActivityLog(`âœ… Excellent engagement (${metrics.attentivePercent}% attentive)`, 'success');
    }
  }

  // UI Updates
  function updateLiveScore(score) {
    const roundedScore = Math.round(score);
    elements.overallScore.textContent = roundedScore + '%';
    elements.attentionBar.style.width = Math.max(5, Math.min(100, roundedScore)) + '%';

    // Update bar color
    elements.attentionBar.classList.remove('bg-red-500', 'bg-yellow-500', 'bg-green-500');
    if (roundedScore < 60) {
      elements.attentionBar.classList.add('bg-red-500');
    } else if (roundedScore < 80) {
      elements.attentionBar.classList.add('bg-yellow-500');
    } else {
      elements.attentionBar.classList.add('bg-green-500');
    }
  }

  function updateLiveMetrics(metrics) {
    elements.liveStudentCount.textContent = metrics.total;
    elements.liveAttentive.textContent = metrics.attentive;
    elements.liveDistracted.textContent = metrics.distracted;
    elements.livePhoneUsers.textContent = metrics.phoneUsers;
  }

  function drawGeneralizedOverlay(persons) {
    if (!state.overlayCtx || !persons.length) return;

    state.overlayCtx.clearRect(0, 0, elements.overlayCanvas.width, elements.overlayCanvas.height);

    const scaleX = elements.overlayCanvas.width / 640;
    const scaleY = elements.overlayCanvas.height / 480;

    // Draw generalized attention heatmap
    const attentiveZones = [];
    const drowsyZones = [];
    const phoneZones = [];

    persons.forEach(person => {
      if (!person.bbox) return;

      const [x1, y1, x2, y2] = person.bbox;
      const zone = {
        x: x1 * scaleX,
        y: y1 * scaleY,
        width: (x2 - x1) * scaleX,
        height: (y2 - y1) * scaleY
      };

      if (person.status === 'ATTENTIVE') {
        attentiveZones.push(zone);
      } else if (person.status === 'DROWSY') {
        drowsyZones.push(zone);           // â† DROWSY gets red
      } else if (person.status === 'USING PHONE') {
        phoneZones.push(zone);            // â† PHONE gets different color
      }
    });

    // Draw attention zones
    state.overlayCtx.fillStyle = 'rgba(16, 185, 129, 0.2)';
    attentiveZones.forEach(zone => {
      state.overlayCtx.fillRect(zone.x, zone.y, zone.width, zone.height);
    });

    state.overlayCtx.fillStyle = 'rgba(239, 68, 68, 0.3)';
    drowsyZones.forEach(zone => {
      state.overlayCtx.fillRect(zone.x, zone.y, zone.width, zone.height);
    });

    state.overlayCtx.fillStyle = 'rgba(245, 152, 21, 0.33)';  // â† Orange for phone
    phoneZones.forEach(zone => {
      state.overlayCtx.fillRect(zone.x, zone.y, zone.width, zone.height);
    });

    // Draw class status overlay
    const totalZones = attentiveZones.length + distractedZones.length;
    if (totalZones > 0) {
      const attentivePercent = Math.round((attentiveZones.length / totalZones) * 100);
      state.overlayCtx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      state.overlayCtx.fillRect(10, 10, 200, 30);
      state.overlayCtx.fillStyle = 'white';
      state.overlayCtx.font = '14px Inter';
      state.overlayCtx.fillText(`Class Attention: ${attentivePercent}%`, 15, 28);
    }
  }

  // Timer Management
  function startSessionTimer() {
    state.sessionStartTime = Date.now();

    state.timerInterval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - state.sessionStartTime) / 1000);
      const minutes = Math.floor(elapsed / 60);
      const seconds = elapsed % 60;
      elements.sessionTimer.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
      sessionData.duration = elapsed;
    }, 1000);
  }

  function stopSessionTimer() {
    if (state.timerInterval) {
      clearInterval(state.timerInterval);
      state.timerInterval = null;
    }
  }

  // Activity Feed Management
  function addActivityLog(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const activity = { timestamp, message, type };
    sessionData.activityLog.push(activity);

    // Update UI
    const feedElement = elements.activityFeed;
    const activityDiv = document.createElement('div');
    activityDiv.className = `activity-item activity-${type}`;
    activityDiv.innerHTML = `<span class="text-gray-500 dark:text-gray-400">${timestamp}</span> ${message}`;

    feedElement.insertBefore(activityDiv, feedElement.firstChild);

    // Keep only recent activities
    while (feedElement.children.length > CONFIG.ACTIVITY_FEED_MAX) {
      feedElement.removeChild(feedElement.lastChild);
    }
  }

  function clearActivityFeed() {
    elements.activityFeed.innerHTML = '<p class="text-sm text-gray-500 dark:text-gray-400">Activity feed cleared...</p>';
    sessionData.activityLog = [];
  }

  // Session Storage Management
  async function saveSession(session) {
    try {
      const response = await fetch(`${CONFIG.API_BASE}/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(session)
      });

      const result = await response.json();

      if (result.success) {
        console.log(`ðŸ’¾ Session "${session.className}" saved to MongoDB`);
        showNotification('Session saved to database', 'success');
      } else {
        throw new Error(result.message || 'Failed to save session');
      }
    } catch (error) {
      console.error('Failed to save session to MongoDB:', error);
      showNotification('Failed to save to database', 'error');

      // Fallback to localStorage
      const sessions = JSON.parse(localStorage.getItem(CONFIG.STORAGE_KEY) || '[]');
      sessions.unshift(session);
      if (sessions.length > 100) sessions.splice(100);
      localStorage.setItem(CONFIG.STORAGE_KEY, JSON.stringify(sessions));
    }
  }


  async function loadRecentSessions() {
    try {
      const response = await fetch(`${CONFIG.API_BASE}/sessions`);
      const result = await response.json();

      if (!result.success) throw new Error('Failed to load sessions');

      const sessions = result.sessions || [];
      const recentSessions = sessions.slice(0, 5);

      elements.recentSessions.innerHTML = '';

      if (recentSessions.length === 0) {
        elements.recentSessions.innerHTML = `
        <div class="text-center py-8 text-gray-500 dark:text-gray-400">
          <span class="material-icons text-4xl mb-2 block">school</span>
          <p>No sessions yet. Start your first class monitoring session!</p>
        </div>
      `;
        return;
      }

      recentSessions.forEach(session => {
        const div = document.createElement('div');
        div.className = 'flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors cursor-pointer';

        const avgScore = session.overallScores && session.overallScores.length > 0 ?
          Math.round(session.overallScores.reduce((sum, s) => sum + s.score, 0) / session.overallScores.length) : 0;

        const duration = Math.floor(session.duration / 60) || 0;
        const date = new Date(session.startTime || session.createdAt).toLocaleDateString();

        div.innerHTML = `
        <div class="flex items-center space-x-4">
          <div class="bg-indigo-100 dark:bg-indigo-900/50 p-3 rounded-full">
            <span class="material-icons text-indigo-500 dark:text-indigo-400">class</span>
          </div>
          <div>
            <p class="font-semibold text-gray-800 dark:text-white">${session.className}</p>
            <p class="text-sm text-gray-500 dark:text-gray-400">${date} â€¢ ${duration} minutes â€¢ ${avgScore}% engagement</p>
          </div>
        </div>
        <button class="view-session-btn text-indigo-600 dark:text-indigo-400 hover:text-indigo-700 dark:hover:text-indigo-300" data-session-id="${session._id || session.id}">
          <span class="material-icons">chevron_right</span>
        </button>
      `;

        div.querySelector('.view-session-btn').addEventListener('click', (e) => {
          e.stopPropagation();
          viewSessionDetails(session);
        });

        elements.recentSessions.appendChild(div);
      });

    } catch (error) {
      console.error('Failed to load recent sessions:', error);
      showNotification('Failed to load sessions from database', 'error');
    }
  }

  async function loadSessionHistory() {
    try {
      const response = await fetch(`${CONFIG.API_BASE}/sessions`);
      const result = await response.json();

      if (!result.success) throw new Error('Failed to load session history');

      const sessions = result.sessions || [];
      renderSessionHistory(sessions);

    } catch (error) {
      console.error('Failed to load session history:', error);
      showNotification('Failed to load session history', 'error');
    }
  }


  function renderSessionHistory(sessions) {
    elements.sessionHistoryContainer.innerHTML = '';

    if (sessions.length === 0) {
      elements.sessionHistoryContainer.innerHTML = `
        <div class="p-8 text-center">
          <span class="material-icons text-6xl text-gray-400 dark:text-gray-500 mb-4 block">history_edu</span>
          <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">No session history</h3>
          <p class="text-gray-500 dark:text-gray-400">Start monitoring classes to see your session history here.</p>
        </div>
      `;
      return;
    }

    const table = document.createElement('table');
    table.className = 'table';
    table.innerHTML = `
      <thead>
        <tr>
          <th>Class Name</th>
          <th>Date</th>
          <th>Duration</th>
          <th>Avg. Attention</th>
          <th>Students</th>
          <th>Status</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody></tbody>
    `;

    const tbody = table.querySelector('tbody');

    sessions.forEach(session => {
      const row = document.createElement('tr');

      const avgScore = session.overallScores.length > 0 ?
        Math.round(session.overallScores.reduce((sum, s) => sum + s.score, 0) / session.overallScores.length) : 0;

      const duration = Math.floor(session.duration / 60);
      const durationSeconds = session.duration % 60;
      const durationStr = `${duration}:${durationSeconds.toString().padStart(2, '0')}`;

      const date = new Date(session.startTime).toLocaleDateString();
      const time = new Date(session.startTime).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      let statusBadge = 'badge-success';
      let statusText = 'Completed';
      if (!session.endTime) {
        statusBadge = 'badge-warning';
        statusText = 'Interrupted';
      }

      let scoreBadge = 'badge-success';
      if (avgScore < 60) scoreBadge = 'badge-danger';
      else if (avgScore < 80) scoreBadge = 'badge-warning';

      row.innerHTML = `
        <td>
          <div class="flex items-center">
            <div class="bg-indigo-100 dark:bg-indigo-900/50 p-2 rounded-lg mr-3">
              <span class="material-icons text-indigo-600 dark:text-indigo-400 text-sm">class</span>
            </div>
            <div>
              <p class="font-medium text-gray-800 dark:text-white">${session.className}</p>
              <p class="text-sm text-gray-500 dark:text-gray-400">${time}</p>
            </div>
          </div>
        </td>
        <td class="text-sm text-gray-600 dark:text-gray-400">${date}</td>
        <td class="text-sm text-gray-600 dark:text-gray-400">${durationStr}</td>
        <td>
          <span class="badge ${scoreBadge}">${avgScore}%</span>
        </td>
        <td class="text-sm text-gray-600 dark:text-gray-400">${session.maxStudentsDetected}</td>
        <td>
          <span class="badge ${statusBadge}">${statusText}</span>
        </td>
        <td>
          <div class="flex items-center space-x-2">
            <button class="text-indigo-600 dark:text-indigo-400 hover:text-indigo-700 dark:hover:text-indigo-300 text-sm font-medium view-session-btn" data-session-id="${session.id}">
              View Report
            </button>
            <button class="text-red-600 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 text-sm font-medium delete-session-btn" data-session-id="${session.id}">
              Delete
            </button>
          </div>
        </td>
      `;

      // Add event listeners
      row.querySelector('.view-session-btn').addEventListener('click', () => {
        viewSessionDetails(session);
      });

      row.querySelector('.delete-session-btn').addEventListener('click', () => {
        deleteSession(session.id);
      });

      tbody.appendChild(row);
    });

    elements.sessionHistoryContainer.appendChild(table);
  }

  function viewSessionDetails(session) {
    // Load session data for analytics
    sessionData = { ...session };

    // Generate analytics
    generateAnalytics();

    // Switch to analytics view
    switchSection('analytics');
    updateNavState(document.querySelector('[data-section="analytics"]'));

    showNotification(`Viewing analytics for "${session.className}"`, 'info');
  }

  async function deleteSession(sessionId) {
    if (!confirm('Are you sure you want to delete this session?')) return;

    try {
      const response = await fetch(`${CONFIG.API_BASE}/sessions/${sessionId}`, {
        method: 'DELETE'
      });

      const result = await response.json();

      if (result.success) {
        showNotification('Session deleted successfully', 'success');
        await loadSessionHistory();
        await loadRecentSessions();
        await updateHomeStats();
      } else {
        throw new Error(result.message || 'Failed to delete session');
      }
    } catch (error) {
      console.error('Failed to delete session:', error);
      showNotification('Failed to delete session', 'error');
    }
  }


  function filterSessions() {
    const period = elements.filterPeriod.value;
    const sessions = Storage.load(CONFIG.STORAGE_KEY, []);

    let filteredSessions = sessions;
    const now = Date.now();

    switch (period) {
      case 'week':
        filteredSessions = sessions.filter(s => (now - s.startTime) <= 7 * 24 * 60 * 60 * 1000);
        break;
      case 'month':
        filteredSessions = sessions.filter(s => (now - s.startTime) <= 30 * 24 * 60 * 60 * 1000);
        break;
      case 'year':
        filteredSessions = sessions.filter(s => (now - s.startTime) <= 365 * 24 * 60 * 60 * 1000);
        break;
    }

    renderSessionHistory(filteredSessions);
  }

  // Home Statistics
  async function updateHomeStats() {
    try {
      const response = await fetch(`${CONFIG.API_BASE}/sessions`);
      const result = await response.json();

      const sessions = result.success ? result.sessions : [];

      // Calculate total session time
      const totalMinutes = sessions.reduce((sum, s) => sum + Math.floor(s.duration / 60), 0);
      const hours = Math.floor(totalMinutes / 60);
      const minutes = totalMinutes % 60;
      elements.totalSessionTime.textContent = `${hours}h ${minutes}m`;

      // Calculate average engagement
      let totalEngagement = 0;
      let engagementCount = 0;
      sessions.forEach(session => {
        if (session.overallScores && session.overallScores.length > 0) {
          const avgScore = session.overallScores.reduce((sum, s) => sum + s.score, 0) / session.overallScores.length;
          totalEngagement += avgScore;
          engagementCount++;
        }
      });
      const avgEngagement = engagementCount > 0 ? Math.round(totalEngagement / engagementCount) : 0;
      elements.avgEngagement.textContent = avgEngagement + '%';

      // Total classes
      elements.totalClasses.textContent = sessions.length;

      // Total insights
      const totalInsights = sessions.reduce((sum, s) => sum + (s.distractionEvents?.length || 0), 0) + sessions.length * 3;
      elements.totalInsights.textContent = totalInsights;

    } catch (error) {
      console.error('Failed to update home stats:', error);
      // Reset to defaults
      elements.totalSessionTime.textContent = '0h 0m';
      elements.avgEngagement.textContent = '0%';
      elements.totalClasses.textContent = '0';
      elements.totalInsights.textContent = '0';
    }
  }


  // Analytics Generation
  function generateAnalytics() {
    if (!sessionData || sessionData.overallScores.length === 0) {
      console.log('No session data for analytics');
      return;
    }

    // Calculate summary statistics
    const avgScore = Math.round(
      sessionData.overallScores.reduce((sum, s) => sum + s.score, 0) / sessionData.overallScores.length
    );

    const durationMinutes = Math.floor(sessionData.duration / 60);
    const durationSeconds = sessionData.duration % 60;
    const durationStr = `${durationMinutes}:${durationSeconds.toString().padStart(2, '0')}`;

    // Update analytics UI
    elements.analyticsClassInfo.textContent = sessionData.className;
    elements.analyticsDuration.textContent = durationStr;
    elements.analyticsAvgScore.textContent = avgScore + '%';
    elements.analyticsMaxStudents.textContent = sessionData.maxStudentsDetected;

    // Determine engagement level
    let engagementLevel = 'Low';
    if (avgScore >= 80) engagementLevel = 'High';
    else if (avgScore >= 60) engagementLevel = 'Medium';
    elements.analyticsEngagement.textContent = engagementLevel;

    // Generate detailed analytics
    generateInsights(avgScore);
    generatePatterns();
    generateRecommendations(avgScore, engagementLevel);

    // Initialize and update chart
    initAnalyticsChart();
  }

  function generateInsights(avgScore) {
    const insights = [];

    // Overall performance insight
    if (avgScore >= 85) {
      insights.push({
        icon: 'trending_up',
        color: 'text-green-600 dark:text-green-400',
        title: 'Excellent Engagement',
        description: `Outstanding class performance with ${avgScore}% average attention`
      });
    } else if (avgScore >= 70) {
      insights.push({
        icon: 'thumb_up',
        color: 'text-blue-600 dark:text-blue-400',
        title: 'Good Engagement',
        description: `Solid class performance with room for improvement`
      });
    } else {
      insights.push({
        icon: 'warning',
        color: 'text-amber-600 dark:text-amber-400',
        title: 'Attention Needed',
        description: `Class engagement below optimal levels`
      });
    }

    // Duration insight
    const durationMinutes = Math.floor(sessionData.duration / 60);
    if (durationMinutes > 45) {
      insights.push({
        icon: 'schedule',
        color: 'text-purple-600 dark:text-purple-400',
        title: 'Extended Session',
        description: `${durationMinutes}-minute session may impact attention levels`
      });
    }

    // Class size insight
    if (sessionData.maxStudentsDetected > 25) {
      insights.push({
        icon: 'groups',
        color: 'text-indigo-600 dark:text-indigo-400',
        title: 'Large Class',
        description: `Managing ${sessionData.maxStudentsDetected} students effectively`
      });
    } else if (sessionData.maxStudentsDetected > 0) {
      insights.push({
        icon: 'groups',
        color: 'text-indigo-600 dark:text-indigo-400',
        title: 'Class Size',
        description: `Optimal class size of ${sessionData.maxStudentsDetected} students`
      });
    }

    // Distraction events insight
    if (sessionData.distractionEvents && sessionData.distractionEvents.length > 0) {
      insights.push({
        icon: 'psychology',
        color: 'text-red-600 dark:text-red-400',
        title: 'Distraction Events',
        description: `${sessionData.distractionEvents.length} significant attention dips detected`
      });
    }

    renderInsights(insights);
  }

  function renderInsights(insights) {
    elements.keyInsights.innerHTML = '';

    insights.forEach(insight => {
      const div = document.createElement('div');
      div.className = 'flex items-start gap-3 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg';
      div.innerHTML = `
        <span class="material-icons ${insight.color} mt-1">${insight.icon}</span>
        <div>
          <p class="font-semibold text-gray-800 dark:text-white">${insight.title}</p>
          <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">${insight.description}</p>
        </div>
      `;
      elements.keyInsights.appendChild(div);
    });
  }

  function generatePatterns() {
    const patterns = [];

    // Analyze attention timeline for patterns
    if (sessionData.classMetrics && sessionData.classMetrics.length > 10) {
      const firstHalf = sessionData.classMetrics.slice(0, Math.floor(sessionData.classMetrics.length / 2));
      const secondHalf = sessionData.classMetrics.slice(Math.floor(sessionData.classMetrics.length / 2));

      const firstHalfAvg = firstHalf.reduce((sum, m) => sum + (m.attentivePercent || 0), 0) / firstHalf.length;
      const secondHalfAvg = secondHalf.reduce((sum, m) => sum + (m.attentivePercent || 0), 0) / secondHalf.length;

      if (secondHalfAvg < firstHalfAvg - 10) {
        patterns.push({
          icon: 'trending_down',
          title: 'Attention Fatigue',
          description: 'Class attention declined significantly in the second half'
        });
      } else if (secondHalfAvg > firstHalfAvg + 10) {
        patterns.push({
          icon: 'trending_up',
          title: 'Growing Engagement',
          description: 'Class attention improved as session progressed'
        });
      } else {
        patterns.push({
          icon: 'horizontal_rule',
          title: 'Consistent Attention',
          description: 'Class maintained steady attention throughout session'
        });
      }
    }

    // Phone usage pattern
    if (sessionData.classMetrics) {
      const phoneEvents = sessionData.classMetrics.filter(m => m.phonePercent > 20);
      if (phoneEvents.length > 0) {
        patterns.push({
          icon: 'phone_android',
          title: 'Phone Usage Detected',
          description: `Phone usage occurred in ${Math.round((phoneEvents.length / sessionData.classMetrics.length) * 100)}% of session`
        });
      } else {
        patterns.push({
          icon: 'phone_disabled',
          title: 'Minimal Phone Usage',
          description: 'Very low phone usage throughout the session'
        });
      }
    }

    renderPatterns(patterns);
  }

  function renderPatterns(patterns) {
    elements.attentionPatterns.innerHTML = '';

    patterns.forEach(pattern => {
      const div = document.createElement('div');
      div.className = 'flex items-start gap-3 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg';
      div.innerHTML = `
        <span class="material-icons text-purple-600 dark:text-purple-400 mt-1">${pattern.icon}</span>
        <div>
          <p class="font-semibold text-gray-800 dark:text-white">${pattern.title}</p>
          <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">${pattern.description}</p>
        </div>
      `;
      elements.attentionPatterns.appendChild(div);
    });
  }

  function generateRecommendations(avgScore, engagementLevel) {
    const recommendations = [];

    if (avgScore < 60) {
      recommendations.push({
        title: 'Break Content Into Segments',
        description: 'Consider shorter lesson segments with interactive breaks to maintain attention'
      });
      recommendations.push({
        title: 'Increase Interaction',
        description: 'Add more questions, polls, or hands-on activities to engage students'
      });
      recommendations.push({
        title: 'Check Environment',
        description: 'Ensure classroom lighting, temperature, and seating are optimal'
      });
      recommendations.push({
        title: 'Use Visual Aids',
        description: 'Incorporate more visual elements like diagrams, videos, or presentations'
      });
    } else if (avgScore < 80) {
      recommendations.push({
        title: 'Maintain Current Approach',
        description: 'Good engagement levels - minor adjustments may help reach excellence'
      });
      recommendations.push({
        title: 'Monitor Distraction Points',
        description: 'Identify and address specific moments of attention loss'
      });
      recommendations.push({
        title: 'Vary Teaching Methods',
        description: 'Mix lecture with group work and individual activities'
      });
      recommendations.push({
        title: 'Regular Check-ins',
        description: 'Ask questions periodically to ensure comprehension and engagement'
      });
    } else {
      recommendations.push({
        title: 'Excellent Teaching Method',
        description: 'Current approach is highly effective - consider replicating this format'
      });
      recommendations.push({
        title: 'Share Best Practices',
        description: 'Document successful strategies for future sessions and colleagues'
      });
      recommendations.push({
        title: 'Advanced Challenges',
        description: 'Consider adding more complex activities to maintain high engagement'
      });
      recommendations.push({
        title: 'Student Leadership',
        description: 'Engage high-performing students as peer mentors or discussion leaders'
      });
    }

    // Additional recommendations based on session data
    if (sessionData.duration > 45 * 60) {
      recommendations.push({
        title: 'Consider Shorter Sessions',
        description: 'Extended sessions may lead to attention fatigue - try 45-minute blocks'
      });
    }

    if (sessionData.distractionEvents && sessionData.distractionEvents.length > 3) {
      recommendations.push({
        title: 'Address Environment',
        description: 'Multiple distraction events suggest external factors - check classroom conditions'
      });
    }

    renderRecommendations(recommendations);
  }

  function renderRecommendations(recommendations) {
    elements.recommendations.innerHTML = '';

    recommendations.forEach(rec => {
      const div = document.createElement('div');
      div.className = 'p-4 bg-white dark:bg-gray-700 rounded-lg border border-indigo-200 dark:border-indigo-800';
      div.innerHTML = `
        <h4 class="font-semibold text-gray-800 dark:text-white mb-2">${rec.title}</h4>
        <p class="text-sm text-gray-600 dark:text-gray-400">${rec.description}</p>
      `;
      elements.recommendations.appendChild(div);
    });
  }

  // ADD THIS FUNCTION AFTER renderRecommendations
// ENHANCED AI ANALYSIS FUNCTION WITH BETTER FORMATTING
async function generateAiAnalysis() {
    if (!sessionData || !sessionData.className) {
        showNotification('No session data available for AI analysis', 'warning');
        return;
    }

    const container = elements.aiAnalysisContainer;
    const loading = elements.aiAnalysisLoading;
    const content = elements.aiAnalysisContent;
    
    // Show loading
    container.classList.remove('hidden');
    loading.classList.remove('hidden');
    content.classList.add('hidden');
    
    try {
        // Prepare session data for AI
        const aiSessionData = {
            ...sessionData,
            // Flatten overallScores if they're objects
            overallScores: sessionData.overallScores.map(score => 
                typeof score === 'object' && score.score ? score.score : score
            )
        };

        const response = await fetch('http://localhost:8000/api/ai-summary', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(aiSessionData)
        });
        
        const result = await response.json();
        
        // Hide loading
        loading.classList.add('hidden');
        content.classList.remove('hidden');
        
        if (result.success) {
            // Format the AI response properly
            const formattedContent = formatAIResponse(result.ai_summary, result);
            content.innerHTML = formattedContent;
            showNotification('AI analysis completed successfully!', 'success');
        } else {
            throw new Error(result.message);
        }
        
    } catch (error) {
        console.error('AI Analysis Error:', error);
        loading.classList.add('hidden');
        content.classList.remove('hidden');
        content.innerHTML = `
            <div class="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg border border-red-200 dark:border-red-700">
                <div class="flex items-center text-red-600 dark:text-red-400">
                    <span class="material-icons mr-2">error</span>
                    <span>Failed to generate AI analysis: ${error.message}</span>
                </div>
            </div>
        `;
        showNotification('Failed to generate AI analysis', 'error');
    }
}

// ADD THIS NEW FUNCTION RIGHT AFTER generateAiAnalysis
function formatAIResponse(rawText, result) {
    if (!rawText) return '<p class="text-gray-500">No analysis available</p>';
    
    // Split into sections and clean up formatting
    let formatted = rawText
        // Remove multiple asterisks and clean headers
        .replace(/\*\*([^*]+)\*\*/g, '<h4 class="ai-section-header">$1</h4>')
        .replace(/\*([^*]+)\*/g, '<strong>$1</strong>')
        
        // Format numbered sections
        .replace(/^(\d+)\.\s*\*\*([^*]+)\*\*/gm, '<h4 class="ai-section-header"><span class="ai-number">$1.</span> $2</h4>')
        
        // Format bullet points
        .replace(/^•\s*(.+)$/gm, '<li class="ai-bullet">$1</li>')
        .replace(/^\*\s*(.+)$/gm, '<li class="ai-bullet">$1</li>')
        
        // Format paragraphs
        .replace(/\n\n/g, '</p><p class="ai-paragraph">')
        .replace(/\n/g, '<br>')
        
        // Wrap in paragraph tags
        .replace(/^/, '<p class="ai-paragraph">')
        .replace(/$/, '</p>')
        
        // Fix list formatting
        .replace(/(<li class="ai-bullet">.*?<\/li>)/gs, (match) => {
            return '<ul class="ai-bullet-list">' + match + '</ul>';
        })
        
        // Clean up empty paragraphs
        .replace(/<p class="ai-paragraph"><\/p>/g, '')
        .replace(/<p class="ai-paragraph"><br><\/p>/g, '');

    return `
        <div class="ai-formatted-content">
            <div class="ai-header">
                <div class="flex items-center space-x-3 mb-6">
                    <div class="bg-purple-100 dark:bg-purple-900/50 p-2 rounded-full">
                        <span class="material-icons text-purple-600 dark:text-purple-400">psychology</span>
                    </div>
                    <div>
                        <h3 class="text-lg font-semibold text-gray-800 dark:text-white">AI Session Analysis</h3>
                        <p class="text-sm text-gray-500 dark:text-gray-400">Generated on ${new Date().toLocaleString()}</p>
                    </div>
                </div>
            </div>
            <div class="ai-content-body">
                ${formatted}
            </div>
            <div class="ai-footer mt-6 pt-4 border-t border-purple-200 dark:border-purple-700">
                <div class="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
                    <div class="flex items-center space-x-4">
                        <span class="flex items-center">
                            <span class="material-icons text-xs mr-1">smart_toy</span>
                            AI Model: ${result.ai_model_used || 'EduGauge AI'}
                        </span>
                        <span class="flex items-center">
                            <span class="material-icons text-xs mr-1">access_time</span>
                            Analysis Type: ${result.analysis_type || 'comprehensive'}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    `;
}


  // Chart Management
  function initAnalyticsChart() {
    if (!elements.attentionChart) return;

    const ctx = elements.attentionChart.getContext('2d');

    // Destroy existing chart
    if (state.attentionChart) {
      state.attentionChart.destroy();
    }

    state.attentionChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Overall Attention (%)',
          data: [],
          borderColor: '#4F46E5',
          backgroundColor: 'rgba(70, 229, 229, 0.32)',
          tension: 0.4,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { intersect: false, mode: 'index' },
        plugins: {
          legend: {
            display: true,
            position: 'top',
            labels: {
              color: getComputedStyle(document.documentElement).getPropertyValue('--tw-text-opacity') ? '#fff' : '#000'
            }
          },
          tooltip: {
            callbacks: {
              title: function (tooltipItems) {
                return 'Time: ' + tooltipItems[0].label;
              }
            }
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            title: { display: true, text: 'Attention %' },
            ticks: {
              color: getComputedStyle(document.documentElement).getPropertyValue('--tw-text-opacity') ? '#9ca3af' : '#6b7280'
            },
            grid: {
              color: getComputedStyle(document.documentElement).getPropertyValue('--tw-text-opacity') ? '#374151' : '#e5e7eb'
            }
          },
          x: {
            title: { display: true, text: 'Time' },
            ticks: {
              color: getComputedStyle(document.documentElement).getPropertyValue('--tw-text-opacity') ? '#9ca3af' : '#6b7280'
            },
            grid: {
              color: getComputedStyle(document.documentElement).getPropertyValue('--tw-text-opacity') ? '#374151' : '#e5e7eb'
            }
          }
        }
      }
    });

    updateAnalyticsChart();
  }

  function updateTimelineChart() {
    // This would be called during live session for real-time updates
    if (state.attentionChart && sessionData.overallScores.length > 0) {
      // Update chart with latest data (limit to last 50 points for performance)
      const recentScores = sessionData.overallScores.slice(-50);
      const labels = recentScores.map(score => {
        const minutes = Math.floor(score.time / 60);
        const seconds = score.time % 60;
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
      });

      state.attentionChart.data.labels = labels;
      state.attentionChart.data.datasets[0].data = recentScores.map(score => score.score);
      state.attentionChart.update('none'); // No animation for better performance
    }
  }

  function updateAnalyticsChart() {
    if (!state.attentionChart || !sessionData.overallScores || sessionData.overallScores.length === 0) return;

    const labels = sessionData.overallScores.map(score => {
      const minutes = Math.floor(score.time / 60);
      const seconds = score.time % 60;
      return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    });

    const attentionData = sessionData.overallScores.map(score => score.score);

    state.attentionChart.data.labels = labels;
    state.attentionChart.data.datasets[0].data = attentionData;
    state.attentionChart.update();
  }

  // Settings Management
  async function loadSettings() {
    try {
      const response = await fetch(`${CONFIG.API_BASE}/settings`);
      const result = await response.json();

      const settings = result.success ? result.settings : {
        autoSave: true,
        lowAttentionAlerts: true,
        sessionReports: false,
        dataRetention: 365,
        sensitivity: 7,
        updateFrequency: 500,
        privacyMode: true,
        cameraQuality: 'medium'
      };

      elements.autoSaveSetting.checked = settings.autoSave;
      elements.lowAttentionAlerts.checked = settings.lowAttentionAlerts;
      elements.sessionReports.checked = settings.sessionReports;
      elements.dataRetention.value = settings.dataRetention;
      elements.sensitivityRange.value = settings.sensitivity;
      elements.updateFrequency.value = settings.updateFrequency;
      elements.privacyMode.checked = settings.privacyMode;
      elements.cameraQuality.value = settings.cameraQuality;

    } catch (error) {
      console.error('Failed to load settings:', error);
      // Use defaults
    }
  }

  async function saveSettings() {
    const settings = {
      autoSave: elements.autoSaveSetting.checked,
      lowAttentionAlerts: elements.lowAttentionAlerts.checked,
      sessionReports: elements.sessionReports.checked,
      dataRetention: parseInt(elements.dataRetention.value),
      sensitivity: parseInt(elements.sensitivityRange.value),
      updateFrequency: parseInt(elements.updateFrequency.value),
      privacyMode: elements.privacyMode.checked,
      cameraQuality: elements.cameraQuality.value
    };

    try {
      const response = await fetch(`${CONFIG.API_BASE}/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });

      const result = await response.json();

      if (result.success) {
        showNotification('Settings saved to database', 'success');
      } else {
        throw new Error('Failed to save settings');
      }
    } catch (error) {
      console.error('Failed to save settings:', error);
      showNotification('Failed to save settings to database', 'error');
    }
  }


  function resetSettings() {
    if (confirm('Reset all settings to default values?')) {
      Storage.remove(CONFIG.SETTINGS_KEY);
      loadSettings();
      showNotification('Settings reset to defaults', 'success');
    }
  }

  // Notification System
  function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
      <div class="flex items-start">
        <span class="material-icons text-lg mr-2">${getNotificationIcon(type)}</span>
        <div class="flex-1">
          <p class="font-medium">${message}</p>
        </div>
        <button class="ml-2 text-gray-400 hover:text-gray-600" onclick="this.parentElement.parentElement.remove()">
          <span class="material-icons text-sm">close</span>
        </button>
      </div>
    `;

    document.body.appendChild(notification);

    // Auto remove after 5 seconds
    setTimeout(() => {
      if (notification.parentElement) {
        notification.remove();
      }
    }, 5000);

    // Update notification badge
    state.notifications.push({ message, type, timestamp: Date.now() });
    updateNotificationBadge();
  }

  function getNotificationIcon(type) {
    switch (type) {
      case 'success': return 'check_circle';
      case 'warning': return 'warning';
      case 'error': return 'error';
      default: return 'info';
    }
  }

  function updateNotificationBadge() {
    const recentNotifications = state.notifications.filter(n =>
      Date.now() - n.timestamp < 60000 // Last minute
    );

    if (recentNotifications.length > 0) {
      elements.notificationBadge.classList.remove('hidden');
    } else {
      elements.notificationBadge.classList.add('hidden');
    }
  }

  // Export and Share Functions
  function exportSessionData() {
    if (!sessionData || !sessionData.className) {
      showNotification('No session data to export', 'warning');
      return;
    }

    const exportData = {
      sessionInfo: {
        id: sessionData.id,
        className: sessionData.className,
        date: new Date(sessionData.startTime).toISOString(),
        duration: sessionData.duration
      },
      summary: {
        avgScore: sessionData.overallScores.length > 0 ?
          Math.round(sessionData.overallScores.reduce((sum, s) => sum + s.score, 0) / sessionData.overallScores.length) : 0,
        maxStudents: sessionData.maxStudentsDetected,
        totalFrames: sessionData.totalFrames,
        distractionEvents: sessionData.distractionEvents?.length || 0
      },
      timeline: sessionData.overallScores,
      classMetrics: sessionData.classMetrics,
      activityLog: sessionData.activityLog,
      insights: {
        screenshots: sessionData.screenshots?.length || 0,
        sessionComplete: !!sessionData.endTime
      }
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `EduGauge_${sessionData.className.replace(/\s+/g, '_')}_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);

    showNotification('Session data exported successfully', 'success');
  }

  function exportAllSessions() {
    const sessions = Storage.load(CONFIG.STORAGE_KEY, []);

    if (sessions.length === 0) {
      showNotification('No sessions to export', 'warning');
      return;
    }

    const exportData = {
      exportDate: new Date().toISOString(),
      totalSessions: sessions.length,
      sessions: sessions.map(session => ({
        ...session,
        exportSummary: {
          avgScore: session.overallScores.length > 0 ?
            Math.round(session.overallScores.reduce((sum, s) => sum + s.score, 0) / session.overallScores.length) : 0,
          duration: session.duration,
          maxStudents: session.maxStudentsDetected
        }
      }))
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `EduGauge_AllSessions_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);

    showNotification(`Exported ${sessions.length} sessions successfully`, 'success');
  }

  function shareReport() {
    if (!sessionData || !sessionData.className) {
      showNotification('No session data to share', 'warning');
      return;
    }

    const avgScore = sessionData.overallScores.length > 0 ?
      Math.round(sessionData.overallScores.reduce((sum, s) => sum + s.score, 0) / sessionData.overallScores.length) : 0;

    const shareText = `ðŸ“Š EduGauge Session Report\n` +
      `ðŸ“š Class: ${sessionData.className}\n` +
      `ðŸ“ˆ Average Attention: ${avgScore}%\n` +
      `ðŸ‘¥ Max Class Size: ${sessionData.maxStudentsDetected}\n` +
      `â±ï¸ Duration: ${Math.floor(sessionData.duration / 60)} minutes\n` +
      `\nGenerated with EduGauge - Classroom Attention Monitor`;

    if (navigator.share) {
      navigator.share({
        title: 'EduGauge Session Report',
        text: shareText,
        url: window.location.href
      }).then(() => {
        showNotification('Report shared successfully', 'success');
      }).catch(() => {
        fallbackShare(shareText);
      });
    } else {
      fallbackShare(shareText);
    }
  }

  function fallbackShare(text) {
    navigator.clipboard.writeText(text).then(() => {
      showNotification('Report summary copied to clipboard!', 'success');
    }).catch(() => {
      showNotification('Unable to copy to clipboard', 'error');
    });
  }

  // Data Management
  function clearAllData() {
    if (confirm('This will delete all session data, settings, and history. This cannot be undone. Continue?')) {
      Storage.clear();

      // Reset UI
      elements.recentSessions.innerHTML = '<div class="text-center py-8 text-gray-500 dark:text-gray-400"><p>No sessions yet.</p></div>';
      elements.sessionHistoryContainer.innerHTML = '<div class="p-8 text-center"><p class="text-gray-500 dark:text-gray-400">No session history</p></div>';

      updateHomeStats();
      loadSettings();

      showNotification('All data cleared successfully', 'success');
    }
  }

  function showHelp() {
    const helpContent = `
      <div class="modal-overlay" id="helpModal">
        <div class="modal-content">
          <div class="flex justify-between items-center mb-6">
            <h2 class="text-xl font-bold text-gray-800 dark:text-white">EduGauge Help</h2>
            <button class="text-gray-500 hover:text-gray-700" onclick="document.getElementById('helpModal').remove()">
              <span class="material-icons">close</span>
            </button>
          </div>

          <div class="space-y-4 text-sm">
            <div>
              <h3 class="font-semibold text-gray-800 dark:text-white mb-2">Getting Started</h3>
              <p class="text-gray-600 dark:text-gray-400">1. Enter your class name on the home page<br>
              2. Click "Start Monitoring" to begin a session<br>
              3. Allow camera access when prompted<br>
              4. Monitor real-time attention levels<br>
              5. End the session to view detailed analytics</p>
            </div>

            <div>
              <h3 class="font-semibold text-gray-800 dark:text-white mb-2">Features</h3>
              <p class="text-gray-600 dark:text-gray-400">â€¢ Real-time attention monitoring<br>
              â€¢ Generalized analytics (no individual tracking)<br>
              â€¢ Session history and reports<br>
              â€¢ Export data in JSON format<br>
              â€¢ Dark mode support<br>
              â€¢ Customizable settings</p>
            </div>

            <div>
              <h3 class="font-semibold text-gray-800 dark:text-white mb-2">Privacy</h3>
              <p class="text-gray-600 dark:text-gray-400">EduGauge focuses on class-level metrics without tracking individual students. All data is stored locally on your device.</p>
            </div>

            <div>
              <h3 class="font-semibold text-gray-800 dark:text-white mb-2">Support</h3>
              <p class="text-gray-600 dark:text-gray-400">For technical support or questions, please contact your system administrator.</p>
            </div>
          </div>
        </div>
      </div>
    `;

    document.body.insertAdjacentHTML('beforeend', helpContent);
  }

  // Event Listeners
  function initEventListeners() {
    // Session controls
    elements.startBtn?.addEventListener('click', startSession);
    elements.endSessionBtn?.addEventListener('click', endSession);
    elements.pauseBtn?.addEventListener('click', pauseSession);
    elements.screenshotBtn?.addEventListener('click', takeScreenshot);
    elements.recordBtn?.addEventListener('click', toggleRecording);

    // Theme toggle
    elements.darkModeToggle?.addEventListener('click', toggleDarkMode);

    // Activity feed
    elements.clearActivityBtn?.addEventListener('click', clearActivityFeed);

    // Home page
    elements.viewAllSessionsBtn?.addEventListener('click', () => {
      switchSection('history');
      updateNavState(document.querySelector('[data-section="history"]'));
    });
    elements.refreshStatsBtn?.addEventListener('click', updateHomeStats);
    elements.notificationBtn?.addEventListener('click', () => {
      showNotification('No new notifications', 'info');
    });

    // History page
    elements.filterSessionsBtn?.addEventListener('click', () => {
      elements.sessionFilter?.classList.toggle('hidden');
    });
    elements.applyFilterBtn?.addEventListener('click', filterSessions);
    elements.exportAllBtn?.addEventListener('click', exportAllSessions);

    // Settings
    elements.saveSettingsBtn?.addEventListener('click', saveSettings);
    elements.resetSettingsBtn?.addEventListener('click', resetSettings);

    // Footer actions
    elements.clearDataBtn?.addEventListener('click', clearAllData);
    elements.helpBtn?.addEventListener('click', showHelp);

    // Export and share
    elements.exportReportBtn?.addEventListener('click', exportSessionData);
    elements.shareReportBtn?.addEventListener('click', shareReport);

     // ADD THIS LINE:
    elements.generateAiAnalysis?.addEventListener('click', generateAiAnalysis);

    // Chart filters
    document.addEventListener('click', (e) => {
      if (e.target.classList.contains('timeline-filter')) {
        document.querySelectorAll('.timeline-filter').forEach(btn => {
          btn.classList.remove('bg-indigo-100', 'dark:bg-indigo-900/50', 'text-indigo-600', 'dark:text-indigo-400');
          btn.classList.add('text-gray-500', 'dark:text-gray-400');
        });
        e.target.classList.add('bg-indigo-100', 'dark:bg-indigo-900/50', 'text-indigo-600', 'dark:text-indigo-400');
        e.target.classList.remove('text-gray-500', 'dark:text-gray-400');
      }
    });

    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
      if (state.isMonitoring) {
        endSession();
      }
    });

    // Handle visibility changes
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && state.isMonitoring) {
        console.log('Page hidden');
      } else if (!document.hidden && state.isMonitoring) {
        console.log('Page visible');
      }
    });

    // Update notification badge periodically
    setInterval(updateNotificationBadge, 30000);
  }

  // Initialize Application
  document.addEventListener('DOMContentLoaded', init);

  

})();