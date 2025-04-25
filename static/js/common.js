// Common utilities and shared variables for FramePack

// Global variables
const timeline = [];







// WebSocket handling
let jobSocket = null;
const jobEventListeners = [];

// Variables for edit mode



// Variables for batch upload


// DOM elements shared across modules
const elements = {
    // Inputs & controls
    globalPrompt: null,
    negativePrompt: null,
    seedInput: null,
    stepsInput: null,
    stepsValue: null,
    guidanceInput: null,
    guidanceValue: null,
    resolutionInput: null,
    enableAdaptive: null,
    durationInput: null,
    
    // Buttons
    addFrameBtn: null,
    previewTimelineBtn: null,
    clearTimelineBtn: null,
    generateBtn: null,
    confirmUploadBtn: null,
    
    // Modal elements
    imageUpload: null,
    segmentPrompt: null,
    segmentDuration: null,
    errorMessage: null,
    
    // Display areas
    timelineContainer: null,
    progressArea: null,
    noProgressArea: null,
    progressBar: null,
    progressMessage: null,
    previewImageContainer: null,
    resultArea: null,
    noResultArea: null,
    resultVideo: null,
    downloadBtn: null,
    livePeekContainer: null,
    livePeekVideo: null,
    
    // New UI elements
    uploadImagesBtn: null,
    generateVideoBtn: null,
    saveJobBtn: null,
    progressContainer: null,
    progressStatus: null,
    previewContainer: null,
    previewImage: null,
    
    // Form inputs
    resolution: null,
    steps: null,
    guidanceScale: null,
    useTeacache: null,
    enableAdaptiveMemory: null,
    outputFormat: null,
    
    // Upload modal elements
    fileInput: null,
    uploadDropArea: null,
    imageUploadContainer: null,
    addToTimelineBtn: null,
    
    // Frame edit modal elements
    frameEditImage: null,
    frameDuration: null,
    replaceImageBtn: null,
    deleteFrameBtn: null,
    saveFrameBtn: null,
    
    // Video viewer modal elements
    modalVideo: null,
    modalDownloadBtn: null,
    
    // Job queue tab elements
    jobsContainer: null,
    jobDetailContainer: null,
    refreshJobsBtn: null,
    
    // Output tab elements
    outputsContainer: null,
    refreshOutputsBtn: null
};

// DOM elements for new UI elements
const progressElements = {
    currentStep: null,
    generatedFrames: null,
    generatedDuration: null,
    currentSegment: null,
    progressBar: null,
    progressMessage: null
};

// Initialize current image related elements





// DOM elements for job queue tab
const jobQueueElements = {
    jobsContainer: null,
    jobDetailContainer: null,
    refreshJobsBtn: null
};

// Utility functions
function formatTimestamp(timestamp) {
    if (!timestamp) return 'Unknown date';
    
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
}

// Show a message using a toast or alert
function showMessage(message, type) {
    // You could implement this with a toast notification
    console.log(`${type.toUpperCase()}: ${message}`);
    
    // For now, use a simple alert
    if (type === 'error') {
        alert(`Error: ${message}`);
    } else if (type === 'success') {
        alert(`Success: ${message}`);
    }
}

// Initialize all DOM elements
function initElements() {
    // Main interface elements
    elements.uploadImagesBtn = document.getElementById('uploadImagesBtn');
    elements.generateVideoBtn = document.getElementById('generateVideoBtn');
    elements.saveJobBtn = document.getElementById('saveJobBtn');
    elements.timelineContainer = document.getElementById('timelineContainer');
    elements.progressContainer = document.getElementById('progressContainer');
    elements.progressBar = document.getElementById('progressBar');
    elements.progressStatus = document.getElementById('progressStatus');
    elements.previewContainer = document.getElementById('previewContainer');
    elements.previewImage = document.getElementById('previewImage');
    
    // Form inputs
    elements.globalPrompt = document.getElementById('globalPrompt');
    elements.negativePrompt = document.getElementById('negativePrompt');
    elements.resolution = document.getElementById('resolution');
    elements.steps = document.getElementById('steps');
    elements.guidanceScale = document.getElementById('guidanceScale');
    elements.useTeacache = document.getElementById('useTeacache');
    elements.enableAdaptiveMemory = document.getElementById('enableAdaptiveMemory');
    elements.outputFormat = document.getElementById('outputFormat');
    
    // Upload modal elements
    elements.fileInput = document.getElementById('fileInput');
    elements.uploadDropArea = document.getElementById('uploadDropArea');
    elements.imageUploadContainer = document.getElementById('imageUploadContainer');
    elements.addToTimelineBtn = document.getElementById('addToTimelineBtn');
    
    // Frame edit modal elements
    elements.frameEditImage = document.getElementById('frameEditImage');
    elements.frameDuration = document.getElementById('frameDuration');
    elements.replaceImageBtn = document.getElementById('replaceImageBtn');
    elements.deleteFrameBtn = document.getElementById('deleteFrameBtn');
    elements.saveFrameBtn = document.getElementById('saveFrameBtn');
    
    // Video viewer modal elements
    elements.modalVideo = document.getElementById('modalVideo');
    elements.modalDownloadBtn = document.getElementById('modalDownloadBtn');
    
    // Initialize job queue elements
    jobQueueElements.jobsContainer = document.getElementById('jobsContainer');
    jobQueueElements.jobDetailContainer = document.getElementById('jobDetailContainer');
    jobQueueElements.refreshJobsBtn = document.getElementById('refreshJobsBtn');
    
    // Initialize output tab elements
    elements.outputsContainer = document.getElementById('outputsContainer');
    elements.refreshOutputsBtn = document.getElementById('refreshOutputsBtn');
}

// Add this function to ensure horizontal layout is maintained
function enforceHorizontalLayout() {
    if (elements.timelineContainer) {
        // Force horizontal layout with inline styles as a fallback
        elements.timelineContainer.style.display = 'flex';
        elements.timelineContainer.style.flexDirection = 'row';
        elements.timelineContainer.style.flexWrap = 'wrap';
        elements.timelineContainer.style.alignItems = 'flex-start';
        
        // Ensure the class is present
        if (!elements.timelineContainer.classList.contains('timeline-container')) {
            elements.timelineContainer.classList.add('timeline-container');
        }
    }
}

// Check if image path exists
async function checkImageExists(imagePath) {
    try {
        // If the path is a file:// URL or absolute path, we can't fetch it directly
        if (imagePath.startsWith('file://') || imagePath.match(/^[A-Z]:\\/)) {
            // For local paths, we need to check if they exist through the server
            // Use the server API to check if the file exists
            const encodedPath = encodeURIComponent(imagePath);
            const response = await fetch(`/api/check_file_exists?path=${encodedPath}`);
            const data = await response.json();
            return data.exists;
        } else if (imagePath.startsWith('/uploads/') || imagePath.startsWith('/static/')) {
            // For server paths, we can check with a HEAD request
            const response = await fetch(imagePath, { method: 'HEAD' });
            return response.ok;
        } else {
            // For other URLs, attempt regular fetch
            const response = await fetch(imagePath, { method: 'HEAD' });
            return response.ok;
        }
    } catch (error) {
        console.error('Error checking image:', error);
        return false;
    }
}

// WebSocket functionality
function connectJobWebsocket(jobId) {
    // Close existing connection if any
    if (jobSocket && jobSocket.readyState !== WebSocket.CLOSED) {
        jobSocket.close();
    }
    
    // Create a new WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/job/${jobId}`;
    
    console.log(`Connecting to WebSocket: ${wsUrl}`);
    jobSocket = new WebSocket(wsUrl);
    
    jobSocket.onopen = function() {
        console.log('WebSocket connection established');
    };
    
    jobSocket.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            console.log('WebSocket message received:', data);
            
            // Notify all listeners
            jobEventListeners.forEach(listener => {
                try {
                    listener(data);
                } catch (listenerError) {
                    console.error('Error in job event listener:', listenerError);
                }
            });
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };
    
    jobSocket.onclose = function(event) {
        if (event.wasClean) {
            console.log(`WebSocket connection closed cleanly, code=${event.code}, reason=${event.reason}`);
        } else {
            console.warn('WebSocket connection died');
            // Try to reconnect after a delay if it wasn't intentionally closed
            setTimeout(() => {
                if (jobId) {
                    connectJobWebsocket(jobId);
                }
            }, 5000);
        }
    };
    
    jobSocket.onerror = function(error) {
        console.error('WebSocket error:', error);
    };
    
    return jobSocket;
}

function disconnectJobWebsocket() {
    if (jobSocket && jobSocket.readyState !== WebSocket.CLOSED) {
        jobSocket.close();
        jobSocket = null;
    }
}

function addJobEventListener(callback) {
    jobEventListeners.push(callback);
    return jobEventListeners.length - 1; // Return index for removal
}

function removeJobEventListener(index) {
    if (index >= 0 && index < jobEventListeners.length) {
        jobEventListeners.splice(index, 1);
    }
}

// Export shared functions and variables
export {
    timeline,
    elements,
    jobQueueElements,
    progressElements,
    formatTimestamp,
    showMessage,
    initElements,
    enforceHorizontalLayout,
    checkImageExists,
    connectJobWebsocket,
    disconnectJobWebsocket,
    addJobEventListener,
    removeJobEventListener
}; 