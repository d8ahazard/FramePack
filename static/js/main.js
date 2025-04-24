// Main JavaScript for FramePack

// Global variables
let timeline = [];
let currentJobId = null;
let statusCheckInterval = null;
let uploadModal = null;
let errorModal = null;

// Variables for edit mode
let editElements = {};
let isEditingMode = false;
let keepCurrentImage = false;
let currentEditIndex = -1;

// Variables for batch upload
let selectedFiles = [];
let editItemModal = null;
let currentEditId = null;

// DOM elements
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
    useTeacache: null,
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
    livePeekVideo: null
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
let currentImageContainer = null;
let currentImage = null;
let keepImageBtn = null;
let replaceImageBtn = null;
let imageUploadContainer = null;

// DOM elements for job queue tab
const jobQueueElements = {
    jobsContainer: null,
    jobDetailContainer: null,
    refreshJobsBtn: null
};

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    // Initialize Bootstrap modals
    const uploadModalElement = document.getElementById('uploadImagesModal');
    if (uploadModalElement) {
        uploadModal = new bootstrap.Modal(uploadModalElement);
    }
    
    const errorModalElement = document.getElementById('errorModal');
    if (errorModalElement) {
        errorModal = new bootstrap.Modal(errorModalElement);
    }

    const frameEditModalElement = document.getElementById('frameEditModal');
    if (frameEditModalElement) {
        editItemModal = new bootstrap.Modal(frameEditModalElement);
    }
    
    const videoViewerModalElement = document.getElementById('videoViewerModal');
    if (videoViewerModalElement) {
        videoViewerModal = new bootstrap.Modal(videoViewerModalElement);
    }
    
    // Initialize UI elements
    initElements();
    
    // Initialize event listeners
    initEventListeners();
    
    // Initialize timeline drop zone
    initTimelineDropZone();
    
    // Add sort buttons event listeners
    const sortAscBtn = document.getElementById('sortAscBtn');
    const sortDescBtn = document.getElementById('sortDescBtn');
    
    if (sortAscBtn) {
        sortAscBtn.addEventListener('click', () => sortTimeline('asc'));
    }
    
    if (sortDescBtn) {
        sortDescBtn.addEventListener('click', () => sortTimeline('desc'));
    }
    
    // Check for running job
    checkForRunningJob();
    
    // Initial load of outputs in the outputs tab
    loadOutputs();
});

// Initialize all DOM elements
function initElements() {
    // Main interface elements
    elements.uploadImagesBtn = document.getElementById('uploadImagesBtn');
    elements.generateVideoBtn = document.getElementById('generateVideoBtn');
    elements.timelineContainer = document.getElementById('timelineContainer');
    elements.progressContainer = document.getElementById('progressContainer');
    elements.progressBar = document.getElementById('progressBar');
    elements.progressStatus = document.getElementById('progressStatus');
    elements.previewContainer = document.getElementById('previewContainer');
    elements.previewImage = document.getElementById('previewImage');
    
    // Form inputs
    elements.frameRate = document.getElementById('frameRate');
    elements.frameTime = document.getElementById('frameTime');
    elements.transitionTime = document.getElementById('transitionTime');
    elements.outputWidth = document.getElementById('outputWidth');
    elements.outputHeight = document.getElementById('outputHeight');
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

// Show upload modal
function showUploadModal() {
    // Clear previous uploads
    if (elements.imageUploadContainer) {
        elements.imageUploadContainer.innerHTML = '';
    }
    selectedFiles = [];
    
    // Show the modal
    if (uploadModal) {
        uploadModal.show();
    }
}

// Initialize event listeners
function initEventListeners() {
    // Button click events
    if (elements.uploadImagesBtn) {
        elements.uploadImagesBtn.addEventListener('click', showUploadModal);
    }
    
    if (elements.generateVideoBtn) {
        elements.generateVideoBtn.addEventListener('click', startGeneration);
    }
    
    // File input change event
    if (elements.fileInput) {
        elements.fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Upload drop area events
    if (elements.uploadDropArea) {
        elements.uploadDropArea.addEventListener('dragover', handleDragOver);
        elements.uploadDropArea.addEventListener('dragleave', handleDragLeave);
        elements.uploadDropArea.addEventListener('drop', handleFileDrop);
        elements.uploadDropArea.addEventListener('click', triggerFileInput);
    }
    
    // Add to timeline button
    if (elements.addToTimelineBtn) {
        elements.addToTimelineBtn.addEventListener('click', handleAddToTimeline);
    }
    
    // Frame edit modal events
    if (elements.replaceImageBtn) {
        elements.replaceImageBtn.addEventListener('click', triggerFileInput);
    }
    
    if (elements.deleteFrameBtn) {
        elements.deleteFrameBtn.addEventListener('click', deleteCurrentFrame);
    }
    
    if (elements.saveFrameBtn) {
        elements.saveFrameBtn.addEventListener('click', saveFrameChanges);
    }
    
    // Job queue tab
    if (jobQueueElements.refreshJobsBtn) {
        jobQueueElements.refreshJobsBtn.addEventListener('click', loadJobQueue);
    }
    
    // Output tab
    if (elements.refreshOutputsBtn) {
        elements.refreshOutputsBtn.addEventListener('click', loadOutputs);
    }
    
    // Tab change events
    const queueTab = document.getElementById('queue-tab');
    if (queueTab) {
        queueTab.addEventListener('shown.bs.tab', () => {
            loadJobQueue();
        });
    }
    
    const outputTab = document.getElementById('output-tab');
    if (outputTab) {
        outputTab.addEventListener('shown.bs.tab', () => {
            loadOutputs();
        });
    }
}

// Load the job queue UI
async function loadJobQueue() {
    try {
        jobQueueElements.jobsContainer.innerHTML = `
            <div class="alert alert-info">
                <i class="bi bi-info-circle"></i> Loading jobs...
            </div>
        `;
        
        const response = await fetch('/api/list_jobs');
        
        if (!response.ok) {
            throw new Error('Failed to fetch job queue');
        }
        
        const jobs = await response.json();
        
        if (jobs.length === 0) {
            jobQueueElements.jobsContainer.innerHTML = `
                <div class="alert alert-secondary">
                    <i class="bi bi-info-circle"></i> No jobs in queue
                </div>
            `;
            return;
        }
        
        // Sort jobs by timestamp (newest first)
        jobs.sort((a, b) => {
            const aTime = a.job_id.split('_')[0] + '_' + a.job_id.split('_')[1];
            const bTime = b.job_id.split('_')[0] + '_' + b.job_id.split('_')[1];
            return bTime.localeCompare(aTime);
        });
        
        // Create a list of jobs
        let jobsHtml = '';
        
        jobs.forEach(job => {
            const jobTime = formatJobTimestamp(job.job_id);
            let statusClass = '';
            let statusIcon = '';
            
            switch (job.status) {
                case 'pending':
                    statusClass = 'pending';
                    statusIcon = '<i class="bi bi-hourglass-split me-2"></i>';
                    break;
                case 'completed':
                    statusClass = 'completed';
                    statusIcon = '<i class="bi bi-check-circle me-2"></i>';
                    break;
                case 'failed':
                    statusClass = 'failed';
                    statusIcon = '<i class="bi bi-exclamation-triangle me-2"></i>';
                    break;
                default:
                    statusIcon = '<i class="bi bi-question-circle me-2"></i>';
            }
            
            jobsHtml += `
                <div class="job-item ${statusClass}" data-job-id="${job.job_id}">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            ${statusIcon}
                            <strong>Job ${jobTime}</strong>
                        </div>
                        <div class="job-actions">
                            ${job.status === 'pending' ? 
                                `<button class="btn btn-sm btn-outline-danger cancel-job-btn" data-job-id="${job.job_id}">
                                    <i class="bi bi-x-circle"></i> Cancel
                                </button>` : 
                                `<span class="badge ${job.status === 'completed' ? 'bg-success' : 'bg-danger'}">${job.status}</span>`
                            }
                        </div>
                    </div>
                </div>
            `;
        });
        
        jobQueueElements.jobsContainer.innerHTML = jobsHtml;
        
        // Add click event listeners to job items
        document.querySelectorAll('.job-item').forEach(jobItem => {
            jobItem.addEventListener('click', (e) => {
                if (e.target.closest('.cancel-job-btn')) {
                    // If the click was on the cancel button, don't select the job
                    return;
                }
                
                // Remove active class from all job items
                document.querySelectorAll('.job-item').forEach(item => {
                    item.classList.remove('active');
                });
                
                // Add active class to clicked job item
                jobItem.classList.add('active');
                
                // Load job details
                const jobId = jobItem.dataset.jobId;
                loadJobDetails(jobId);
            });
        });
        
        // Add click event listeners to cancel buttons
        document.querySelectorAll('.cancel-job-btn').forEach(button => {
            button.addEventListener('click', async (e) => {
                e.stopPropagation(); // Prevent job selection
                const jobId = button.dataset.jobId;
                
                if (confirm(`Are you sure you want to cancel job ${formatJobTimestamp(jobId)}?`)) {
                    await cancelJob(jobId);
                }
            });
        });
        
    } catch (error) {
        console.error('Error loading job queue:', error);
        jobQueueElements.jobsContainer.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i> Failed to load job queue: ${error.message}
            </div>
        `;
    }
}

// Load job details
async function loadJobDetails(jobId) {
    try {
        jobQueueElements.jobDetailContainer.innerHTML = `
            <div class="alert alert-info">
                <i class="bi bi-info-circle"></i> Loading job details...
            </div>
        `;
        
        const response = await fetch(`/api/job_status/${jobId}`);
        
        if (!response.ok) {
            throw new Error('Failed to fetch job details');
        }
        
        const job = await response.json();
        
        // Create job details UI
        let statusBadgeClass = '';
        switch (job.status) {
            case 'pending':
                statusBadgeClass = 'bg-primary';
                break;
            case 'completed':
                statusBadgeClass = 'bg-success';
                break;
            case 'failed':
                statusBadgeClass = 'bg-danger';
                break;
            default:
                statusBadgeClass = 'bg-secondary';
        }
        
        let detailsHtml = `
            <div class="card mb-3">
                <div class="card-body">
                    <h3 class="fs-5 mb-3">Job ${formatJobTimestamp(jobId)}</h3>
                    <span class="badge ${statusBadgeClass} mb-3">${job.status}</span>
                    
                    <div class="row mb-3">
                        <div class="col-md-4 fw-bold">Job ID:</div>
                        <div class="col-md-8">${jobId}</div>
                    </div>
        `;
        
        if (job.status === 'pending') {
            // For pending jobs, show progress information
            const progressPercentage = job.progress || 0;
            
            detailsHtml += `
                <div class="row mb-3">
                    <div class="col-md-4 fw-bold">Progress:</div>
                    <div class="col-md-8">
                        <div class="progress">
                            <div class="progress-bar" role="progressbar" style="width: ${progressPercentage}%" 
                                aria-valuenow="${progressPercentage}" aria-valuemin="0" aria-valuemax="100">
                                ${progressPercentage}%
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="row mb-3">
                    <div class="col-md-4 fw-bold">Status:</div>
                    <div class="col-md-8">${job.message || 'Processing...'}</div>
                </div>
            `;
            
            if (job.preview_image) {
                detailsHtml += `
                    <div class="row mb-3">
                        <div class="col-md-4 fw-bold">Preview:</div>
                        <div class="col-md-8">
                            <img src="${job.preview_image}" class="img-fluid rounded" alt="Preview">
                        </div>
                    </div>
                `;
            }
            
        } else if (job.status === 'completed') {
            // For completed jobs, show the result video
            detailsHtml += `
                <div class="row mb-3">
                    <div class="col-12">
                        <video src="${job.result_video}" controls class="img-fluid rounded mb-3"></video>
                        <a href="${job.result_video}" download class="btn btn-primary btn-sm">
                            <i class="bi bi-download"></i> Download Video
                        </a>
                    </div>
                </div>
            `;
        } else if (job.status === 'failed') {
            // For failed jobs, show the error message
            detailsHtml += `
                <div class="row mb-3">
                    <div class="col-md-4 fw-bold">Error:</div>
                    <div class="col-md-8 text-danger">${job.message || 'Unknown error'}</div>
                </div>
            `;
        }
        
        detailsHtml += `
                </div>
            </div>
        `;
        
        jobQueueElements.jobDetailContainer.innerHTML = detailsHtml;
        
    } catch (error) {
        console.error('Error loading job details:', error);
        jobQueueElements.jobDetailContainer.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i> Failed to load job details: ${error.message}
            </div>
        `;
    }
}

// Cancel a job
async function cancelJob(jobId) {
    try {
        const response = await fetch(`/api/cancel_job/${jobId}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('Failed to cancel job');
        }
        
        const result = await response.json();
        
        if (result.success) {
            // Refresh the job queue to show updated status
            loadJobQueue();
            
            // Show success message
            showMessage('Job cancelled successfully', 'success');
        } else {
            throw new Error(result.error || 'Failed to cancel job');
        }
        
    } catch (error) {
        console.error('Error cancelling job:', error);
        showMessage(`Failed to cancel job: ${error.message}`, 'error');
    }
}

// Load outputs for the outputs tab
async function loadOutputs() {
    try {
        elements.outputsContainer.innerHTML = `
            <div class="col">
                <div class="alert alert-info">
                    <i class="bi bi-info-circle"></i> Loading output videos...
                </div>
            </div>
        `;
        
        console.log('Fetching outputs from server...');
        const response = await fetch('/api/list_outputs', {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
                'Cache-Control': 'no-cache'
            }
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Error response:', errorText);
            throw new Error(`Failed to fetch outputs (${response.status}): ${errorText}`);
        }
        
        const outputs = await response.json();
        console.log('Received outputs:', outputs);
        
        if (!outputs || outputs.length === 0) {
            elements.outputsContainer.innerHTML = `
                <div class="col">
                    <div class="alert alert-secondary">
                        <i class="bi bi-info-circle"></i> No output videos available
                    </div>
                </div>
            `;
            return;
        }
        
        // Sort outputs by timestamp (newest first)
        outputs.sort((a, b) => {
            return b.timestamp - a.timestamp;
        });
        
        // Create output cards
        let outputsHtml = '';
        
        outputs.forEach(output => {
            const timestamp = formatTimestamp(output.timestamp);
            
            outputsHtml += `
                <div class="col">
                    <div class="card h-100">
                        <div class="card-body">
                            <h5 class="card-title fs-6">${output.name}</h5>
                            <p class="card-text small text-muted">${timestamp}</p>
                            <div class="ratio ratio-16x9 mb-3">
                                <video src="${output.path}" class="img-fluid rounded" poster="${output.thumbnail || ''}" preload="none"></video>
                            </div>
                            <div class="d-grid gap-2">
                                <button class="btn btn-primary btn-sm view-video-btn" data-video="${output.path}" data-name="${output.name}">
                                    <i class="bi bi-play-fill"></i> Play Video
                                </button>
                                <a href="${output.path}" download="${output.name}" class="btn btn-outline-secondary btn-sm">
                                    <i class="bi bi-download"></i> Download
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        elements.outputsContainer.innerHTML = outputsHtml;
        
        // Add click event listeners to view buttons
        document.querySelectorAll('.view-video-btn').forEach(button => {
            button.addEventListener('click', () => {
                const videoPath = button.dataset.video;
                const videoName = button.dataset.name;
                openVideoViewer(videoPath, videoName);
            });
        });
        
    } catch (error) {
        console.error('Error loading outputs:', error);
        elements.outputsContainer.innerHTML = `
            <div class="col">
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle"></i> Failed to load outputs: ${error.message}
                </div>
            </div>
        `;
    }
}

// Open video viewer modal
function openVideoViewer(videoPath, videoName) {
    elements.modalVideo.src = videoPath;
    document.getElementById('videoViewerModalLabel').textContent = videoName;
    elements.modalDownloadBtn.href = videoPath;
    elements.modalDownloadBtn.download = videoName;
    
    // Show the modal
    const videoViewerModal = new bootstrap.Modal(document.getElementById('videoViewerModal'));
    videoViewerModal.show();
}

// Format job timestamp for display
function formatJobTimestamp(jobId) {
    const parts = jobId.split('_');
    if (parts.length >= 2) {
        const dateStr = parts[0];
        const timeStr = parts[1];
        
        if (dateStr.length >= 6 && timeStr.length >= 6) {
            const day = dateStr.substring(0, 2);
            const month = dateStr.substring(2, 4);
            const year = dateStr.substring(4, 6);
            
            const hour = timeStr.substring(0, 2);
            const minute = timeStr.substring(2, 4);
            const second = timeStr.substring(4, 6);
            
            return `${day}/${month}/${year} ${hour}:${minute}:${second}`;
        }
    }
    return jobId;
}

// Format timestamp for display
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

// Add CSS for job queue items
document.addEventListener('DOMContentLoaded', function() {
    const style = document.createElement('style');
    style.textContent = `
        .job-list {
            max-height: 70vh;
            overflow-y: auto;
        }
        .job-item {
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 4px;
            border: 1px solid #dee2e6;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .job-item:hover {
            background-color: #f8f9fa;
        }
        .job-item.active {
            background-color: #e9ecef;
            border-color: #adb5bd;
        }
        .job-item.pending {
            border-left: 4px solid #0d6efd;
        }
        .job-item.completed {
            border-left: 4px solid #198754;
        }
        .job-item.failed {
            border-left: 4px solid #dc3545;
        }
    `;
    document.head.appendChild(style);
});

// Function to handle file selection from input
function handleFileSelect(e) {
    const files = e.target.files;
    processSelectedFiles(files);
}

// Function to handle drag over event
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.uploadDropArea.classList.add('active');
}

// Function to handle drag leave event
function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.uploadDropArea.classList.remove('active');
}

// Function to handle file drop event
function handleFileDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    elements.uploadDropArea.classList.remove('active');
    
    const files = e.dataTransfer.files;
    processSelectedFiles(files);
}

// Function to trigger file input click
function triggerFileInput() {
    elements.fileInput.click();
}

// Function to process selected files
function processSelectedFiles(files) {
    if (!files.length) return;
    
    // Clear previous uploads if keeping current image is not enabled
    if (!keepCurrentImage) {
        elements.imageUploadContainer.innerHTML = '';
        selectedFiles = [];
    }
    
    // Process each file
    Array.from(files).forEach(file => {
        // Only process image files
        if (!file.type.match('image.*')) return;
        
        const reader = new FileReader();
        
        reader.onload = (e) => {
            const imgSrc = e.target.result;
            const fileName = file.name;
            
            // Add to selected files array
            selectedFiles.push({
                file: file,
                src: imgSrc,
                name: fileName
            });
            
            // Create thumbnail
            const thumbnailHtml = `
                <div class="col-4 col-md-3 mb-3">
                    <div class="card">
                        <img src="${imgSrc}" class="card-img-top" alt="${fileName}">
                        <div class="card-body p-2">
                            <p class="card-text small text-muted text-truncate">${fileName}</p>
                        </div>
                    </div>
                </div>
            `;
            
            elements.imageUploadContainer.insertAdjacentHTML('beforeend', thumbnailHtml);
        };
        
        reader.readAsDataURL(file);
    });
}

// Function to add images to timeline
function handleAddToTimeline() {
    if (selectedFiles.length === 0) {
        alert('Please select at least one image to add to the timeline.');
        return;
    }
    
    // For each selected file, add to timeline
    selectedFiles.forEach(fileObj => {
        addItemToTimeline(fileObj);
    });
    
    // Close modal and clean up
    uploadModal.hide();
    selectedFiles = [];
    updateTimelineStatus();
}

// Function to add an item to the timeline
function addItemToTimeline(fileObj) {
    const timelineItem = document.createElement('div');
    timelineItem.className = 'card mb-3 timeline-item';
    timelineItem.draggable = true;
    
    const frameDuration = elements.frameTime ? parseFloat(elements.frameTime.value) : 0.5;
    
    timelineItem.innerHTML = `
        <div class="card-body">
            <div class="row">
                <div class="col-md-4 mb-3 mb-md-0">
                    <div class="drag-handle"><i class="bi bi-grip-vertical"></i></div>
                    <img src="${fileObj.src}" class="img-fluid rounded" alt="${fileObj.name}">
                    <span class="duration-badge">${frameDuration}s</span>
                </div>
                <div class="col-md-8">
                    <textarea class="form-control prompt-text mb-2" placeholder="Frame prompt (optional)"></textarea>
                    <div class="d-flex justify-content-between">
                        <button class="btn btn-sm btn-outline-primary edit-frame-btn">
                            <i class="bi bi-pencil"></i> Edit
                        </button>
                        <button class="btn btn-sm btn-outline-danger delete-frame-btn">
                            <i class="bi bi-trash"></i> Remove
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Attach event listeners
    const editBtn = timelineItem.querySelector('.edit-frame-btn');
    const deleteBtn = timelineItem.querySelector('.delete-frame-btn');
    
    if (editBtn) {
        editBtn.addEventListener('click', () => {
            showFrameEditModal(timelineItem);
        });
    }
    
    if (deleteBtn) {
        deleteBtn.addEventListener('click', () => {
            timelineItem.remove();
            updateTimelineStatus();
            updateTimelineArray();
        });
    }
    
    // Add drag and drop event listeners
    timelineItem.addEventListener('dragstart', handleDragStart);
    timelineItem.addEventListener('dragover', handleTimelineDragOver);
    timelineItem.addEventListener('dragleave', handleTimelineDragLeave);
    timelineItem.addEventListener('drop', handleTimelineDrop);
    timelineItem.addEventListener('dragend', handleDragEnd);
    
    timeline.push({
        src: fileObj.src,
        file: fileObj.file,
        duration: frameDuration,
        prompt: ''
    });
    
    elements.timelineContainer.appendChild(timelineItem);
}

// Timeline drag and drop event handlers
let dragSrcEl = null;

function handleDragStart(e) {
    this.classList.add('dragging');
    dragSrcEl = this;
    
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/html', this.innerHTML);
}

function handleTimelineDragOver(e) {
    if (e.preventDefault) {
        e.preventDefault();
    }
    
    e.dataTransfer.dropEffect = 'move';
    this.classList.add('drag-over');
    
    return false;
}

function handleTimelineDragLeave(e) {
    this.classList.remove('drag-over');
}

function handleTimelineDrop(e) {
    if (e.stopPropagation) {
        e.stopPropagation();
    }
    
    this.classList.remove('drag-over');
    
    // Only proceed if we're dropping onto a different item
    if (dragSrcEl !== this) {
        // Get position of source and target elements
        const items = Array.from(elements.timelineContainer.querySelectorAll('.timeline-item'));
        const srcIndex = items.indexOf(dragSrcEl);
        const targetIndex = items.indexOf(this);
        
        // Swap elements in the DOM
        if (targetIndex < srcIndex) {
            // Moving up
            elements.timelineContainer.insertBefore(dragSrcEl, this);
        } else {
            // Moving down
            if (this.nextSibling) {
                elements.timelineContainer.insertBefore(dragSrcEl, this.nextSibling);
            } else {
                elements.timelineContainer.appendChild(dragSrcEl);
            }
        }
        
        // Update the timeline array to match the new order
        updateTimelineArray();
    }
    
    return false;
}

function handleDragEnd(e) {
    // Remove all drag classes
    document.querySelectorAll('.timeline-item').forEach(item => {
        item.classList.remove('dragging', 'drag-over');
    });
}

// Function to update the timeline array based on the DOM order
function updateTimelineArray() {
    const newTimeline = [];
    const items = Array.from(elements.timelineContainer.querySelectorAll('.timeline-item'));
    
    items.forEach((item, index) => {
        const img = item.querySelector('img');
        const promptText = item.querySelector('.prompt-text');
        const durationBadge = item.querySelector('.duration-badge');
        
        // Find matching item in original timeline
        const originalItem = timeline.find(t => {
            return t.src === img.src;
        });
        
        if (originalItem) {
            // Update with current values
            newTimeline.push({
                ...originalItem,
                prompt: promptText ? promptText.value : ''
            });
        }
    });
    
    // Replace timeline with new ordered array
    timeline = newTimeline;
}

// Add timeline drop zone for drag & drop from desktop
function initTimelineDropZone() {
    if (!elements.timelineContainer) return;
    
    // Initialize empty timeline UI
    updateTimelineStatus();
    
    elements.timelineContainer.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        // If we're over the empty state dropzone or the container itself
        const dropzone = elements.timelineContainer.querySelector('.timeline-dropzone');
        if (dropzone) {
            dropzone.classList.add('active');
        } else {
            elements.timelineContainer.classList.add('active-dropzone');
        }
    });
    
    elements.timelineContainer.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        // If we're over the empty state dropzone or the container itself
        const dropzone = elements.timelineContainer.querySelector('.timeline-dropzone');
        if (dropzone) {
            dropzone.classList.remove('active');
        } else {
            elements.timelineContainer.classList.remove('active-dropzone');
        }
    });
    
    elements.timelineContainer.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        
        // Remove active classes
        const dropzone = elements.timelineContainer.querySelector('.timeline-dropzone');
        if (dropzone) {
            dropzone.classList.remove('active');
        } else {
            elements.timelineContainer.classList.remove('active-dropzone');
        }
        
        // Handle files dropped directly from the desktop
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            processSelectedFiles(e.dataTransfer.files);
            
            // If files were processed, add them to timeline
            if (selectedFiles.length > 0) {
                // Clear the empty state UI first if needed
                if (timeline.length === 0) {
                    elements.timelineContainer.innerHTML = '';
                }
                
                selectedFiles.forEach(fileObj => {
                    addItemToTimeline(fileObj);
                });
                selectedFiles = [];
                updateTimelineStatus();
            }
        }
    });
}

// Function to show frame edit modal
function showFrameEditModal(timelineItem) {
    const index = Array.from(elements.timelineContainer.children).indexOf(timelineItem);
    if (index === -1) return;
    
    currentEditIndex = index;
    const currentFrame = timeline[index];
    
    // Fill modal with frame data
    elements.frameEditImage.src = currentFrame.src;
    elements.frameDuration.value = currentFrame.duration;
    
    // Show modal
    editItemModal.show();
}

// Function to delete current frame
function deleteCurrentFrame() {
    if (currentEditIndex === -1) return;
    
    // Remove from timeline array
    timeline.splice(currentEditIndex, 1);
    
    // Remove from DOM
    const timelineItems = elements.timelineContainer.querySelectorAll('.timeline-item');
    if (timelineItems[currentEditIndex]) {
        timelineItems[currentEditIndex].remove();
    }
    
    // Close modal
    editItemModal.hide();
    currentEditIndex = -1;
    
    updateTimelineStatus();
}

// Function to save frame changes
function saveFrameChanges() {
    if (currentEditIndex === -1) return;
    
    const duration = parseFloat(elements.frameDuration.value);
    
    // Update timeline array
    timeline[currentEditIndex].duration = duration;
    
    // Update DOM
    const timelineItems = elements.timelineContainer.querySelectorAll('.timeline-item');
    if (timelineItems[currentEditIndex]) {
        const durationBadge = timelineItems[currentEditIndex].querySelector('.duration-badge');
        if (durationBadge) {
            durationBadge.textContent = `${duration}s`;
        }
    }
    
    // Close modal
    editItemModal.hide();
    currentEditIndex = -1;
}

// Function to update timeline status
function updateTimelineStatus() {
    // Enable/disable generate button based on timeline
    if (elements.generateVideoBtn) {
        elements.generateVideoBtn.disabled = timeline.length < 1;
    }
    
    // Show message when timeline is empty
    if (timeline.length === 0) {
        elements.timelineContainer.innerHTML = `
            <div class="timeline-dropzone">
                <div class="text-center py-5">
                    <i class="bi bi-images fs-1 mb-3"></i>
                    <h5>Drop Images Here</h5>
                    <p class="text-muted">Drag and drop images to add them to your timeline</p>
                    <p>- or -</p>
                    <button class="btn btn-primary" id="dropzoneUploadBtn">
                        <i class="bi bi-upload me-1"></i> Select Files
                    </button>
                </div>
            </div>
        `;
        
        // Add click event for the upload button
        const dropzoneUploadBtn = document.getElementById('dropzoneUploadBtn');
        if (dropzoneUploadBtn) {
            dropzoneUploadBtn.addEventListener('click', showUploadModal);
        }
    }
}

// Function to start video generation
function startGeneration() {
    if (timeline.length === 0) {
        alert('Please add at least one image to the timeline before generating.');
        return;
    }
    
    // Collect settings
    const settings = {
        frameRate: elements.frameRate ? parseInt(elements.frameRate.value) : 30,
        outputWidth: elements.outputWidth ? parseInt(elements.outputWidth.value) : 1920,
        outputHeight: elements.outputHeight ? parseInt(elements.outputHeight.value) : 1080,
        outputFormat: elements.outputFormat ? elements.outputFormat.value : 'mp4'
    };
    
    // Collect segments data
    const segments = timeline.map(item => {
        const promptInput = Array.from(elements.timelineContainer.children)
            .find((_, i) => i === timeline.indexOf(item))
            ?.querySelector('.prompt-text');
        
        return {
            image_path: item.file.name,
            prompt: promptInput ? promptInput.value : '',
            duration: item.duration
        };
    });
    
    // Show progress UI
    elements.progressContainer.classList.remove('d-none');
    elements.progressBar.style.width = '0%';
    elements.progressBar.setAttribute('aria-valuenow', 0);
    elements.progressBar.textContent = '0%';
    elements.progressStatus.textContent = 'Preparing...';
    
    // TODO: Implement actual API call for generation
    // For now, simulate a generation process
    simulateGeneration();
}

// Simulation function for development (will be replaced with actual API call)
function simulateGeneration() {
    let progress = 0;
    const interval = setInterval(() => {
        progress += 5;
        elements.progressBar.style.width = `${progress}%`;
        elements.progressBar.setAttribute('aria-valuenow', progress);
        elements.progressBar.textContent = `${progress}%`;
        
        if (progress === 25) {
            elements.progressStatus.textContent = 'Processing images...';
        } else if (progress === 50) {
            elements.progressStatus.textContent = 'Generating frames...';
            elements.previewContainer.classList.remove('d-none');
            elements.previewImage.src = timeline[0].src;
        } else if (progress === 75) {
            elements.progressStatus.textContent = 'Creating video...';
        }
        
        if (progress >= 100) {
            clearInterval(interval);
            elements.progressStatus.textContent = 'Generation completed!';
            
            // After a delay, switch to the output tab to show the result
            setTimeout(() => {
                const outputTab = document.getElementById('output-tab');
                if (outputTab) {
                    outputTab.click();
                }
            }, 2000);
        }
    }, 200);
}

// Check for running job
function checkForRunningJob() {
    // TODO: Implement actual API call to check for running jobs
    console.log('Checking for running jobs...');
}

// Function to sort timeline items
function sortTimeline(direction) {
    // Get all timeline items
    const timelineItems = Array.from(elements.timelineContainer.querySelectorAll('.timeline-item'));
    
    if (timelineItems.length <= 1) {
        return; // Nothing to sort
    }
    
    // Get sorted items based on file names
    const sortedItems = timelineItems.sort((a, b) => {
        const fileNameA = a.querySelector('img').alt || '';
        const fileNameB = b.querySelector('img').alt || '';
        
        if (direction === 'asc') {
            return fileNameA.localeCompare(fileNameB);
        } else {
            return fileNameB.localeCompare(fileNameA);
        }
    });
    
    // Reorder DOM elements
    sortedItems.forEach(item => {
        elements.timelineContainer.appendChild(item);
    });
    
    // Update timeline array to match the new order
    updateTimelineArray();
} 