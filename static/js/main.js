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
    
    // Initialize theme
    initTheme();
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
    
    // Theme toggle
    const darkModeToggle = document.getElementById('darkModeToggle');
    if (darkModeToggle) {
        darkModeToggle.addEventListener('change', toggleDarkMode);
    }
    
    // File input change event
    if (elements.fileInput) {
        elements.fileInput.addEventListener('change', handleFileSelect);
        // Add click event listener to prevent propagation
        elements.fileInput.addEventListener('click', (e) => {
            e.stopPropagation();
        });
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
    const outputTab = document.getElementById('output-tab');
    const editorTab = document.getElementById('editor-tab');
    
    if (queueTab) {
        queueTab.addEventListener('shown.bs.tab', () => {
            loadJobQueue();
        });
    }
    
    if (outputTab) {
        outputTab.addEventListener('shown.bs.tab', () => {
            loadOutputs();
        });
    }
    
    if (editorTab) {
        editorTab.addEventListener('shown.bs.tab', () => {
            enforceHorizontalLayout();
        });
    }
    
    // Ensure horizontal layout is applied when tabs change
    const tabs = document.querySelectorAll('button[data-bs-toggle="tab"]');
    tabs.forEach(tab => {
        tab.addEventListener('shown.bs.tab', () => {
            // Small delay to ensure DOM is updated
            setTimeout(enforceHorizontalLayout, 50);
        });
    });
    
    // Add Clear Timeline button
    // First check if it already exists to avoid duplicates
    if (!document.getElementById('clearTimelineBtn')) {
        const generateBtn = document.getElementById('generateVideoBtn');
        if (generateBtn) {
            const clearBtn = document.createElement('button');
            clearBtn.id = 'clearTimelineBtn';
            clearBtn.className = 'btn btn-outline-danger me-2';
            clearBtn.innerHTML = '<i class="bi bi-trash"></i> Clear Timeline';
            clearBtn.addEventListener('click', clearTimeline);
            
            // Insert before the generate button
            generateBtn.parentNode.insertBefore(clearBtn, generateBtn);
        }
    }
    
    // Apply horizontal layout on window resize
    window.addEventListener('resize', enforceHorizontalLayout);
}

// Load the job queue UI
async function loadJobQueue() {
    try {
        const response = await fetch('/api/list_jobs');
        if (!response.ok) {
            throw new Error(`Failed to fetch jobs: ${response.statusText}`);
        }
        
        const jobs = await response.json();
        const jobsContainer = document.getElementById('jobsContainer');
        
        // Clear the container
        jobsContainer.innerHTML = '';
        
        if (jobs.length === 0) {
            jobsContainer.innerHTML = `
                <div class="alert alert-secondary">
                    <i class="bi bi-info-circle"></i> No jobs in queue
                </div>
            `;
            return;
        }
        
        // Add management controls at the top
        const managementControls = document.createElement('div');
        managementControls.className = 'mb-3 d-flex justify-content-end';
        
        // Count completed jobs
        const completedJobs = jobs.filter(job => job.status === 'completed');
        if (completedJobs.length > 0) {
            const clearCompletedBtn = document.createElement('button');
            clearCompletedBtn.className = 'btn btn-sm btn-outline-danger';
            clearCompletedBtn.innerHTML = '<i class="bi bi-trash"></i> Clear Completed Jobs';
            clearCompletedBtn.onclick = () => {
                if (confirm(`Are you sure you want to delete all ${completedJobs.length} completed jobs?`)) {
                    clearCompletedJobs();
                }
            };
            managementControls.appendChild(clearCompletedBtn);
        }
        
        jobsContainer.appendChild(managementControls);
        
        // Sort jobs: running first, then queued, then completed/failed by timestamp (newest first)
        jobs.sort((a, b) => {
            // First, prioritize by status
            const statusOrder = { 'running': 0, 'queued': 1, 'completed': 2, 'failed': 3 };
            const statusDiff = statusOrder[a.status] - statusOrder[b.status];
            
            if (statusDiff !== 0) return statusDiff;
            
            // If same status, sort by timestamp (extracted from job_id)
            const aTime = parseInt(a.job_id.split('-')[0]);
            const bTime = parseInt(b.job_id.split('-')[0]);
            return bTime - aTime; // Newest first
        });
        
        // Add each job to the container
        jobs.forEach(job => {
            const jobItem = document.createElement('div');
            jobItem.className = `job-item ${job.status}`;
            jobItem.dataset.jobId = job.job_id;
            
            // Determine the status badge class
            let statusBadgeClass = 'bg-secondary';
            if (job.status === 'completed') {
                statusBadgeClass = 'bg-success';
            } else if (job.status === 'failed') {
                statusBadgeClass = 'bg-danger';
            } else if (job.status === 'running') {
                statusBadgeClass = 'bg-primary';
            } else if (job.status === 'queued') {
                statusBadgeClass = 'bg-warning';
            }
            
            // Job title/name display
            const jobTitle = job.job_name || formatJobTimestamp(job.job_id);
            
            // Check if job has missing images
            const hasInvalidImages = job.is_valid === false;
            const invalidBadge = hasInvalidImages ? 
                `<span class="badge bg-danger ms-2">Missing Images</span>` : '';
            
            jobItem.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <div class="fw-bold">${jobTitle}</div>
                        <div class="text-muted small">ID: ${job.job_id}</div>
                    </div>
                    <div>
                        <span class="badge ${statusBadgeClass}">${job.status}</span>
                        ${invalidBadge}
                    </div>
                </div>
                <div class="mt-2">
                    <div class="text-muted small">${job.message || 'No message'}</div>
                </div>
                ${job.status === 'running' ? `
                <div class="progress mt-2" style="height: 5px;">
                    <div class="progress-bar" role="progressbar" style="width: ${job.progress}%" aria-valuenow="${job.progress}" aria-valuemin="0" aria-valuemax="100"></div>
                </div>` : ''}
            `;
            
            jobItem.addEventListener('click', () => {
                // Remove active class from all job items
                document.querySelectorAll('.job-item').forEach(item => {
                    item.classList.remove('active');
                });
                
                // Add active class to the clicked job item
                jobItem.classList.add('active');
                
                // Load the job details
                loadJobDetails(job.job_id);
            });
            
            jobsContainer.appendChild(jobItem);
        });
        
    } catch (error) {
        console.error('Error loading job queue:', error);
        document.getElementById('jobsContainer').innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i> Error loading jobs: ${error.message}
            </div>
        `;
    }
}

// Function to clear all completed jobs
async function clearCompletedJobs() {
    // Get the list of jobs
    try {
        const response = await fetch('/api/list_jobs');
        if (!response.ok) {
            throw new Error(`Failed to fetch jobs: ${response.statusText}`);
        }
        
        const jobs = await response.json();
        
        // Filter for completed jobs
        const completedJobs = jobs.filter(job => job.status === 'completed');
        
        // Show loading message
        const jobsContainer = document.getElementById('jobsContainer');
        jobsContainer.innerHTML = `
            <div class="alert alert-info">
                <i class="bi bi-hourglass"></i> Cleaning up ${completedJobs.length} completed jobs...
            </div>
        `;
        
        // Delete each completed job
        const deletePromises = completedJobs.map(job => 
            fetch(`/api/job/${job.job_id}`, { method: 'DELETE' })
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`Failed to delete job ${job.job_id}`);
                    }
                    return response.json();
                })
        );
        
        // Wait for all deletions to complete
        await Promise.all(deletePromises);
        
        // Refresh the job queue
        loadJobQueue();
        
        // Clear job details if showing a completed job
        document.getElementById('jobDetailContainer').innerHTML = '';
        document.getElementById('jobMediaContainer').classList.add('d-none');
        
        // Show success message
        showMessage(`Successfully deleted ${completedJobs.length} completed jobs`, 'success');
        
    } catch (error) {
        console.error('Error clearing completed jobs:', error);
        showMessage(`Failed to clear completed jobs: ${error.message}`, 'error');
        
        // Refresh anyway to show current state
        loadJobQueue();
    }
}

// Load job details
async function loadJobDetails(jobId) {
    try {
        const response = await fetch(`/api/job_status/${jobId}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch job details: ${response.statusText}`);
        }
        
        const jobData = await response.json();
        
        // Clear previous content
        const jobDetailContainer = document.getElementById('jobDetailContainer');
        jobDetailContainer.innerHTML = '';
        
        const jobMediaContainer = document.getElementById('jobMediaContainer');
        
        // Basic job information card
        const jobInfoCard = document.createElement('div');
        jobInfoCard.className = 'card mb-3';
        
        // Determine the status badge class
        let statusBadgeClass = 'bg-secondary';
        if (jobData.status === 'completed') {
            statusBadgeClass = 'bg-success';
        } else if (jobData.status === 'failed') {
            statusBadgeClass = 'bg-danger';
        } else if (jobData.status === 'running') {
            statusBadgeClass = 'bg-primary';
        } else if (jobData.status === 'queued') {
            statusBadgeClass = 'bg-warning';
        }
        
        // Check if job has missing images
        const hasInvalidImages = jobData.is_valid === false;
        
        // Display job name if available
        const jobName = jobData.job_name ? 
            `<p><strong>Name:</strong> ${jobData.job_name}</p>` : '';
        
        // Invalid images warning
        const invalidImagesWarning = hasInvalidImages ? 
            `<div class="alert alert-warning">
                <i class="bi bi-exclamation-triangle me-2"></i>
                This job is missing ${jobData.missing_images.length} image(s). You can reload 
                the job to timeline but must fix the missing images before rerunning.
            </div>` : '';
        
        jobInfoCard.innerHTML = `
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="card-title mb-0">Job Information</h5>
                <span class="badge ${statusBadgeClass}">${jobData.status}</span>
            </div>
            <div class="card-body">
                ${invalidImagesWarning}
                <p><strong>Job ID:</strong> ${jobId}</p>
                ${jobName}
                <p><strong>Status:</strong> ${jobData.status.charAt(0).toUpperCase() + jobData.status.slice(1)}</p>
                <p><strong>Progress:</strong> ${jobData.progress}%</p>
                <p><strong>Message:</strong> ${jobData.message || 'No message'}</p>
                <p><strong>Created:</strong> ${formatJobTimestamp(jobId)}</p>
                ${jobData.result_video ? `<p><strong>Result:</strong> <a href="${jobData.result_video}" target="_blank">${jobData.result_video.split('/').pop()}</a></p>` : ''}
                
                ${jobData.status === 'running' || jobData.status === 'completed' ? 
                    `<div class="progress mb-3">
                        <div class="progress-bar" role="progressbar" style="width: ${jobData.progress}%" aria-valuenow="${jobData.progress}" aria-valuemin="0" aria-valuemax="100">${jobData.progress}%</div>
                    </div>` : ''}
            </div>
        `;
        
        // Add the job info card
        jobDetailContainer.appendChild(jobInfoCard);
        
        // Show video and latents if job is running or completed
        if (jobData.status === 'running' || jobData.status === 'completed') {
            jobMediaContainer.classList.remove('d-none');
            
            // Handle video preview
            const videoContainer = document.createElement('div');
            videoContainer.className = 'col-md-6';
            videoContainer.innerHTML = `
                <div class="card mb-3">
                    <div class="card-header">
                        <h5 class="card-title mb-0 fs-6">Current Output</h5>
                    </div>
                    <div class="card-body text-center">
                        ${jobData.result_video ? 
                            `<video id="jobCurrentVideo" src="${jobData.result_video}" controls class="img-fluid rounded"></video>` :
                            `<div class="text-muted py-5"><i class="bi bi-film me-2"></i>Video not available yet</div>`
                        }
                    </div>
                </div>
            `;
            
            // Handle latents preview
            const latentsContainer = document.createElement('div');
            latentsContainer.className = 'col-md-6';
            
            // Determine which image to show, with fallbacks
            let latentImageSrc = '';
            if (jobData.current_latents) {
                latentImageSrc = jobData.current_latents;
            } else if (jobData.segments && jobData.segments.length > 0) {
                latentImageSrc = jobData.segments[0];
            }
            
            latentsContainer.innerHTML = `
                <div class="card mb-3">
                    <div class="card-header">
                        <h5 class="card-title mb-0 fs-6">Current Latents</h5>
                    </div>
                    <div class="card-body text-center">
                        ${latentImageSrc ? 
                            `<img id="jobCurrentLatents" src="${latentImageSrc}" class="img-fluid rounded" alt="Current latents">` :
                            `<div class="text-muted py-5"><i class="bi bi-image me-2"></i>Latent image not available yet</div>`
                        }
                    </div>
                </div>
            `;
            
            // Clear existing content and add new containers
            jobMediaContainer.innerHTML = '';
            const rowContainer = document.createElement('div');
            rowContainer.className = 'row';
            rowContainer.appendChild(videoContainer);
            rowContainer.appendChild(latentsContainer);
            jobMediaContainer.appendChild(rowContainer);
        } else {
            jobMediaContainer.classList.add('d-none');
        }
        
        // Add action buttons based on job status
        const actionContainer = document.createElement('div');
        actionContainer.className = 'mt-3 d-flex flex-wrap gap-2';
        
        // Cancel button if job is running or queued
        if (jobData.status === 'running' || jobData.status === 'queued') {
            const cancelBtn = document.createElement('button');
            cancelBtn.className = 'btn btn-danger';
            cancelBtn.innerHTML = '<i class="bi bi-x-circle"></i> Cancel Job';
            cancelBtn.onclick = () => cancelJob(jobId);
            actionContainer.appendChild(cancelBtn);
        }
        
        // View and download buttons if job is completed
        if (jobData.status === 'completed' && jobData.result_video) {
            const viewBtn = document.createElement('button');
            viewBtn.className = 'btn btn-primary';
            viewBtn.innerHTML = '<i class="bi bi-play-circle"></i> View Video';
            viewBtn.onclick = () => openVideoViewer(jobData.result_video, jobData.result_video.split('/').pop());
            actionContainer.appendChild(viewBtn);
            
            const downloadBtn = document.createElement('a');
            downloadBtn.className = 'btn btn-outline-primary';
            downloadBtn.href = jobData.result_video;
            downloadBtn.download = jobData.result_video.split('/').pop();
            downloadBtn.innerHTML = '<i class="bi bi-download"></i> Download Video';
            actionContainer.appendChild(downloadBtn);
        }
        
        // Reload to timeline button for any job that has settings
        if (jobData.job_settings) {
            const reloadBtn = document.createElement('button');
            reloadBtn.className = 'btn btn-outline-secondary';
            reloadBtn.innerHTML = '<i class="bi bi-arrow-clockwise"></i> Load to Timeline';
            reloadBtn.onclick = () => loadJobToTimeline(jobId);
            actionContainer.appendChild(reloadBtn);
        }
        
        // Rerun button for completed or failed jobs (only if images are valid)
        if ((jobData.status === 'completed' || jobData.status === 'failed') && 
            jobData.job_settings && jobData.is_valid !== false) {
            const rerunBtn = document.createElement('button');
            rerunBtn.className = 'btn btn-outline-success';
            rerunBtn.innerHTML = '<i class="bi bi-arrow-repeat"></i> Rerun Job';
            rerunBtn.onclick = () => rerunJob(jobId);
            actionContainer.appendChild(rerunBtn);
        }
        
        // Delete button for any job
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'btn btn-outline-danger';
        deleteBtn.innerHTML = '<i class="bi bi-trash"></i> Delete Job';
        deleteBtn.onclick = () => {
            if (confirm(`Are you sure you want to delete this job? This will remove all files related to ${jobId}.`)) {
                deleteJob(jobId);
            }
        };
        actionContainer.appendChild(deleteBtn);
        
        // Add the action container
        jobDetailContainer.appendChild(actionContainer);
        
    } catch (error) {
        console.error('Error loading job details:', error);
        document.getElementById('jobDetailContainer').innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i> Error loading job details: ${error.message}
            </div>
        `;
        document.getElementById('jobMediaContainer').classList.add('d-none');
    }
}

// Load job to timeline
async function loadJobToTimeline(jobId) {
    try {
        // Show confirmation if timeline has content
        if (timeline.length > 0) {
            if (!confirm('Loading this job will replace your current timeline. Continue?')) {
                return;
            }
        }
        
        // Show loading indicator
        const jobDetailContainer = document.getElementById('jobDetailContainer');
        const loadingMessage = document.createElement('div');
        loadingMessage.className = 'alert alert-info mt-3';
        loadingMessage.innerHTML = '<i class="bi bi-hourglass"></i> Loading job to timeline...';
        jobDetailContainer.appendChild(loadingMessage);
        
        // Fetch job data
        const response = await fetch(`/api/reload_job/${jobId}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`Failed to load job data: ${response.statusText}`);
        }
        
        const jobData = await response.json();
        
        if (!jobData.settings || !jobData.settings.segments) {
            throw new Error('Invalid job data: missing segments');
        }
        
        // Clear current timeline
        elements.timelineContainer.innerHTML = '';
        timeline = [];
        
        // Set form values
        if (elements.globalPrompt) {
            elements.globalPrompt.value = jobData.settings.global_prompt || '';
        }
        
        if (elements.negativePrompt) {
            elements.negativePrompt.value = jobData.settings.negative_prompt || '';
        }
        
        if (elements.steps) {
            elements.steps.value = jobData.settings.steps || 25;
        }
        
        if (elements.guidanceScale) {
            elements.guidanceScale.value = jobData.settings.guidance_scale || 10.0;
        }
        
        if (elements.resolution) {
            elements.resolution.value = jobData.settings.resolution || 640;
        }
        
        if (elements.useTeacache) {
            elements.useTeacache.checked = jobData.settings.use_teacache !== false;
        }
        
        if (elements.enableAdaptiveMemory) {
            elements.enableAdaptiveMemory.checked = jobData.settings.enable_adaptive_memory !== false;
        }
        
        // Add each segment to timeline
        for (const segment of jobData.settings.segments) {
            // Create a file object representation
            const fileObj = {
                serverPath: segment.image_path,
                duration: segment.duration || 3.0,
                prompt: segment.prompt || '',
                name: segment.image_path.split('/').pop() // Extract filename
            };
            
            // Create a dummy source for display
            fileObj.src = segment.image_path;
            
            // Add to timeline
            addItemToTimeline(fileObj);
        }
        
        // Update timeline status
        updateTimelineStatus();
        
        // Switch to editor tab
        const editorTab = document.getElementById('editor-tab');
        if (editorTab) {
            bootstrap.Tab.getOrCreateInstance(editorTab).show();
        }
        
        // Remove loading message
        loadingMessage.remove();
        
        // Show success message
        showMessage('Job loaded to timeline successfully', 'success');
        
        // Validate images
        if (!jobData.is_valid) {
            showMessage(`Warning: ${jobData.missing_images.length} images are missing. You'll need to replace them before generating.`, 'warning');
        }
        
    } catch (error) {
        console.error('Error loading job to timeline:', error);
        showMessage(`Failed to load job: ${error.message}`, 'error');
    }
}

// Rerun a job
async function rerunJob(jobId) {
    try {
        // Confirm with user
        if (!confirm('Are you sure you want to rerun this job?')) {
            return;
        }
        
        // Show loading indicator
        const jobDetailContainer = document.getElementById('jobDetailContainer');
        const loadingMessage = document.createElement('div');
        loadingMessage.className = 'alert alert-info mt-3';
        loadingMessage.innerHTML = '<i class="bi bi-hourglass"></i> Starting job rerun...';
        jobDetailContainer.appendChild(loadingMessage);
        
        // Call rerun API
        const response = await fetch(`/api/rerun_job/${jobId}`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to rerun job');
        }
        
        const result = await response.json();
        
        // Remove loading message
        loadingMessage.remove();
        
        // Show success and switch to the new job
        showMessage('Job restarted successfully!', 'success');
        
        // Refresh job queue
        await loadJobQueue();
        
        // Select the new job
        const jobItems = document.querySelectorAll('.job-item');
        jobItems.forEach(item => {
            item.classList.remove('active');
            if (item.dataset.jobId === result.job_id) {
                item.classList.add('active');
                // Scroll to the new job
                item.scrollIntoView({ behavior: 'smooth' });
            }
        });
        
        // Load details of the new job
        loadJobDetails(result.job_id);
        
    } catch (error) {
        console.error('Error rerunning job:', error);
        showMessage(`Failed to rerun job: ${error.message}`, 'error');
    }
}

// Delete a job
async function deleteJob(jobId) {
    try {
        const response = await fetch(`/api/job/${jobId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error('Failed to delete job');
        }
        
        const result = await response.json();
        
        if (result.success) {
            // Refresh the job queue to show updated list
            loadJobQueue();
            
            // Clear job details
            document.getElementById('jobDetailContainer').innerHTML = '';
            document.getElementById('jobMediaContainer').classList.add('d-none');
            
            // Show success message
            showMessage('Job deleted successfully', 'success');
        } else {
            throw new Error(result.error || 'Failed to delete job');
        }
        
    } catch (error) {
        console.error('Error deleting job:', error);
        showMessage(`Failed to delete job: ${error.message}`, 'error');
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
    const modal = document.getElementById('videoViewerModal');
    const video = document.getElementById('modalVideo');
    const downloadBtn = document.getElementById('modalDownloadBtn');
    const modalTitle = document.getElementById('videoViewerModalLabel');
    
    video.src = videoPath;
    video.load();
    modalTitle.textContent = videoName || 'Video Viewer';
    downloadBtn.href = videoPath;
    downloadBtn.download = videoName || 'video';
    
    // Handle fullscreen button click
    document.getElementById('fullscreenBtn').addEventListener('click', function() {
        if (video.requestFullscreen) {
            video.requestFullscreen();
        } else if (video.webkitRequestFullscreen) { /* Safari */
            video.webkitRequestFullscreen();
        } else if (video.msRequestFullscreen) { /* IE11 */
            video.msRequestFullscreen();
        }
    });
    
    // Reset the video when the modal is hidden
    modal.addEventListener('hidden.bs.modal', function() {
        video.pause();
        video.currentTime = 0;
        video.src = '';
    }, { once: true });
    
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
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
            border: 1px solid #dee2e6;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 15px;
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
        
        /* Invalid image styling */
        .invalid-image {
            opacity: 0.6;
            border: 2px dashed #dc3545 !important;
        }
        .invalid-image-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: #dc3545;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 10;
        }
        .timeline-item {
            position: relative;
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
    if (files.length > 0) {
        console.log(`Dropped ${files.length} files onto upload area`);
        
        // Clear previous files if we're not in edit mode
        if (!keepCurrentImage) {
            selectedFiles = [];
        }
        
        // Process the files
        processSelectedFiles(files);
        
        // Show the upload modal to review the dropped files
        if (uploadModal && !uploadModal._isShown) {
            uploadModal.show();
        }
    }
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
    
    // Convert FileList to Array to ensure proper iteration
    const filesArray = Array.from(files);
    
    // Process each file
    filesArray.forEach(file => {
        // Only process image files
        if (!file.type.match('image.*')) {
            console.log('Skipping non-image file:', file.name);
            return;
        }
        
        const reader = new FileReader();
        
        reader.onload = (e) => {
            const imgSrc = e.target.result;
            const fileName = file.name;
            
            // Add to selected files array
            selectedFiles.push({
                file: file,
                src: imgSrc,
                name: fileName,
                duration: 3.0  // Default duration is 3.0 seconds
            });
            
            // Create thumbnail
            const thumbnailHtml = `
                <div class="col-4 col-md-3 mb-3">
                    <div class="card">
                        <img src="${imgSrc}" class="card-img-top" alt="${fileName}">
                        <div class="card-body p-2">
                            <p class="card-text small text-muted text-truncate">${fileName}</p>
                            <div class="input-group input-group-sm">
                                <span class="input-group-text">Duration</span>
                                <input type="number" class="form-control image-duration" 
                                    value="3.0" 
                                    min="0.1" max="10" step="0.1">
                                <span class="input-group-text">s</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            elements.imageUploadContainer.insertAdjacentHTML('beforeend', thumbnailHtml);
            
            // Add change event to duration inputs
            const allDurationInputs = elements.imageUploadContainer.querySelectorAll('.image-duration');
            const lastInput = allDurationInputs[allDurationInputs.length - 1];
            if (lastInput) {
                lastInput.addEventListener('change', (e) => {
                    const index = Array.from(allDurationInputs).indexOf(e.target);
                    if (index >= 0 && index < selectedFiles.length) {
                        selectedFiles[index].duration = parseFloat(e.target.value);
                    }
                });
            }
            
            console.log(`Processed file: ${fileName}, total files: ${selectedFiles.length}`);
        };
        
        reader.readAsDataURL(file);
    });
    
    // The upload modal will be shown by the caller
}

// Function to handle adding to timeline
function handleAddToTimeline() {
    if (selectedFiles.length === 0) {
        alert('Please select at least one image to add to the timeline.');
        return;
    }
    
    console.log(`Adding ${selectedFiles.length} files to timeline`);
    
    // Show loading indicator
    const addToTimelineBtn = document.getElementById('addToTimelineBtn');
    if (addToTimelineBtn) {
        addToTimelineBtn.disabled = true;
        addToTimelineBtn.innerHTML = '<i class="bi bi-hourglass-split me-2"></i> Uploading...';
    }
    
    // First, upload all files to the server
    const uploadPromises = selectedFiles.map(fileObj => uploadFileToServer(fileObj.file));
    
    Promise.all(uploadPromises)
        .then(serverPaths => {
            console.log('All files uploaded successfully', serverPaths);
            
            // Check if we're in edit mode (replacing an image)
            if (keepCurrentImage && currentEditIndex >= 0) {
                // We're replacing an existing image in the timeline
                const timelineItems = Array.from(elements.timelineContainer.children);
                if (currentEditIndex < timelineItems.length) {
                    // Get the first uploaded file (only allow one replacement)
                    if (serverPaths[0]) {
                        // Update the timeline array
                        if (timeline[currentEditIndex]) {
                            timeline[currentEditIndex].serverPath = serverPaths[0];
                            timeline[currentEditIndex].src = selectedFiles[0].src;
                            timeline[currentEditIndex].file = selectedFiles[0].file;
                            timeline[currentEditIndex].valid = true; // Mark as valid since it's new
                            
                            // Update the DOM
                            const imgElement = timelineItems[currentEditIndex].querySelector('img');
                            if (imgElement) {
                                imgElement.src = selectedFiles[0].src;
                                imgElement.title = serverPaths[0];
                                imgElement.classList.remove('invalid-image');
                            }
                            
                            // Remove any invalid badge
                            const invalidBadge = timelineItems[currentEditIndex].querySelector('.invalid-image-badge');
                            if (invalidBadge) {
                                invalidBadge.remove();
                            }
                            
                            // Remove any replace button
                            const replaceBtn = timelineItems[currentEditIndex].querySelector('.replace-image-btn');
                            if (replaceBtn) {
                                replaceBtn.remove();
                            }
                            
                            console.log(`Replaced image at index ${currentEditIndex} with ${serverPaths[0]}`);
                            showMessage('Image replaced successfully', 'success');
                        }
                    }
                }
                
                // Reset edit mode
                keepCurrentImage = false;
                currentEditIndex = -1;
            } else {
                // Normal mode - add each file to the timeline with its server path
                selectedFiles.forEach((fileObj, index) => {
                    if (serverPaths[index]) {
                        // Use the server path instead of local file reference
                        // This is the EXACT path returned from the server
                        fileObj.serverPath = serverPaths[index];
                        console.log(`Adding file ${index + 1}/${selectedFiles.length}: ${fileObj.name} (${serverPaths[index]})`);
                        addItemToTimeline(fileObj);
                    }
                });
                
                // Show confirmation toast or message
                const count = selectedFiles.length;
                const message = count === 1 
                    ? '1 image added to timeline' 
                    : `${count} images added to timeline`;
                
                // Make sure timeline UI is updated
                updateTimelineStatus();
                
                // Remove any existing secondary drop zone to let updateTimelineStatus create it new
                const existingDropZone = document.getElementById('secondaryDropZone');
                if (existingDropZone) {
                    existingDropZone.remove();
                }
                
                // Call updateTimelineStatus again to ensure the secondary drop zone is created
                updateTimelineStatus();
            }
            
            // Reset selected files
            selectedFiles = [];
            
            // Close modal
            uploadModal.hide();
        })
        .catch(error => {
            console.error('Error uploading files:', error);
            alert('Error uploading files: ' + error.message);
        })
        .finally(() => {
            // Reset the button state
            if (addToTimelineBtn) {
                addToTimelineBtn.disabled = false;
                addToTimelineBtn.innerHTML = 'Add to Timeline';
            }
            
            // Reset edit mode
            keepCurrentImage = false;
            currentEditIndex = -1;
        });
}

// Function to upload a file to the server
function uploadFileToServer(file) {
    return new Promise((resolve, reject) => {
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/api/upload_image', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to upload file');
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                // Use the full server path returned from the API
                const serverPath = data.path;
                
                if (!serverPath) {
                    throw new Error('No valid file path returned from server');
                }
                
                console.log(`File uploaded: ${serverPath}`);
                resolve(serverPath);
            } else {
                reject(new Error(data.error || 'Unknown error uploading file'));
            }
        })
        .catch(error => {
            console.error('Error uploading file:', error);
            reject(error);
        });
    });
}

// Check if an image path exists
async function checkImageExists(imagePath) {
    try {
        const response = await fetch(imagePath, { method: 'HEAD' });
        return response.ok;
    } catch (error) {
        console.error('Error checking image:', error);
        return false;
    }
}

// Function to add an item to the timeline
async function addItemToTimeline(fileObj) {
    // Remove the main dropzone if this is the first item
    const mainDropZone = elements.timelineContainer.querySelector('.timeline-dropzone');
    if (mainDropZone) {
        mainDropZone.remove();
    }
    
    const timelineItem = document.createElement('div');
    timelineItem.className = 'timeline-item card';
    timelineItem.draggable = true;
    
    // Get the display source and server path
    let displaySrc = '';
    let serverPath = '';
    
    if (fileObj.serverPath) {
        // We already have a server path from a previous upload
        serverPath = fileObj.serverPath;
        displaySrc = fileObj.src || URL.createObjectURL(fileObj.file);
    } else if (fileObj.file) {
        // We have a file but no server path yet - upload it
        uploadFileToServer(fileObj.file).then(response => {
            if (response.success) {
                // Update the timeline item with the new server path
                const imgElement = timelineItem.querySelector('img');
                if (imgElement) {
                    imgElement.title = response.path;
                }
                
                // Update the timeline array
                const index = Array.from(elements.timelineContainer.children).indexOf(timelineItem);
                if (index >= 0 && index < timeline.length) {
                    timeline[index].serverPath = response.path;
                }
                
                console.log(`File uploaded: ${response.filename}, server path: ${response.path}`);
            } else {
                console.error('Failed to upload file:', response.error);
                showMessage(`Failed to upload file: ${response.error}`, 'error');
            }
        }).catch(error => {
            console.error('Error uploading file:', error);
            showMessage(`Error uploading file: ${error.message}`, 'error');
        });
        
        displaySrc = URL.createObjectURL(fileObj.file);
    }
    
    // Default duration for new frames is 3.0 seconds
    const frameDuration = fileObj.duration || 3.0;
    
    // Create a frame number for visual reference
    const frameNumber = elements.timelineContainer.querySelectorAll('.timeline-item').length + 1;
    
    // Check if image is missing
    let imageExists = true;
    let invalidImageClass = '';
    let invalidImageBadge = '';
    
    if (serverPath) {
        // Only check server path if it's a loaded/saved image
        imageExists = await checkImageExists(serverPath);
        
        if (!imageExists) {
            invalidImageClass = 'invalid-image';
            invalidImageBadge = `
                <div class="invalid-image-badge">
                    <i class="bi bi-exclamation-triangle"></i> Missing
                </div>
            `;
        }
    }
    
    timelineItem.innerHTML = `
        <div class="frame-number">${frameNumber}</div>
        <img src="${displaySrc}" class="img-fluid rounded ${invalidImageClass}" alt="${fileObj.name}" title="${serverPath}">
        ${invalidImageBadge}
        
        <div class="timeline-item-duration">
            <label class="form-label small mb-1">Duration</label>
            <div class="input-group input-group-sm">
                <input type="number" class="form-control duration-input" value="${frameDuration}" min="0.1" max="10" step="0.1">
                <span class="input-group-text">sec</span>
            </div>
        </div>
        
        <div class="timeline-item-prompt">
            <label class="form-label small mb-1">Prompt</label>
            <textarea class="form-control prompt-text" rows="2" placeholder="Frame prompt (optional)">${fileObj.prompt || ''}</textarea>
        </div>
        
        <div class="timeline-item-actions">
            <button class="btn btn-sm btn-outline-primary edit-frame-btn">
                <i class="bi bi-pencil"></i> Edit
            </button>
            <button class="btn btn-sm btn-outline-danger delete-frame-btn">
                <i class="bi bi-trash"></i> Remove
            </button>
            ${!imageExists ? `
            <button class="btn btn-sm btn-warning replace-image-btn">
                <i class="bi bi-arrow-repeat"></i> Replace
            </button>` : ''}
        </div>
    `;
    
    // Attach event listeners
    const editBtn = timelineItem.querySelector('.edit-frame-btn');
    const deleteBtn = timelineItem.querySelector('.delete-frame-btn');
    const durationInput = timelineItem.querySelector('.duration-input');
    const replaceBtn = timelineItem.querySelector('.replace-image-btn');
    
    if (editBtn) {
        editBtn.addEventListener('click', () => {
            showFrameEditModal(timelineItem);
        });
    }
    
    if (deleteBtn) {
        deleteBtn.addEventListener('click', () => {
            timelineItem.remove();
            updateTimelineArray();
            updateTimelineStatus();
            
            // If timeline is now empty after removal, ensure no secondary drop zone
            if (timeline.length === 0) {
                const secondaryDropZone = document.getElementById('secondaryDropZone');
                if (secondaryDropZone) {
                    secondaryDropZone.remove();
                }
            }
        });
    }
    
    if (durationInput) {
        durationInput.addEventListener('change', () => {
            const index = Array.from(elements.timelineContainer.children).indexOf(timelineItem);
            if (index >= 0 && index < timeline.length) {
                timeline[index].duration = parseFloat(durationInput.value);
            }
        });
    }
    
    if (replaceBtn) {
        replaceBtn.addEventListener('click', () => {
            // Set the current item for replacement
            currentEditIndex = Array.from(elements.timelineContainer.children).indexOf(timelineItem);
            
            // Set the flag to keep the current image in the timeline
            keepCurrentImage = true;
            
            // Show the upload modal to select a replacement
            showUploadModal();
        });
    }
    
    // Add drag and drop event listeners
    timelineItem.addEventListener('dragstart', handleDragStart);
    timelineItem.addEventListener('dragover', handleTimelineDragOver);
    timelineItem.addEventListener('dragleave', handleTimelineDragLeave);
    timelineItem.addEventListener('drop', handleTimelineDrop);
    timelineItem.addEventListener('dragend', handleDragEnd);
    
    timeline.push({
        src: displaySrc,
        file: fileObj.file,
        serverPath: serverPath,
        duration: frameDuration,
        prompt: fileObj.prompt || '',
        valid: imageExists
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
    
    // Add visual indicator that this is a valid drop target
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
            // Moving left (up in horizontal layout)
            elements.timelineContainer.insertBefore(dragSrcEl, this);
        } else {
            // Moving right (down in horizontal layout)
            if (this.nextSibling) {
                elements.timelineContainer.insertBefore(dragSrcEl, this.nextSibling);
            } else {
                elements.timelineContainer.appendChild(dragSrcEl);
            }
        }
        
        // Update the timeline array to match the new order
        updateTimelineArray();
        
        // Update frame numbers and the last frame info
        updateTimelineStatus();
    }
    
    return false;
}

function handleDragEnd(e) {
    // Remove visual styles when drag ends
    this.classList.remove('dragging');
    
    const items = elements.timelineContainer.querySelectorAll('.timeline-item');
    items.forEach(item => {
        item.classList.remove('drag-over');
    });
}

// Function to update the timeline array based on the DOM order
function updateTimelineArray() {
    const newTimeline = [];
    const items = Array.from(elements.timelineContainer.querySelectorAll('.timeline-item'));
    
    items.forEach((item, index) => {
        const img = item.querySelector('img');
        const promptText = item.querySelector('.prompt-text');
        const durationInput = item.querySelector('.duration-input');
        
        // Find matching item in original timeline
        const originalItem = timeline.find(t => {
            return t.src === img.src;
        });
        
        if (originalItem) {
            // CRITICAL: Make sure we preserve the serverPath property
            newTimeline.push({
                ...originalItem,
                prompt: promptText ? promptText.value : '',
                duration: durationInput ? parseFloat(durationInput.value) : originalItem.duration,
                // Explicitly include serverPath to ensure it's preserved
                serverPath: originalItem.serverPath
            });
        }
    });
    
    // Replace timeline with new ordered array
    timeline = newTimeline;
    
    // Log for debugging
    console.log('Updated timeline array:', timeline.map(item => ({
        name: item.file?.name,
        serverPath: item.serverPath,
        duration: item.duration
    })));
}

// Add timeline drop zone for drag & drop from desktop
function initTimelineDropZone() {
    if (!elements.timelineContainer) return;
    
    // Initialize empty timeline UI
    updateTimelineStatus();
    
    // Ensure horizontal layout
    enforceHorizontalLayout();
    
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
            // Instead of processing immediately, show the upload modal first
            selectedFiles = []; // Clear any previously selected files
            processSelectedFiles(e.dataTransfer.files);
            
            // Show the upload modal to allow user to confirm/adjust before adding to timeline
            if (uploadModal) {
                uploadModal.show();
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
    
    // Update UI
    updateTimelineArray();
    updateTimelineStatus();
    
    // If timeline is now empty after removal, ensure no secondary drop zone
    if (timeline.length === 0) {
        const secondaryDropZone = document.getElementById('secondaryDropZone');
        if (secondaryDropZone) {
            secondaryDropZone.remove();
        }
    }
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
        const durationInput = timelineItems[currentEditIndex].querySelector('.duration-input');
        if (durationInput) {
            durationInput.value = duration;
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
        // Need at least 1 frame to generate a video
        elements.generateVideoBtn.disabled = timeline.length < 1;
    }
    
    // Also update Clear Timeline button if it exists
    const clearBtn = document.getElementById('clearTimelineBtn');
    if (clearBtn) {
        clearBtn.disabled = timeline.length < 1;
    }
    
    // Show message when timeline is empty
    if (timeline.length === 0) {
        elements.timelineContainer.innerHTML = `
            <div class="timeline-dropzone">
                <div class="text-center py-5">
                    <i class="bi bi-images fs-1 mb-3"></i>
                    <h5 class="fw-bold mb-3">Add Images to Timeline</h5>
                    <p class="text-muted mb-4">Upload one or more images to create videos</p>
                    <button class="btn btn-primary mt-2 px-4 py-2" id="dropzoneUploadBtn">
                        <i class="bi bi-upload me-2"></i> Browse for Images
                    </button>
                </div>
            </div>
        `;
        
        // Add click event for the upload button
        const dropzoneUploadBtn = document.getElementById('dropzoneUploadBtn');
        if (dropzoneUploadBtn) {
            dropzoneUploadBtn.addEventListener('click', showUploadModal);
        }
        
        // Remove any secondary drop zone when empty
        const secondaryDropZone = document.getElementById('secondaryDropZone');
        if (secondaryDropZone) {
            secondaryDropZone.remove();
        }
    } else {
        // When timeline has items, remove the main dropzone if it exists
        const mainDropZone = elements.timelineContainer.querySelector('.timeline-dropzone');
        if (mainDropZone) {
            mainDropZone.remove();
        }
        
        // Add a secondary drop zone below the timeline when items exist
        if (!document.getElementById('secondaryDropZone')) {
            const secondaryDropZone = document.createElement('div');
            secondaryDropZone.id = 'secondaryDropZone';
            secondaryDropZone.className = 'timeline-dropzone mt-4';
            secondaryDropZone.innerHTML = `
                <div class="text-center py-3">
                    <i class="bi bi-plus-circle fs-2 mb-2"></i>
                    <h6 class="fw-bold mb-2">Add More Images</h6>
                    <p class="text-muted small mb-0">Drag and drop more images here</p>
                </div>
            `;
            
            // Add event listeners for the secondary drop zone
            secondaryDropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.stopPropagation();
                secondaryDropZone.classList.add('active');
            });
            
            secondaryDropZone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                e.stopPropagation();
                secondaryDropZone.classList.remove('active');
            });
            
            secondaryDropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                e.stopPropagation();
                secondaryDropZone.classList.remove('active');
                
                // Handle files dropped on the secondary zone
                if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                    // Clear previous selection and process the new files
                    selectedFiles = [];
                    processSelectedFiles(e.dataTransfer.files);
                    
                    // Show the upload modal
                    if (uploadModal) {
                        uploadModal.show();
                    }
                }
            });
            
            secondaryDropZone.addEventListener('click', () => {
                showUploadModal();
            });
            
            // Append the secondary drop zone after the timeline container
            elements.timelineContainer.after(secondaryDropZone);
        }
        
        // Add note to the last item that its duration is not used
        const timelineItems = elements.timelineContainer.querySelectorAll('.timeline-item');
        const lastIndex = timelineItems.length - 1;
        
        // First remove any previous notes from all items
        timelineItems.forEach((item, idx) => {
            const existingNote = item.querySelector('.last-frame-info');
            if (existingNote) {
                existingNote.remove();
            }
            
            // Re-enable the duration input for all items
            const durationInput = item.querySelector('.duration-input');
            if (durationInput) {
                durationInput.disabled = false;
            }
        });
        
        // Add note to the last item
        if (lastIndex >= 0) {
            const lastItem = timelineItems[lastIndex];
            const durationSection = lastItem.querySelector('.timeline-item-duration');
            
            if (durationSection) {
                const note = document.createElement('div');
                note.className = 'last-frame-info';
                note.innerHTML = '<i class="bi bi-info-circle"></i> Duration not used for last frame';
                durationSection.appendChild(note);
                
                // Disable the duration input for the last item
                const durationInput = lastItem.querySelector('.duration-input');
                if (durationInput) {
                    durationInput.disabled = true;
                }
            }
        }
        
        // Update frame numbers for each card
        timelineItems.forEach((item, idx) => {
            const frameNumber = item.querySelector('.frame-number');
            if (frameNumber) {
                frameNumber.textContent = idx + 1;
            }
        });
    }
    
    // Ensure horizontal layout
    enforceHorizontalLayout();
}

// Function to clear the timeline
function clearTimeline() {
    if (timeline.length === 0) {
        return; // Nothing to clear
    }
    
    if (confirm('Are you sure you want to clear the timeline? This will remove all images.')) {
        // Clear UI
        elements.timelineContainer.innerHTML = '';
        
        // Clear timeline array
        timeline = [];
        
        // Update status
        updateTimelineStatus();
        
        // Show confirmation message
        showMessage('Timeline cleared', 'info');
    }
}

function startGeneration() {
    if (timeline.length === 0) {
        alert('Please add at least one image to the timeline before generating.');
        return;
    }
    
    // Check for invalid images in the timeline
    const invalidImages = timeline.filter(item => item.valid === false);
    if (invalidImages.length > 0) {
        alert(`Cannot start generation: ${invalidImages.length} image(s) are missing or invalid. Please replace them before generating.`);
        return;
    }
    
    // Show progress UI
    elements.progressContainer.classList.remove('d-none');
    elements.progressBar.style.width = '0%';
    elements.progressBar.setAttribute('aria-valuenow', 0);
    elements.progressBar.textContent = '0%';
    elements.progressStatus.textContent = 'Preparing generation request...';
    
    // Collect segments data
    const segments = [];
    
    // For a single image, we need to create a special case
    if (timeline.length === 1) {
        const singleFrame = timeline[0];
        const promptInput = elements.timelineContainer.querySelector('.prompt-text');
        
        // Use the full server path directly
        const imagePath = singleFrame.serverPath;
        if (!imagePath) {
            console.error('No server path found for the single image:', singleFrame);
            alert('Error: Missing server path for the image. Please try re-uploading.');
            return;
        }
        
        console.log(`Single image path: ${imagePath}`);
        
        // Add single segment with longer duration
        segments.push({
            image_path: imagePath,
            prompt: promptInput ? promptInput.value : '',
            duration: singleFrame.duration || 3.0
        });
    } else {
        // Process multiple images: Create N-1 segments for N frames
        for (let i = 0; i < timeline.length - 1; i++) {
            const currentFrame = timeline[i];
            const promptInput = Array.from(elements.timelineContainer.children)
                .find((_, idx) => idx === i)
                ?.querySelector('.prompt-text');
            
            // Use the full server path directly
            const imagePath = currentFrame.serverPath;
            if (!imagePath) {
                console.error(`No server path found for image at index ${i}:`, currentFrame);
                alert(`Error: Missing server path for image ${i + 1}. Please try re-uploading.`);
                return;
            }
            
            // Log each path for debugging
            console.log(`Segment ${i + 1} path: ${imagePath}`);
            
            segments.push({
                image_path: imagePath,
                prompt: promptInput ? promptInput.value : '',
                duration: currentFrame.duration
            });
        }
    }
    
    if (segments.length === 0) {
        // Reset progress UI
        elements.progressContainer.classList.add('d-none');
        return;
    }
    
    // Generate a simple job name based on first image filename and timestamp
    const currentDate = new Date();
    const timestamp = currentDate.toISOString().slice(0, 16).replace('T', ' ');
    const firstImageName = segments[0].image_path.split('/').pop().split('.')[0];
    const jobName = `${firstImageName} - ${timestamp}`;
    
    // Create request payload with all form settings
    const payload = {
        global_prompt: elements.globalPrompt ? elements.globalPrompt.value : "",
        negative_prompt: elements.negativePrompt ? elements.negativePrompt.value : "",
        segments: segments,
        job_name: jobName,
        seed: Math.floor(Math.random() * 100000),
        steps: elements.steps ? parseInt(elements.steps.value) : 25,
        guidance_scale: elements.guidanceScale ? parseFloat(elements.guidanceScale.value) : 10.0,
        use_teacache: elements.useTeacache ? elements.useTeacache.checked : true,
        enable_adaptive_memory: elements.enableAdaptiveMemory ? elements.enableAdaptiveMemory.checked : true,
        resolution: elements.resolution ? parseInt(elements.resolution.value) : 640,
        mp4_crf: 16,
        gpu_memory_preservation: 6.0
    };
    
    console.log('Sending generation request with payload:', JSON.stringify(payload, null, 2));
    
    // Make API call
    fetch('/api/generate_video', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (!response.ok) {
            return response.text().then(text => {
                throw new Error(`Failed to start generation: ${text}`);
            });
        }
        return response.json();
    })
    .then(data => {
        console.log('Generation started:', data);
        elements.progressStatus.textContent = 'Generation started! Monitoring progress...';
        
        // Store current job ID
        currentJobId = data.job_id;
        
        // Clear the timeline after successfully starting a job
        // Clear UI
        elements.timelineContainer.innerHTML = '';
        
        // Clear timeline array
        timeline = [];
        
        // Update status
        updateTimelineStatus();
        
        // Show message
        showMessage('Job started successfully and timeline cleared for new work', 'success');
        
        // Start polling for job status
        pollJobStatus(data.job_id);
    })
    .catch(error => {
        console.error('Error starting generation:', error);
        elements.progressStatus.textContent = 'Error: ' + error.message;
        elements.progressBar.classList.add('bg-danger');
    });
}

// Function to poll job status
function pollJobStatus(jobId) {
    // Set up interval for polling the job status
    const statusInterval = setInterval(async () => {
        try {
            const response = await fetch(`/api/job_status/${jobId}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch job status: ${response.statusText}`);
            }
            
            const data = await response.json();
            const progressBar = document.getElementById('progressBar');
            const progressStatus = document.getElementById('progressStatus');
            const progressContainer = document.getElementById('progressContainer');
            const generateBtn = document.getElementById('generateVideoBtn');
            const currentJobImage = document.getElementById('currentJobImage');
            const currentJobThumbnail = document.getElementById('currentJobThumbnail');
            
            // Update the progress bar and status message
            progressBar.style.width = `${data.progress}%`;
            progressBar.setAttribute('aria-valuenow', data.progress);
            progressBar.textContent = `${data.progress}%`;
            progressStatus.textContent = data.message || 'Processing...';
            
            // Show the thumbnail if we have segments
            if (data.segments && data.segments.length > 0) {
                currentJobImage.src = data.segments[0];
                currentJobThumbnail.classList.remove('d-none');
                
                // Set up click handler to go to job queue tab and select this job
                currentJobThumbnail.onclick = () => {
                    // Switch to the job queue tab
                    const queueTab = document.getElementById('queue-tab');
                    bootstrap.Tab.getOrCreateInstance(queueTab).show();
                    
                    // Select this job
                    loadJobDetails(jobId);
                    
                    // Highlight this job in the list
                    const jobItems = document.querySelectorAll('.job-item');
                    jobItems.forEach(item => {
                        item.classList.remove('active');
                        if (item.dataset.jobId === jobId) {
                            item.classList.add('active');
                        }
                    });
                };
            } else {
                // Hide the thumbnail if no segments are available yet
                currentJobThumbnail.classList.add('d-none');
            }
            
            // Check if job is completed or failed
            if (data.status === 'completed' || data.status === 'failed') {
                clearInterval(statusInterval);
                
                if (data.status === 'completed') {
                    progressStatus.textContent = 'Video generation completed!';
                    
                    // If we have a result video, show it
                    if (data.result_video) {
                        const resultContainer = document.createElement('div');
                        resultContainer.className = 'mt-3';
                        resultContainer.innerHTML = `
                            <h4 class="fs-6">Result</h4>
                            <div class="d-flex flex-column">
                                <video id="resultVideo" src="${data.result_video}" controls class="img-fluid rounded mb-2"></video>
                                <div class="btn-group">
                                    <a href="${data.result_video}" download class="btn btn-primary">
                                        <i class="bi bi-download"></i> Download
                                    </a>
                                    <button type="button" class="btn btn-outline-primary" onclick="openVideoViewer('${data.result_video}', '${data.result_video.split('/').pop()}')">
                                        <i class="bi bi-fullscreen"></i> Fullscreen
                                    </button>
                                </div>
                            </div>
                        `;
                        progressContainer.appendChild(resultContainer);
                    }
                    
                    // Success message and offer to clear timeline
                    showMessage('Video generation completed successfully!', 'success');
                    
                    // Clear the timeline after successful generation
                    if (confirm('Generation completed! Would you like to clear the timeline for a new project?')) {
                        // Clear timeline
                        elements.timelineContainer.innerHTML = '';
                        timeline = [];
                        updateTimelineStatus();
                    }
                    
                } else {
                    progressStatus.textContent = `Failed: ${data.message}`;
                    showMessage(`Video generation failed: ${data.message}`, 'danger');
                }
                
                // Re-enable the generate button
                generateBtn.disabled = false;
                
                // Load the updated job list
                loadJobQueue();
                
                // Load the updated outputs
                loadOutputs();
                
                return;
            }
            
        } catch (error) {
            console.error('Error polling job status:', error);
            clearInterval(statusInterval);
            document.getElementById('progressStatus').textContent = `Error: ${error.message}`;
            document.getElementById('generateVideoBtn').disabled = false;
        }
    }, 2000);
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

// Function to initialize the theme based on saved preference
function initTheme() {
    const darkModeToggle = document.getElementById('darkModeToggle');
    const htmlElement = document.documentElement;
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    
    // Apply saved theme or detect system preference
    if (savedTheme) {
        htmlElement.setAttribute('data-bs-theme', savedTheme);
        darkModeToggle.checked = savedTheme === 'dark';
    } else {
        // Use system preference as fallback
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        if (prefersDark) {
            htmlElement.setAttribute('data-bs-theme', 'dark');
            darkModeToggle.checked = true;
        }
    }
    
    // Update icon based on current theme
    updateThemeIcon(darkModeToggle.checked);
}

// Function to toggle dark mode
function toggleDarkMode() {
    const darkModeToggle = document.getElementById('darkModeToggle');
    const htmlElement = document.documentElement;
    
    if (darkModeToggle.checked) {
        htmlElement.setAttribute('data-bs-theme', 'dark');
        localStorage.setItem('theme', 'dark');
    } else {
        htmlElement.setAttribute('data-bs-theme', 'light');
        localStorage.setItem('theme', 'light');
    }
    
    // Update icon based on current theme
    updateThemeIcon(darkModeToggle.checked);
}

// Function to update theme icon
function updateThemeIcon(isDark) {
    const icon = document.querySelector('label[for="darkModeToggle"] i');
    if (icon) {
        if (isDark) {
            icon.className = 'bi bi-moon-stars-fill';
        } else {
            icon.className = 'bi bi-brightness-high';
        }
    }
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