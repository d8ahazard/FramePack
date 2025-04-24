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

// Add these variables at the top of your script
let activeJobId = null;
let jobQueueInterval = null;
let jobDetailsInterval = null;

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
    
    // Add queue tab event listener for auto-updates
    const queueTabBtn = document.getElementById('queue-tab');
    if (queueTabBtn) {
        queueTabBtn.addEventListener('shown.bs.tab', function (e) {
            loadJobQueue();
            startJobQueuePolling();
        });
        
        queueTabBtn.addEventListener('hidden.bs.tab', function (e) {
            stopJobQueuePolling();
        });
    }
}

// Load the job queue UI
async function loadJobQueue() {
    const jobsContainer = document.getElementById('jobsContainer');
    if (!jobsContainer) return;
    
    try {
        const response = await fetch('/api/list_jobs');
        const jobs = await response.json();
        
        if (jobs.length === 0) {
            jobsContainer.innerHTML = `
                <div class="alert alert-info">
                    <i class="bi bi-info-circle me-2"></i>
                    No jobs in queue
                </div>
            `;
            return;
        }
        
        // Sort jobs: running, queued, completed, failed
        jobs.sort((a, b) => {
            const statusOrder = {
                'running': 0,
                'queued': 1,
                'completed': 2,
                'failed': 3
            };
            
            const aOrder = statusOrder[a.status] || 99;
            const bOrder = statusOrder[b.status] || 99;
            
            // Sort by status first, then by job_id (newer first) for same status
            if (aOrder === bOrder) {
                // Assuming job_id contains timestamp info (newer = higher)
                return b.job_id.localeCompare(a.job_id);
            }
            
            return aOrder - bOrder;
        });
        
        // Check if there's a running job
        const runningJob = jobs.find(job => job.status === 'running' || job.status === 'queued');
        if (runningJob) {
            activeJobId = runningJob.job_id;
            
            // If we're in the editor tab, show the progress bar
            const editorTab = document.getElementById('editor-tab-pane');
            if (editorTab && editorTab.classList.contains('active')) {
                showProgress(runningJob.progress, runningJob.message);
                
                // If we have current latents, update the preview
                if (runningJob.current_latents) {
                    updateCurrentJobPreview(runningJob.current_latents);
                }
            }
        }
        
        // Build job cards
        jobsContainer.innerHTML = ''; // Clear existing content
        
        jobs.forEach(job => {
            const jobCard = document.createElement('div');
            jobCard.className = `card mb-2 job-card ${job.job_id === activeJobId ? 'border-primary' : ''}`;
            jobCard.dataset.jobId = job.job_id;
            
            // Determine status badge
            let statusBadge = 'bg-secondary';
            if (job.status === 'completed') statusBadge = 'bg-success';
            else if (job.status === 'running') statusBadge = 'bg-primary';
            else if (job.status === 'failed') statusBadge = 'bg-danger';
            else if (job.status === 'queued') statusBadge = 'bg-warning';
            
            // Format job name or use id
            const displayName = job.job_name || `Job ${job.job_id.substring(0, 8)}...`;
            
            jobCard.innerHTML = `
                <div class="card-body py-2 px-3">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <h6 class="card-title mb-0">${displayName}</h6>
                            <small class="text-muted">${formatJobTimestamp(job.job_id)}</small>
                        </div>
                        <span class="badge ${statusBadge}">${job.status}</span>
                    </div>
                    ${job.status === 'running' ? 
                        `<div class="progress mt-2" style="height: 5px;">
                            <div class="progress-bar" role="progressbar" style="width: ${job.progress}%" 
                                aria-valuenow="${job.progress}" aria-valuemin="0" aria-valuemax="100"></div>
                        </div>` : ''}
                </div>
            `;
            
            jobCard.addEventListener('click', () => loadJobDetails(job.job_id));
            jobsContainer.appendChild(jobCard);
        });
        
    } catch (error) {
        console.error('Error loading job queue:', error);
        jobsContainer.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle me-2"></i>
                Failed to load jobs: ${error.message}
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
    const jobDetailContainer = document.getElementById('jobDetailContainer');
    const jobMediaContainer = document.getElementById('jobMediaContainer');
    
    if (!jobDetailContainer) return;
    
    try {
        // Show loading state
        jobDetailContainer.innerHTML = `
            <div class="alert alert-info">
                <i class="bi bi-hourglass-split me-2"></i>
                Loading job details...
            </div>
        `;
        
        // Fetch job status
        const response = await fetch(`/api/job_status/${jobId}`);
        const jobData = await response.json();
        
        if (jobData.error) {
            jobDetailContainer.innerHTML = `
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle me-2"></i>
                    ${jobData.error}
                </div>
            `;
            return;
        }
        
        // Create the job info card
        const jobInfoCard = document.createElement('div');
        jobInfoCard.className = 'card mb-4';
        
        // Determine status badge class
        let statusBadgeClass = 'bg-secondary';
        if (jobData.status === 'completed') {
            statusBadgeClass = 'bg-success';
        } else if (jobData.status === 'running') {
            statusBadgeClass = 'bg-primary';
        } else if (jobData.status === 'failed') {
            statusBadgeClass = 'bg-danger';
        }
        
        // Check for invalid images
        const hasInvalidImages = !jobData.is_valid && jobData.missing_images && jobData.missing_images.length > 0;
        
        // Optional job name
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
                <span class="badge ${statusBadgeClass}">${jobData.status.charAt(0).toUpperCase() + jobData.status.slice(1)}</span>
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
            jobMediaContainer.innerHTML = ''; // Clear previous content
            
            // Show current latents if available - displayed first for better visibility during generation
            if (jobData.current_latents) {
                const latentsCard = document.createElement('div');
                latentsCard.className = 'card mb-4';
                
                // Add cache-busting timestamp to ensure image is refreshed
                const cacheBuster = `?t=${new Date().getTime()}`;
                
                latentsCard.innerHTML = `
                    <div class="card-header">
                        <h5 class="card-title mb-0">Current Progress</h5>
                    </div>
                    <div class="card-body">
                        <div style="height: 100px; overflow: hidden;">
                            <img src="${jobData.current_latents}${cacheBuster}" class="img-fluid border rounded" style="width: 100%; object-fit: contain;" alt="Latent visualization">
                        </div>
                    </div>
                `;
                jobMediaContainer.appendChild(latentsCard);
            }
            
            // Handle video preview - either from result_video or from segments
            const videoSrc = jobData.result_video || (jobData.segments && jobData.segments.length > 0 ? jobData.segments[0] : null);
            
            if (videoSrc) {
                const videoCard = document.createElement('div');
                videoCard.className = 'card mb-4';
                videoCard.innerHTML = `
                    <div class="card-header">
                        <h5 class="card-title mb-0">Output Video</h5>
                    </div>
                    <div class="card-body">
                        <div class="text-center" style="max-height: 360px; overflow: hidden;">
                            <video class="img-fluid mb-3" style="max-width: 100%; max-height: 320px; object-fit: contain;" controls autoplay loop>
                                <source src="${videoSrc}" type="video/mp4">
                                Your browser does not support video playback.
                            </video>
                        </div>
                        <div class="d-grid gap-2">
                            <button class="btn btn-primary" onclick="openVideoViewer('${videoSrc}', '${jobData.job_name || 'Job ' + jobId}')">
                                <i class="bi bi-play-circle me-2"></i>Open in Viewer
                            </button>
                            <a class="btn btn-outline-primary" href="${videoSrc}" download>
                                <i class="bi bi-download me-2"></i>Download Video
                            </a>
                        </div>
                    </div>
                `;
                jobMediaContainer.appendChild(videoCard);
            } else if (jobData.status === 'running') {
                // Show processing message only when no video yet but job is running
                const noVideoCard = document.createElement('div');
                noVideoCard.className = 'card mb-4';
                noVideoCard.innerHTML = `
                    <div class="card-header">
                        <h5 class="card-title mb-0">Output Video</h5>
                    </div>
                    <div class="card-body">
                        <div class="alert alert-info">
                            <i class="bi bi-hourglass-split me-2"></i>
                            Video is still processing...
                        </div>
                    </div>
                `;
                jobMediaContainer.appendChild(noVideoCard);
            }
            
            // Add job actions card
            const actionsCard = document.createElement('div');
            actionsCard.className = 'card';
            
            let actionButtons = '';
            
            // Reload to timeline
            if (jobData.job_settings) {
                actionButtons += `
                    <button class="btn btn-outline-primary mb-2" onclick="loadJobToTimeline('${jobId}')">
                        <i class="bi bi-arrow-repeat me-2"></i>Load to Timeline
                    </button>
                `;
            }
            
            // Rerun job (if completed or failed)
            if ((jobData.status === 'completed' || jobData.status === 'failed') && jobData.is_valid) {
                actionButtons += `
                    <button class="btn btn-outline-success mb-2" onclick="rerunJob('${jobId}')">
                        <i class="bi bi-play me-2"></i>Rerun Job
                    </button>
                `;
            }
            
            // Cancel job (if running)
            if (jobData.status === 'running' || jobData.status === 'queued') {
                actionButtons += `
                    <button class="btn btn-outline-warning mb-2" onclick="cancelJob('${jobId}')">
                        <i class="bi bi-x-circle me-2"></i>Cancel Job
                    </button>
                `;
            }
            
            // Delete job
            actionButtons += `
                <button class="btn btn-outline-danger" onclick="deleteJob('${jobId}')">
                    <i class="bi bi-trash me-2"></i>Delete Job
                </button>
            `;
            
            actionsCard.innerHTML = `
                <div class="card-header">
                    <h5 class="card-title mb-0">Job Actions</h5>
                </div>
                <div class="card-body">
                    <div class="d-grid gap-2">
                        ${actionButtons}
                    </div>
                </div>
            `;
            
            jobMediaContainer.appendChild(actionsCard);
        } else {
            jobMediaContainer.classList.add('d-none');
        }
        
        // Start polling if job is still running
        if (jobData.status === 'running' || jobData.status === 'queued') {
            // Set up real-time updates for this job
            const jobCard = document.querySelector(`[data-job-id="${jobId}"]`);
            if (jobCard) {
                // Highlight this card
                document.querySelectorAll('.job-card').forEach(card => {
                    card.classList.remove('border-primary');
                });
                jobCard.classList.add('border-primary');
            }
            
            // Set up auto-refresh for job details
            startJobDetailPolling(jobId);
        } else {
            // Clear polling for this job
            stopJobDetailPolling();
        }
        
    } catch (error) {
        console.error('Error loading job details:', error);
        jobDetailContainer.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle me-2"></i>
                Failed to load job details: ${error.message || 'Unknown error'}
            </div>
        `;
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
    // Show confirmation dialog with checkbox for image deletion
    const confirmResult = await showConfirmDialog(
        'Delete Job',
        `Are you sure you want to delete job ${jobId}?`,
        'This will remove the job and its output videos. You can also choose to delete associated image files if they are not used by other jobs.',
        'Delete related image files if not used by other jobs'
    );
    
    if (!confirmResult.confirmed) {
        return;
    }
    
    try {
        // Make the API request to delete the job
        const response = await fetch(`/api/job/${jobId}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                delete_images: confirmResult.checkbox
            })
        });
        
        // Process the response
        let resultMessage = '';
        
        if (!response.ok) {
            // Try to get error message from response
            try {
                const errorData = await response.json();
                throw new Error(errorData.error || `Failed to delete job: ${response.status}`);
            } catch (jsonError) {
                // If can't parse JSON, use status text
                throw new Error(`Failed to delete job: ${response.statusText}`);
            }
        }
        
        // Process successful response
        try {
            const result = await response.json();
            resultMessage = `Job ${jobId} deleted successfully${result.deleted_images ? `. Deleted ${result.deleted_images} unused image files.` : ''}`;
        } catch (jsonError) {
            // If can't parse result, just use a generic success message
            resultMessage = `Job ${jobId} deleted successfully`;
        }
        
        // Show success message
        showMessage(resultMessage, 'success');
        
        // Close any open modals
        const confirmModal = document.getElementById('confirmModal');
        if (confirmModal) {
            const bsModal = bootstrap.Modal.getInstance(confirmModal);
            if (bsModal) {
                bsModal.hide();
            }
        }
        
        // If this was the active job, clear it
        if (activeJobId === jobId) {
            activeJobId = null;
            hideProgress();
        }
        
        // Refresh the UI elements
        loadJobQueue();
        loadOutputs();
        
        // Clear job details if this was the currently selected job
        const jobDetailContainer = document.getElementById('jobDetailContainer');
        const jobMediaContainer = document.getElementById('jobMediaContainer');
        
        if (jobDetailContainer) {
            jobDetailContainer.innerHTML = `
                <div class="alert alert-secondary">
                    <i class="bi bi-info-circle"></i> Select a job to view details
                </div>
            `;
        }
        
        if (jobMediaContainer) {
            jobMediaContainer.classList.add('d-none');
        }
        
    } catch (error) {
        console.error('Error deleting job:', error);
        showMessage(`Failed to delete job: ${error.message}`, 'danger');
        
        // Still refresh UI in case of partial deletion
        loadJobQueue();
        loadOutputs();
    }
}

// Cancel a job
async function cancelJob(jobId) {
    try {
        // Show confirmation dialog
        const confirmResult = await showConfirmDialog(
            'Cancel Job',
            `Are you sure you want to cancel job ${jobId}?`,
            'This will stop the job if it is currently running.'
        );
        
        if (!confirmResult.confirmed) {
            return;
        }
        
        // Make the API request to cancel the job
        const response = await fetch(`/api/job/${jobId}/cancel`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            try {
                const errorData = await response.json();
                throw new Error(errorData.error || `Failed to cancel job: ${response.status}`);
            } catch (jsonError) {
                throw new Error(`Failed to cancel job: ${response.statusText}`);
            }
        }
        
        const result = await response.json();
        
        // Show success message
        showMessage(result.message || `Job ${jobId} cancelled successfully`, 'success');
        
        // Refresh the job queue
        loadJobQueue();
        
    } catch (error) {
        console.error('Error cancelling job:', error);
        showMessage(`Failed to cancel job: ${error.message}`, 'danger');
    }
}

async function deleteOutputVideo(videoPath, videoName) {
    // Show confirmation dialog
    const confirmResult = await showConfirmDialog(
        'Delete Video',
        `Are you sure you want to delete "${videoName}"?`,
        'This will permanently remove the video file from the server.',
        'Also delete the associated job data'
    );
    
    if (!confirmResult.confirmed) {
        return;
    }
    
    try {
        // Extract job ID from filename if possible
        const jobId = videoPath.split('/').pop().split('.')[0];
        
        if (confirmResult.checkbox && jobId) {
            // Delete the job with the video
            await deleteJob(jobId);
            
            // Close any open modal
            const videoViewerModal = document.getElementById('videoViewerModal');
            if (videoViewerModal) {
                const bsModal = bootstrap.Modal.getInstance(videoViewerModal);
                if (bsModal) {
                    bsModal.hide();
                }
            }
            
            return; // Job deletion already refreshes both jobs and outputs
        }
        
        // Otherwise, just delete the video file
        const response = await fetch('/api/delete_video', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ video_path: videoPath })
        });
        
        if (!response.ok) {
            try {
                const errorData = await response.json();
                throw new Error(errorData.error || `Failed to delete video: ${response.status}`);
            } catch (jsonError) {
                throw new Error(`Failed to delete video: ${response.statusText}`);
            }
        }
        
        const result = await response.json();
        
        // Close any open modal
        const videoViewerModal = document.getElementById('videoViewerModal');
        if (videoViewerModal) {
            const bsModal = bootstrap.Modal.getInstance(videoViewerModal);
            if (bsModal) {
                bsModal.hide();
            }
        }
        
        // Show success message
        showMessage(result.message || 'Video deleted successfully', 'success');
        
        // Refresh the outputs list
        loadOutputs();
        
    } catch (error) {
        console.error('Error deleting video:', error);
        showMessage(`Failed to delete video: ${error.message}`, 'danger');
        
        // Still refresh outputs in case of partial deletion
        loadOutputs();
    }
}

// Function to start video generation
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
    fetch(`/api/job_status/${jobId}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showMessage(data.error, 'danger');
                hideProgress();
                clearInterval(pollingInterval);
                return;
            }
            
            // Update progress
            updateProgress(data.progress, data.message);
            
            // Update latent preview if available
            if (data.current_latents) {
                updateCurrentJobPreview(data.current_latents, jobId);
            }
            
            // If job is completed or failed, stop polling
            if (data.status === 'completed') {
                showSuccess(data.result_video);
                clearInterval(jobDetailsInterval);
                jobDetailsInterval = null;
                clearInterval(jobQueueInterval);
                jobQueueInterval = null;
            } else if (data.status === 'failed') {
                showMessage(`Generation failed: ${data.message}`, 'danger');
                hideProgress();
                clearInterval(jobDetailsInterval);
                jobDetailsInterval = null;
                clearInterval(jobQueueInterval);
                jobQueueInterval = null;
            }
        })
        .catch(error => {
            console.error('Error polling job status:', error);
        });

        
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