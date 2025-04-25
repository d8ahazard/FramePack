// Job Queue module for FramePack
// Handles job queue, job details, and job operations

import { 
    elements, 
    timeline,
    showMessage,
    connectJobWebsocket,
    disconnectJobWebsocket,
    addJobEventListener,
    removeJobEventListener
} from './common.js';

// Add import for openVideoViewer used in loadJobDetails
import { openVideoViewer } from './outputs.js';

// Initialize module variables
let currentJobId = null;
let jobListenerIndex = -1;

// Module initialization function
function initJobQueue() {
    console.log('Job Queue module initialized');
    
    // Add queue styles
    addQueueStyles();
    
    // Initial load of job queue
    loadJobQueue();
    
    // Setup job WebSocket listener if not already set up
    if (jobListenerIndex < 0) {
        setupJobWebSocketListener();
    }
    
    // Clean up websocket when leaving the page
    window.addEventListener('beforeunload', () => {
        if (jobListenerIndex >= 0) {
            removeJobEventListener(jobListenerIndex);
            disconnectJobWebsocket();
        }
    });
    
    // Export loadJobQueue for access from other modules
    window.loadJobQueue = loadJobQueue;
}

// Set up WebSocket listener for job events
function setupJobWebSocketListener() {
    // Connect to WebSocket for job updates if not already connected
    connectJobWebsocket();
    
    // Register event listener for job status updates
    jobListenerIndex = addJobEventListener(event => {
        // Check if this is a job status update
        if (event.type === 'job_update') {
            console.log('Job update received via WebSocket:', event);
            
            // Extract job details
            const jobId = event.job_id;
            const status = event.status;
            const progress = event.progress;
            const message = event.message;
            
            // Update the job in the queue UI
            updateJobInQueue(jobId, status, progress, message);
            
            // If this is the currently active job in the editor, update the editor UI
            if (window.currentActiveJobId === jobId) {
                // Update the editor progress display
                updateEditorProgress(jobId, status, progress, message);
            }
            
            // For completed or failed jobs, ensure we update the full job list
            if (status === 'completed' || status === 'failed') {
                // Reload the entire job queue to get the most up-to-date information
                setTimeout(() => loadJobQueue(), 500);
            }
        }
    });
}

// Update a specific job in the queue UI without reloading everything
function updateJobInQueue(jobId, status, progress, message) {
    // Find the job item in the DOM
    const jobItem = document.querySelector(`.job-item[data-job-id="${jobId}"]`);
    if (!jobItem) {
        console.log(`Job ${jobId} not found in queue UI, will be updated on next reload`);
        return;
    }
    
    // Update status badge
    const statusBadge = jobItem.querySelector('.status-badge');
    if (statusBadge) {
        // Remove all existing badge classes
        statusBadge.classList.remove('bg-secondary', 'bg-success', 'bg-danger', 'bg-primary', 'bg-warning', 'bg-info');
        
        // Add the appropriate badge class based on status
        if (status === 'completed') {
            statusBadge.classList.add('bg-success');
            statusBadge.textContent = 'Completed';
        } else if (status === 'failed') {
            statusBadge.classList.add('bg-danger');
            statusBadge.textContent = 'Failed';
        } else if (status === 'running') {
            statusBadge.classList.add('bg-primary');
            statusBadge.textContent = 'Running';
        } else if (status === 'queued') {
            statusBadge.classList.add('bg-warning');
            statusBadge.textContent = 'Queued';
        } else if (status === 'cancelled') {
            statusBadge.classList.add('bg-secondary');
            statusBadge.textContent = 'Cancelled';
        } else if (status === 'saved') {
            statusBadge.classList.add('bg-info');
            statusBadge.textContent = 'Saved';
        }
    }
    
    // Update progress information
    const progressBar = jobItem.querySelector('.progress-bar');
    if (progressBar) {
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
        progressBar.textContent = `${progress}%`;
    }
    
    // Update status message
    const statusMessage = jobItem.querySelector('.job-message');
    if (statusMessage) {
        statusMessage.textContent = message || '';
    }
    
    // Update the overall job item class to reflect the new status
    jobItem.className = `job-item ${status || 'unknown'}`;
    
    console.log(`Updated job ${jobId} in queue UI: status=${status}, progress=${progress}`);
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
        
        // Sort jobs: running first, then queued by position, then saved, then completed/failed/cancelled by timestamp (newest first)
        jobs.sort((a, b) => {
            // First, prioritize by status
            const statusOrder = { 'running': 0, 'queued': 1, 'saved': 2, 'completed': 3, 'failed': 4, 'cancelled': 5 };
            const statusDiff = (statusOrder[a.status] || 99) - (statusOrder[b.status] || 99);
            
            if (statusDiff !== 0) return statusDiff;
            
            // For queued jobs, sort by queue_position
            if (a.status === 'queued' && b.status === 'queued') {
                return a.queue_position - b.queue_position;
            }
            
            // For others, sort by timestamp (extracted from job_id or created_timestamp if available)
            const aTime = a.created_timestamp || parseInt(a.job_id.split('-')[0]);
            const bTime = b.created_timestamp || parseInt(b.job_id.split('-')[0]);
            return bTime - aTime; // Newest first
        });
        
        // Group jobs by status
        const runningJobs = jobs.filter(job => job.status === 'running');
        const queuedJobs = jobs.filter(job => job.status === 'queued');
        const savedJobs = jobs.filter(job => job.status === 'saved');
        const completedJobs = jobs.filter(job => job.status === 'completed');
        const failedJobs = jobs.filter(job => job.status === 'failed');
        const cancelledJobs = jobs.filter(job => job.status === 'cancelled');
        
        // Create section for running jobs
        if (runningJobs.length > 0) {
            const runningSection = document.createElement('div');
            runningSection.className = 'mb-3';
            runningSection.innerHTML = `<h6 class="fw-bold">Running</h6>`;
            
            runningJobs.forEach(job => {
                const jobItem = createJobItem(job);
                runningSection.appendChild(jobItem);
            });
            
            jobsContainer.appendChild(runningSection);
        }
        
        // Create section for queued jobs with drag-and-drop support
        if (queuedJobs.length > 0) {
            const queuedSection = document.createElement('div');
            queuedSection.className = 'mb-3';
            queuedSection.innerHTML = `
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <h6 class="fw-bold mb-0">Queued</h6>
                    <button id="runAllQueued" class="btn btn-sm btn-primary">
                        <i class="bi bi-play-fill"></i> Run All
                    </button>
                </div>
            `;
            
            const queueList = document.createElement('div');
            queueList.id = 'queuedJobsList';
            queueList.className = 'queue-list';
            
            // Make the queue list sortable
            queuedJobs.forEach(job => {
                const jobItem = createJobItem(job);
                jobItem.classList.add('draggable');
                jobItem.dataset.queuePosition = job.queue_position;
                queueList.appendChild(jobItem);
            });
            
            queuedSection.appendChild(queueList);
            jobsContainer.appendChild(queuedSection);
            
            // Initialize drag and drop functionality
            initDragAndDrop();
            
            // Run all button event listener
            document.getElementById('runAllQueued').addEventListener('click', () => {
                if (confirm(`Run all ${queuedJobs.length} queued jobs?`)) {
                    processQueue();
                }
            });
        }
        
        // Other job sections (completed, failed, cancelled)
        if (savedJobs.length > 0) {
            const savedSection = document.createElement('div');
            savedSection.className = 'mb-3';
            savedSection.innerHTML = `<h6 class="fw-bold">Saved</h6>`;
            
            savedJobs.forEach(job => {
                const jobItem = createJobItem(job);
                savedSection.appendChild(jobItem);
            });
            
            jobsContainer.appendChild(savedSection);
        }
        
        if (completedJobs.length > 0) {
            const completedSection = document.createElement('div');
            completedSection.className = 'mb-3';
            completedSection.innerHTML = `<h6 class="fw-bold">Completed</h6>`;
            
            completedJobs.forEach(job => {
                const jobItem = createJobItem(job);
                completedSection.appendChild(jobItem);
            });
            
            jobsContainer.appendChild(completedSection);
        }
        
        if (failedJobs.length > 0 || cancelledJobs.length > 0) {
            const failedSection = document.createElement('div');
            failedSection.className = 'mb-3';
            failedSection.innerHTML = `<h6 class="fw-bold">Failed/Cancelled</h6>`;
            
            [...failedJobs, ...cancelledJobs].forEach(job => {
                const jobItem = createJobItem(job);
                failedSection.appendChild(jobItem);
            });
            
            jobsContainer.appendChild(failedSection);
        }
        
        // Count completed jobs and add clear button at the bottom if needed
        if (completedJobs.length > 0) {
            const clearCompletedContainer = document.createElement('div');
            clearCompletedContainer.className = 'mt-3 text-center';
            
            const clearCompletedBtn = document.createElement('button');
            clearCompletedBtn.className = 'btn btn-sm btn-outline-danger';
            clearCompletedBtn.innerHTML = '<i class="bi bi-trash"></i> Clear Completed Jobs';
            clearCompletedBtn.onclick = () => {
                if (confirm(`Are you sure you want to delete all ${completedJobs.length} completed jobs?`)) {
                    clearCompletedJobs();
                }
            };
            
            clearCompletedContainer.appendChild(clearCompletedBtn);
            jobsContainer.appendChild(clearCompletedContainer);
        }
        
        // Cancel all button if there are running or queued jobs
        if (runningJobs.length > 0 || queuedJobs.length > 0) {
            const cancelAllContainer = document.createElement('div');
            cancelAllContainer.className = 'mt-3 text-center';
            
            const cancelAllBtn = document.createElement('button');
            cancelAllBtn.className = 'btn btn-sm btn-danger';
            cancelAllBtn.innerHTML = '<i class="bi bi-x-circle"></i> Cancel All Jobs';
            cancelAllBtn.onclick = () => {
                if (confirm(`Are you sure you want to cancel all running and queued jobs?`)) {
                    cancelAllJobs();
                }
            };
            
            cancelAllContainer.appendChild(cancelAllBtn);
            jobsContainer.appendChild(cancelAllContainer);
        }
        
    } catch (error) {
        console.error('Error loading job queue:', error);
        document.getElementById('jobsContainer').innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i> Error loading jobs: ${error.message}
            </div>
        `;
    }
}

// Helper function to create job items
function createJobItem(job) {
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
    } else if (job.status === 'cancelled') {
        statusBadgeClass = 'bg-secondary';
    } else if (job.status === 'saved') {
        statusBadgeClass = 'bg-info';
    }
    
    // Job title/name display
    const jobTitle = formatJobTimestamp(job.job_id);
    
    // Check if job has missing images
    const hasInvalidImages = job.is_valid === false;
    const invalidBadge = hasInvalidImages ? 
        `<span class="badge bg-danger ms-2">Missing Images</span>` : '';
    
    // Queue position badge
    const queuePositionBadge = job.status === 'queued' ? 
        `<span class="badge bg-dark ms-2">#${job.queue_position}</span>` : '';
    
    jobItem.innerHTML = `
        <div class="d-flex justify-content-between align-items-center">
            <div>
                <div class="fw-bold">ID: ${job.job_id}</div>
            </div>
            <div>
                <span class="badge ${statusBadgeClass}">${job.status}</span>
                ${queuePositionBadge}
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
    
    return jobItem;
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
        loadJobQueue().then(r => {}).catch(e => {
            console.error('Error refreshing job queue:', e);
            showMessage(`Failed to refresh job queue: ${e.message}`, 'error');
        });
    }
}

// Initialize drag and drop functionality for queue reordering
function initDragAndDrop() {
    const queueList = document.getElementById('queuedJobsList');
    if (!queueList) return;
    
    let draggedItem = null;
    
    // Add drag event listeners to all draggable items
    document.querySelectorAll('.draggable').forEach(item => {
        item.setAttribute('draggable', 'true');
        
        item.addEventListener('dragstart', function() {
            draggedItem = this;
            setTimeout(() => this.classList.add('dragging'), 0);
        });
        
        item.addEventListener('dragend', function() {
            this.classList.remove('dragging');
            draggedItem = null;
            
            // Update queue order on server
            updateQueueOrder();
        });
        
        item.addEventListener('dragover', function(e) {
            e.preventDefault();
            if (draggedItem === this) return;
            
            const bbox = this.getBoundingClientRect();
            const midY = bbox.y + bbox.height / 2;
            const isAfter = e.clientY > midY;
            
            if (isAfter) {
                queueList.insertBefore(draggedItem, this.nextSibling);
            } else {
                queueList.insertBefore(draggedItem, this);
            }
        });
    });
}

// Update the queue order on the server
async function updateQueueOrder() {
    const queueList = document.getElementById('queuedJobsList');
    if (!queueList) return;
    
    // Get job IDs in current order
    const jobIds = Array.from(queueList.querySelectorAll('.job-item')).map(item => item.dataset.jobId);
    
    try {
        const response = await fetch('/api/update_queue_order', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(jobIds)
        });
        
        if (!response.ok) {
            throw new Error('Failed to update queue order');
        }
        
        // Refresh the job queue after a short delay
        setTimeout(() => {
            loadJobQueue();
        }, 500);
        
    } catch (error) {
        console.error('Error updating queue order:', error);
        showMessage('Failed to update queue order: ' + error.message, 'error');
    }
}

// Process all queued jobs
async function processQueue() {
    try {
        // Show loading message
        showMessage('Starting to process queued jobs...', 'info');
        
        // Refresh the queue to show the changes
        loadJobQueue().then(r => {}).catch(e => {
            console.error('Error refreshing job queue:', e);
            showMessage('Failed to refresh job queue: ' + e.message, 'error');
        });
        
    } catch (error) {
        console.error('Error processing queue:', error);
        showMessage('Error processing queue: ' + error.message, 'error');
    }
}

// Cancel all jobs
async function cancelAllJobs() {
    try {
        const response = await fetch('/api/cancel_all_jobs', {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error('Failed to cancel jobs');
        }
        
        const result = await response.json();
        
        // Show success message
        showMessage(result.message, 'success');
        
        // Refresh the job queue
        loadJobQueue().then(r => {}).catch(e => {
            console.error('Error refreshing job queue:', e);
            showMessage('Failed to refresh job queue: ' + e.message, 'error');
        });
        
    } catch (error) {
        console.error('Error cancelling all jobs:', error);
        showMessage('Error cancelling jobs: ' + error.message, 'error');
    }
}

// Load job details
async function loadJobDetails(jobId) {
    try {
        // Disconnect from previous job's websocket if any
        if (currentJobId && currentJobId !== jobId) {
            disconnectJobWebsocket();
            if (jobListenerIndex >= 0) {
                removeJobEventListener(jobListenerIndex);
                jobListenerIndex = -1;
            }
        }
        
        // Set current job ID
        currentJobId = jobId;
        
        const response = await fetch(`/api/job_status/${jobId}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch job details: ${response.statusText}`);
        }
        
        const jobData = await response.json();
        
        // Display job details
        displayJobDetails(jobData);
        
        // If job is active, connect to websocket for live updates
        if (jobData.status === 'running' || jobData.status === 'queued') {
            // Connect to websocket for live updates
            connectJobWebsocket(jobId);
            
            // Register listener for updates
            jobListenerIndex = addJobEventListener((data) => {
                if (data.job_id === jobId) {
                    // Update the UI with new data
                    displayJobDetails(data);
                }
            });
        }
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

// Display job details - separated from loadJobDetails for websocket updates
function displayJobDetails(jobData) {
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
    } else if (jobData.status === 'cancelled') {
        statusBadgeClass = 'bg-secondary';
    } else if (jobData.status === 'saved') {
        statusBadgeClass = 'bg-info';
    }
    
    // Check if job has missing images
    const hasInvalidImages = jobData.is_valid === false;
    
    // Invalid images warning
    const invalidImagesWarning = hasInvalidImages ? 
        `<div class="alert alert-warning">
            <i class="bi bi-exclamation-triangle me-2"></i>
            This job is missing ${jobData.missing_images.length} image(s). You can reload 
            the job to timeline but must fix the missing images before rerunning.
        </div>` : '';
    
    // Queue position info
    const queueInfo = jobData.status === 'queued' ? 
        `<p><strong>Queue Position:</strong> ${jobData.queue_position}</p>` : '';
    
    jobInfoCard.innerHTML = `
        <div class="card-header d-flex justify-content-between align-items-center">
            <h5 class="card-title mb-0">Job Information</h5>
            <span class="badge ${statusBadgeClass}">${jobData.status}</span>
        </div>
        <div class="card-body">
            ${invalidImagesWarning}
            <p><strong>Status:</strong> ${jobData.status.charAt(0).toUpperCase() + jobData.status.slice(1)}</p>
            ${queueInfo}
            <p><strong>Progress:</strong> ${jobData.progress}%</p>
            <p><strong>Message:</strong> ${jobData.message || 'No message'}</p>
            <p><strong>Created:</strong> ${formatJobTimestamp(jobData.job_id)}</p>
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
        
        // Determine which image to show - use the latest segment to avoid duplicates
        let latentImageSrc = '';
        if (jobData.current_latents) {
            latentImageSrc = jobData.current_latents;
        } else if (jobData.segments && jobData.segments.length > 0) {
            // Use the latest segment to avoid showing duplicates
            latentImageSrc = jobData.segments[jobData.segments.length - 1];
        }
        
        latentsContainer.innerHTML = `
            <div class="card mb-3">
                <div class="card-header">
                    <h5 class="card-title mb-0 fs-6">Current Latents</h5>
                </div>
                <div class="card-body text-center">
                    ${latentImageSrc ? 
                        `<img id="jobCurrentLatents" src="${latentImageSrc}" class="img-fluid rounded" alt="Current latents" title="Latent representation (864×64)">` :
                        `<div class="text-muted py-5"><i class="bi bi-image me-2"></i>Latent image not available yet</div>`
                    }
                </div>
                ${latentImageSrc ? `<div class="card-footer p-2 text-center">
                    <small class="text-muted">Latent dimensions: 864×64 pixels</small>
                </div>` : ''}
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
        cancelBtn.onclick = () => cancelJob(jobData.job_id);
        actionContainer.appendChild(cancelBtn);
    }
    
    // Queue button for completed, failed, or cancelled jobs
    if ((jobData.status === 'completed' || jobData.status === 'failed' || 
         jobData.status === 'cancelled' || jobData.status === 'saved') && 
        jobData.job_settings && jobData.is_valid !== false) {
        const queueBtn = document.createElement('button');
        queueBtn.className = 'btn btn-warning';
        queueBtn.innerHTML = '<i class="bi bi-hourglass"></i> Queue Job';
        queueBtn.onclick = () => requeueJob(jobData.job_id);
        actionContainer.appendChild(queueBtn);
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
        reloadBtn.onclick = () => loadJobToTimeline(jobData.job_id);
        actionContainer.appendChild(reloadBtn);
    }
    
    // Rerun button for completed or failed jobs (only if images are valid)
    if ((jobData.status === 'completed' || jobData.status === 'failed' || 
         jobData.status === 'cancelled' || jobData.status === 'saved') && 
        jobData.job_settings && jobData.is_valid !== false) {
        const rerunBtn = document.createElement('button');
        rerunBtn.className = 'btn btn-outline-success';
        rerunBtn.innerHTML = '<i class="bi bi-arrow-repeat"></i> Rerun Job';
        rerunBtn.onclick = () => rerunJob(jobData.job_id);
        actionContainer.appendChild(rerunBtn);
    }
    
    // Delete button for any job
    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'btn btn-outline-danger';
    deleteBtn.innerHTML = '<i class="bi bi-trash"></i> Delete Job';
    deleteBtn.onclick = () => {
        if (confirm(`Are you sure you want to delete this job? This will remove all files related to ${jobData.job_id}.`)) {
            deleteJob(jobData.job_id).then(r => {}).catch(e => {
                console.error('Error deleting job:', e);
                showMessage(`Failed to delete job: ${e.message}`, 'error');

            });
        }
    };
    actionContainer.appendChild(deleteBtn);
    
    // Add the action container
    jobDetailContainer.appendChild(actionContainer);
}

// Delete a job
async function deleteJob(jobId) {
    // Cancel the job first?
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
            loadJobQueue().then(r => {}).catch(e => {
                console.error('Error refreshing job queue:', e);
                showMessage(`Failed to refresh job queue: ${e.message}`, 'error');

            });
            
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
        console.log("Loading JobData:", jobData);

                // Get the job settings - just use whatever structure is there
                const settings = (jobData.job_settings && typeof jobData.job_settings === 'object') ? 
                    (jobData.job_settings.framepack || jobData.job_settings) : 
                    {};
                console.log("Got job settings:", settings);
                
                // Use segments from the framepack settings first, then job data, or empty array as fallback
                const segments = settings.segments || jobData.segments || [];
                console.log("Got segments from settings:", segments);
        
        // Step 1: Clear the timeline
        elements.timelineContainer.innerHTML = '';
        timeline.length = 0; // Clear array while keeping reference
        
        // Step 2: Set form values from settings
        if (elements.globalPrompt) {
            elements.globalPrompt.value = settings.global_prompt || '';
        }
        
        if (elements.negativePrompt) {
            elements.negativePrompt.value = settings.negative_prompt || '';
        }
        
        if (elements.steps) {
            elements.steps.value = settings.steps || 25;
        }
        
        if (elements.guidanceScale) {
            elements.guidanceScale.value = settings.guidance_scale || 10.0;
        }
        
        if (elements.resolution) {
            elements.resolution.value = settings.resolution || 640;
        }
        
        if (elements.useTeacache) {
            elements.useTeacache.checked = settings.use_teacache !== false;
        }
        
        if (elements.enableAdaptiveMemory) {
            elements.enableAdaptiveMemory.checked = settings.enable_adaptive_memory !== false;
        }
        
        // Step 3: Add segments to timeline
        const editorModule = await import('./editor.js');
        
        // Maps for segment configurations from settings if they exist
        const segmentConfigs = new Map();
        
        // Try to get segment configurations if they exist
        if (settings.segments && Array.isArray(settings.segments)) {
            console.log(`Processing ${settings.segments.length} segments from job settings`);
            settings.segments.forEach((seg, index) => {
                if (seg && seg.image_path) {
                    segmentConfigs.set(seg.image_path, {
                        prompt: seg.prompt || '',
                        duration: seg.duration || 3.0
                    });
                    console.log(`Segment ${index + 1}: ${seg.image_path} (prompt: "${seg.prompt?.substring(0, 30)}${seg.prompt?.length > 30 ? '...' : ''}", duration: ${seg.duration}s)`);
                } else {
                    console.warn(`Segment ${index + 1} has invalid or missing image_path:`, seg);
                }
            });
        } else {
            console.warn("No segments array found in job settings or invalid format:", settings);
        }
        
        // Add each segment to the timeline - use whatever we have
        console.log(`Adding ${segments.length} segments to timeline`);
        let successCount = 0;
        
        for (const segment of segments) {
            try {
                // Handle both string segments and object segments
                const segmentPath = typeof segment === 'object' ? segment.image_path : segment;
                
                if (!segmentPath) {
                    console.warn("Invalid segment (missing path):", segment);
                    continue;
                }
                
                // Get config for this segment if it exists
                const config = segmentConfigs.get(segmentPath) || {};
                
                // Log segment being processed
                console.log(`Processing segment: ${segmentPath}`);
            
            // Create a file object representation with proper path handling for browser security
            // Use the serve_file API endpoint to load the image instead of direct file path
            const fileName = segmentPath.split('\\').pop().split('/').pop(); // Extract filename (works for both / and \ paths)
            
            // Create URL to load the image through server API instead of direct file path
            const imageUrl = `/api/serve_file?path=${encodeURIComponent(segmentPath)}`;
            
            const fileObj = {
                serverPath: segmentPath,
                name: fileName,
                src: imageUrl, // Use server API to serve the file
                prompt: config.prompt || '',
                duration: config.duration || 3.0
            };
            
            console.log(`Adding segment to timeline: ${fileName} from ${imageUrl}`);
            
                // Add to timeline
                await editorModule.addItemToTimeline(fileObj);
                successCount++;
            } catch (error) {
                console.error(`Error adding segment to timeline:`, error);
                // Continue with next segment
            }
        }
        
        // Step 4: Update UI and finish
        editorModule.updateTimelineStatus();

        // Switch to editor tab
        const editorTab = document.getElementById('editor-tab');
        if (editorTab) {
            bootstrap.Tab.getOrCreateInstance(editorTab).show();
        }

        // Remove loading message
        loadingMessage.remove();

        // Provide feedback based on how many segments were loaded
        if (successCount === 0 && segments.length > 0) {
            showMessage(`Failed to load any segments to the timeline. Check console for details.`, 'error');
        } else if (successCount < segments.length) {
            showMessage(`Loaded ${successCount} of ${segments.length} segments to the timeline.`, 'warning');
        } else if (segments.length > 0) {
            showMessage(`Successfully loaded ${segments.length} segments to the timeline.`, 'success');
        } else {
            showMessage(`Job loaded but no segments were found.`, 'warning');
        }
        
        // Show warning for missing images
        if (jobData.is_valid === false && jobData.missing_images && jobData.missing_images.length > 0) {
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

        // Get current job data first to check if we need to update the structure
        const jobResponse = await fetch(`/api/job/${jobId}`);
        if (jobResponse.ok) {
            const jobData = await jobResponse.json();

            // Check if we need to update the job to use ModuleJobSettings format
            if (jobData.job_settings && !jobData.job_settings.module_type && !jobData.job_settings.settings) {
                console.log('Converting job to ModuleJobSettings format before rerunning');

                // Update the job to use ModuleJobSettings format
                await fetch(`/api/save_job/${jobId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        ...jobData,
                        job_settings: {
                            module_type: "framepack",
                            settings: jobData.job_settings
                        }
                    })
                });
            }
        }

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

// Queue a job (without running it immediately)
async function requeueJob(jobId) {
    try {
        // First get the current job data
        const jobResponse = await fetch(`/api/job/${jobId}`);
        if (!jobResponse.ok) {
            throw new Error('Failed to fetch job data');
        }

        const jobData = await jobResponse.json();

        // Ensure job settings has the proper ModuleJobSettings structure before queuing
        if (jobData.job_settings && !jobData.job_settings.module_type && !jobData.job_settings.settings) {
            // Update job with proper ModuleJobSettings structure
            const updateResponse = await fetch(`/api/save_job/${jobId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    ...jobData,
                    job_settings: {
                        module_type: "framepack",
                        settings: jobData.job_settings
                    }
                })
            });

            if (!updateResponse.ok) {
                console.warn('Failed to update job to new ModuleJobSettings format');
            }
        }

        // Queue the job
        const response = await fetch(`/api/requeue_job/${jobId}`, {
            method: 'POST'
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to queue job');
        }

        const result = await response.json();

        if (result.success) {
            // Refresh the job queue
            loadJobQueue();

            // Show success message
            showMessage(`Job queued at position ${result.queue_position}`, 'success');

            // Load details for the queued job
            loadJobDetails(jobId).then(r => {}).catch(e => {
                console.error('Error loading job details:', e);
                showMessage(`Failed to load job details: ${e.message}`, 'error');
            });
        } else {
            throw new Error(result.error || 'Failed to queue job');
        }

    } catch (error) {
        console.error('Error queueing job:', error);
        showMessage(`Failed to queue job: ${error.message}`, 'error');
    }
}

// Add CSS for draggable queue items
function addQueueStyles() {
    // Check if style is already added
    if (document.getElementById('queueStyles')) return;

    const style = document.createElement('style');
    style.id = 'queueStyles';
    style.textContent = `
        .queue-list {
            border: 1px solid #f0f0f0;
            border-radius: 6px;
            padding: 5px;
            background: #f9f9f9;
        }
        .draggable {
            cursor: grab;
            position: relative;
        }
        .draggable::before {
            content: ":::";
            position: absolute;
            left: 8px;
            top: 50%;
            transform: translateY(-50%);
            color: #aaa;
            font-weight: bold;
            letter-spacing: 1px;
        }
        .draggable .d-flex {
            padding-left: 20px;
        }
        .draggable.dragging {
            opacity: 0.5;
        }
    `;
    document.head.appendChild(style);
}

// Export module functions
export {
    initJobQueue,
    loadJobQueue,
    loadJobDetails,
    loadJobToTimeline,
    rerunJob,
    formatJobTimestamp,
    clearCompletedJobs,
    deleteJob,
    cancelJob,
    displayJobDetails
};
