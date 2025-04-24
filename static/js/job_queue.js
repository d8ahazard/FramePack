// Job Queue module for FramePack
// Handles job queue, job details, and job operations

import { 
    elements, 
    jobQueueElements, 
    timeline,
    formatTimestamp, 
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
    
    // Remove refresh button event listener since we removed the button
    // and now have websocket support
    
    // Initial load of job queue
    loadJobQueue();
    
    // Clean up websocket when leaving the page
    window.addEventListener('beforeunload', () => {
        if (jobListenerIndex >= 0) {
            removeJobEventListener(jobListenerIndex);
            disconnectJobWebsocket();
        }
    });
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
        
        // Count completed jobs and add clear button at the bottom if needed
        const completedJobs = jobs.filter(job => job.status === 'completed');
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
        
    } catch (error) {
        console.error('Error loading job queue:', error);
        document.getElementById('jobsContainer').innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle"></i> Error loading jobs: ${error.message}
            </div>
        `;
    }
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
        loadJobQueue();
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
            <p><strong>Job ID:</strong> ${jobData.job_id}</p>
            ${jobName}
            <p><strong>Status:</strong> ${jobData.status.charAt(0).toUpperCase() + jobData.status.slice(1)}</p>
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
    if ((jobData.status === 'completed' || jobData.status === 'failed') && 
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
            deleteJob(jobData.job_id);
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
        timeline.length = 0; // Clear array while keeping reference
        
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
        
        // Import addItemToTimeline from the editor module
        // Since this creates a circular dependency, we'll use a dynamic import
        const editorModule = await import('./editor.js');
        
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
            await editorModule.addItemToTimeline(fileObj);
        }
        
        // Update timeline status
        editorModule.updateTimelineStatus();
        
        // Switch to editor tab
        const editorTab = document.getElementById('editor-tab');
        if (editorTab) {
            bootstrap.Tab.getOrCreateInstance(editorTab).show();
        }
        
        // Remove loading message
        loadingMessage.remove();
        
        // Navigate to the editor tab
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
