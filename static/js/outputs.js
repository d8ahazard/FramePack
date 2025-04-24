// Outputs module for FramePack
// Handles output videos listing and playback

import { 
    elements, 
    formatTimestamp, 
    showMessage 
} from './common.js';

// Initialize module variables
let videoViewerModal = null;

// Module initialization function
function initOutputs() {
    console.log('Outputs module initialized');
    
    // Initialize modals
    const videoViewerModalElement = document.getElementById('videoViewerModal');
    if (videoViewerModalElement) {
        videoViewerModal = new bootstrap.Modal(videoViewerModalElement);
    }
    
    // Add event listeners
    if (elements.refreshOutputsBtn) {
        elements.refreshOutputsBtn.addEventListener('click', loadOutputs);
    }
    
    // Initial load of outputs
    loadOutputs();
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

// Export module functions
export {
    initOutputs,
    loadOutputs,
    openVideoViewer
};
