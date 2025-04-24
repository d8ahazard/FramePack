// Editor module for FramePack
// Handles timeline, file uploads, and video generation

import { 
    timeline, 
    elements, 
    currentEditIndex, 
    keepCurrentImage, 
    showMessage, 
    enforceHorizontalLayout, 
    checkImageExists 
} from './common.js';

import {
    handleFileSelect,
    handleDragOver,
    handleDragLeave,
    handleFileDrop,
    triggerFileInput,
    processSelectedFiles,
    uploadFileToServer,
    showUploadModal,
    selectedFiles,
    clearSelectedFiles
} from './files.js';

import {
    loadJobQueue
} from './job_queue.js';

// Initialize module variables
let uploadModal = null;
let editItemModal = null;
let dragSrcEl = null;

// Module initialization function
function initEditor() {
    console.log('Editor module initialized');
    
    // Initialize modals
    const uploadModalElement = document.getElementById('uploadImagesModal');
    if (uploadModalElement) {
        uploadModal = new bootstrap.Modal(uploadModalElement);
    }
    
    const frameEditModalElement = document.getElementById('frameEditModal');
    if (frameEditModalElement) {
        editItemModal = new bootstrap.Modal(frameEditModalElement);
    }
    
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
}

// Initialize event listeners for the editor module
function initEventListeners() {
    // Button click events
    if (elements.uploadImagesBtn) {
        console.log('uploadImagesBtn clicked');
        elements.uploadImagesBtn.addEventListener('click', showUploadModal);
    }
    
    if (elements.generateVideoBtn) {
        console.log('generateVideoBtn clicked');
        elements.generateVideoBtn.addEventListener('click', startGeneration);
    }
    
    // File input change event
    if (elements.fileInput) {
        console.log('fileInput change event');
        elements.fileInput.addEventListener('change', handleFileSelect);
        // Add click event listener to prevent propagation
        elements.fileInput.addEventListener('click', (e) => {
            e.stopPropagation();
        });
    }
    
    // Upload drop area events
    if (elements.uploadDropArea) {
        console.log('uploadDropArea events');
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
}

// Show upload modal

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
            // Fix typeError: assignment to constant.
            clearSelectedFiles(); // Clear any previously selected files
            processSelectedFiles(e.dataTransfer.files);
            
            // Show the upload modal to allow user to confirm/adjust before adding to timeline
            if (uploadModal) {
                uploadModal.show();
            }
        }
    });
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
                    clearSelectedFiles();
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

// Function to sort timeline - placeholder to be defined later
function sortTimeline(direction) {
    console.log('sortTimeline will be implemented');
}

// Function to start generation
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
        const currentJobId = data.job_id;
        
        // Clear the timeline after successfully starting a job
        // Clear UI
        elements.timelineContainer.innerHTML = '';
        
        // Clear timeline array
        timeline.length = 0;
        
        // Update status
        updateTimelineStatus();
        
        // Refresh the job queue
        loadJobQueue();
        // Show message
        //showMessage('Job started successfully and timeline cleared for new work', 'success');
        
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
                
                // Add a title to help users understand this is a latent representation
                currentJobImage.title = "Latent representation (864×64)";
                
                // Set up click handler to go to job queue tab and select this job
                currentJobThumbnail.onclick = () => {
                    // Switch to the job queue tab
                    const queueTab = document.getElementById('queue-tab');
                    bootstrap.Tab.getOrCreateInstance(queueTab).show();
                    
                    // Load this job in the job queue tab
                    // Need to use a dynamic import to avoid circular dependencies
                    import('./job_queue.js').then(jobQueueModule => {
                        jobQueueModule.loadJobDetails(jobId);
                        
                        // Highlight this job in the list
                        const jobItems = document.querySelectorAll('.job-item');
                        jobItems.forEach(item => {
                            item.classList.remove('active');
                            if (item.dataset.jobId === jobId) {
                                item.classList.add('active');
                            }
                        });
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
                                    <button type="button" class="btn btn-outline-primary" onclick="window.openVideoViewerFn('${data.result_video}', '${data.result_video.split('/').pop()}')">
                                        <i class="bi bi-fullscreen"></i> Fullscreen
                                    </button>
                                </div>
                            </div>
                        `;
                        progressContainer.appendChild(resultContainer);
                        
                        // Define the openVideoViewer function globally for the button to use
                        import('./outputs.js').then(outputsModule => {
                            window.openVideoViewerFn = outputsModule.openVideoViewer;
                        });
                    }
                    
                    // Success message and offer to clear timeline
                    showMessage('Video generation completed successfully!', 'success');
                    
                    // Clear the timeline after successful generation
                    if (confirm('Generation completed! Would you like to clear the timeline for a new project?')) {
                        // Clear timeline
                        elements.timelineContainer.innerHTML = '';
                        timeline.length = 0;
                        updateTimelineStatus();
                    }
                    
                } else {
                    progressStatus.textContent = `Failed: ${data.message}`;
                    showMessage(`Video generation failed: ${data.message}`, 'danger');
                }
                
                // Re-enable the generate button
                generateBtn.disabled = false;
                
                // Load the updated job list by importing the job queue module
                import('./job_queue.js').then(jobQueueModule => {
                    jobQueueModule.loadJobQueue();
                });
                
                // Load the updated outputs by importing the outputs module
                import('./outputs.js').then(outputsModule => {
                    outputsModule.loadOutputs();
                });
                
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

// Function to clear the timeline
function clearTimeline() {
    if (timeline.length === 0) {
        return; // Nothing to clear
    }
    
    if (confirm('Are you sure you want to clear the timeline? This will remove all images.')) {
        // Clear UI
        elements.timelineContainer.innerHTML = '';
        
        // Clear timeline array
        timeline.length = 0; // Clear array while keeping reference
        
        // Update status
        updateTimelineStatus();
        
        // Show confirmation message
        showMessage('Timeline cleared', 'info');
    }
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
            clearSelectedFiles();
            
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

// Function to add an item to the timeline
async function addItemToTimeline(fileObj) {
    // Remove the main dropzone if this is the first item
    const mainDropZone = elements.timelineContainer.querySelector('.timeline-item');
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
    
    return timelineItem;
}

// Timeline drag and drop event handlers
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
    timeline.length = 0;
    newTimeline.forEach(item => timeline.push(item));
    
    // Log for debugging
    console.log('Updated timeline array:', timeline.map(item => ({
        name: item.file?.name,
        serverPath: item.serverPath,
        duration: item.duration
    })));
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

// Update exported functions
export {
    initEditor,
    updateTimelineStatus,
    clearTimeline,
    handleAddToTimeline,
    addItemToTimeline,
    updateTimelineArray,
    showFrameEditModal,
    saveFrameChanges,
    deleteCurrentFrame,
    startGeneration,
    pollJobStatus,
    uploadModal
};
