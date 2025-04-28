// Editor module for FramePack
// Handles timeline, file uploads, and video generation

import { 
    timeline, 
    elements, 
    showMessage, 
    enforceHorizontalLayout, 
    checkImageExists,
    connectJobWebsocket} from './common.js';

import {
    loadJobQueue,
    runJob
} from './job_queue.js';

// Track current active job ID in the editor
window.currentActiveJobId = null;
let editorJobListenerIndex = -1;

let keepCurrentImage = false;
let currentEditIndex = -1;

// Initialize module variables
let editItemModal = null;
let dragSrcEl = null;
// Don't import from editor.js to prevent circular dependency
let uploadModal = null;

// Initialize module variables
let selectedFiles = [];

// Getter function for selectedFiles to always return the current value
function getSelectedFiles() {
    return selectedFiles;
}

// Function to check if there are selected files
function hasSelectedFiles() {
    return selectedFiles && selectedFiles.length > 0;
}

// Function to safely close the upload modal
function closeUploadModal() {
    if (uploadModal) {
        uploadModal.hide();
        return true;
    }
    return false;
}

// Function to clear the selectedFiles array
function clearSelectedFiles() {
    selectedFiles.length = 0;
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

// Function to handle file selection from input
function handleFileSelect(e) {
    const files = e.target.files;
    processSelectedFiles(files);
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
    filesArray.forEach((file, index) => {
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
            
            // Create thumbnail (don't apply special styling yet)
            const thumbnailHtml = `
                <div class="col-4 col-md-3 mb-3" data-file-index="${selectedFiles.length - 1}">
                    <div class="card">
                        <img src="${imgSrc}" class="card-img-top" alt="${fileName}">
                        <div class="card-body p-2">
                            <p class="card-text small text-muted text-truncate">${fileName}</p>
                            <div class="input-group input-group-sm duration-input-container">
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
            const newInput = elements.imageUploadContainer.querySelector(`[data-file-index="${selectedFiles.length - 1}"] .image-duration`);
            if (newInput) {
                newInput.addEventListener('change', (e) => {
                    const fileIndex = parseInt(e.target.closest('[data-file-index]').getAttribute('data-file-index'));
                    if (fileIndex >= 0 && fileIndex < selectedFiles.length) {
                        selectedFiles[fileIndex].duration = parseFloat(e.target.value);
                    }
                });
            }
            
            console.log(`Processed file: ${fileName}, total files: ${selectedFiles.length}`);
            
            // Check if this was the last file of the batch we're processing
            const isLastBatchFile = index === filesArray.length - 1;
            
            // If this was the last file in the batch, update all thumbnails
            if (isLastBatchFile) {
                setTimeout(updateLastFileDisplay, 0);
            }
        };
        
        reader.readAsDataURL(file);
    });
    
    // The upload modal will be shown by the caller
}

// Function to update the display for the last file
function updateLastFileDisplay() {
    const allThumbnails = elements.imageUploadContainer.querySelectorAll('[data-file-index]');
    if (allThumbnails.length === 0) return;
    
    // Reset all thumbnails first
    allThumbnails.forEach(thumbnail => {
        const durationContainer = thumbnail.querySelector('.duration-input-container');
        const warningMsg = thumbnail.querySelector('.last-frame-warning');
        
        // Remove any existing warning messages
        if (warningMsg) {
            warningMsg.remove();
        }
        
        // Show all duration inputs
        if (durationContainer) {
            durationContainer.classList.remove('d-none');
        }
    });
    
    // Then mark only the last one
    const lastThumbnail = allThumbnails[allThumbnails.length - 1];
    if (lastThumbnail) {
        const durationContainer = lastThumbnail.querySelector('.duration-input-container');
        if (durationContainer) {
            durationContainer.classList.add('d-none');
        }
        
        const cardBody = lastThumbnail.querySelector('.card-body');
        if (cardBody) {
            cardBody.insertAdjacentHTML('beforeend', 
                '<p class="small text-muted mt-1 last-frame-warning"><i class="bi bi-info-circle"></i> Last frame duration not used</p>'
            );
        }
    }
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
                const serverPath = data.upload_url;
                
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

// Update editor UI based on job status
function updateEditorProgress(jobId, status, progress, message) {
    // Only update if we have a progress container
    if (!elements.progressContainer) return;
    
    // Update the current active job ID
    window.currentActiveJobId = jobId;
    
    // Show the progress container
    elements.progressContainer.classList.remove('d-none');
    
    // Update progress bar
    if (elements.progressBar) {
        elements.progressBar.style.width = `${progress}%`;
        elements.progressBar.setAttribute('aria-valuenow', progress);
        elements.progressBar.textContent = `${progress}%`;
    }
    
    // Update status message
    if (elements.progressStatus) {
        elements.progressStatus.textContent = message || 'Processing...';
    }
    
    // Handle different job statuses
    if (status === 'completed') {
        // Show completion message
        showMessage('Video generation completed!', 'success');
        
        // Update UI for completed state
        if (elements.progressBar) {
            elements.progressBar.classList.remove('progress-bar-animated', 'progress-bar-striped');
            elements.progressBar.classList.add('bg-success');
        }
        
        // After a delay, hide the progress container
        setTimeout(() => {
            // Switch to the job queue tab to show the completed job
            const queueTab = document.getElementById('queue-tab');
            if (queueTab) {
                queueTab.click();
            }
            
            // Reset the editor state
            elements.progressContainer.classList.add('d-none');
            window.currentActiveJobId = null;
            
            // Reload the job queue to see the completed job
            loadJobQueue();
        }, 3000);
    } 
    else if (status === 'failed' || status === 'cancelled') {
        // Show error message
        showMessage(`Job ${status}: ${message}`, 'error');
        
        // Update UI for failed state
        if (elements.progressBar) {
            elements.progressBar.classList.remove('progress-bar-animated', 'progress-bar-striped');
            elements.progressBar.classList.add('bg-danger');
        }
        
        // After a delay, hide the progress container
        setTimeout(() => {
            elements.progressContainer.classList.add('d-none');
            window.currentActiveJobId = null;
        }, 5000);
    } 
    else if (status === 'running') {
        // Ensure progress bar is styled correctly for running state
        if (elements.progressBar) {
            elements.progressBar.classList.add('progress-bar-animated', 'progress-bar-striped');
            elements.progressBar.classList.remove('bg-success', 'bg-danger');
            elements.progressBar.classList.add('bg-primary');
        }
        
        // Update preview image if provided
        if (event && event.current_latents) {
            updateProgressPreview(event.current_latents);
        }
    }
}

// Update the preview image in the progress area
function updateProgressPreview(previewUrl) {
    if (!elements.previewContainer || !elements.previewImage) return;
    
    // Show the preview container
    elements.previewContainer.classList.remove('d-none');
    
    // Update the preview image
    elements.previewImage.src = previewUrl;
}

// Function to show upload modal
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
    
    // Connect to the job websocket if not already connected
    // This ensures we'll show any running job's progress in the editor
    import('./job_queue.js').then(jobQueueModule => {
        if (typeof jobQueueModule.connectJobWebsocket === 'function') {
            jobQueueModule.connectJobWebsocket();
            console.log('Connected to job websocket from editor');
        }
    }).catch(err => console.error('Error connecting to job websocket from editor:', err));
    
    // Add sort buttons event listeners
    const sortAscBtn = document.getElementById('sortAscBtn');
    const sortDescBtn = document.getElementById('sortDescBtn');
    
    if (sortAscBtn) {
        sortAscBtn.addEventListener('click', () => sortTimeline('asc'));
    }
    
    if (sortDescBtn) {
        sortDescBtn.addEventListener('click', () => sortTimeline('desc'));
    }

    console.log('Files module initialized');
    
    // Initialize upload modal
    // Add event listeners
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
}

// Initialize event listeners for the editor module
function initEventListeners() {
    
    // Button click events
    if (elements.uploadImagesBtn) {
        console.log('uploadImagesBtn clicked');
        elements.uploadImagesBtn.addEventListener('click', showUploadModal || (() => {}));
    }
    
    if (elements.generateVideoBtn) {
        console.log('generateVideoBtn clicked');
        elements.generateVideoBtn.addEventListener('click', startGeneration);
    }
    
    // Add Save Job button event
    const saveJobBtn = document.getElementById('saveJobBtn');
    if (saveJobBtn) {
        console.log('saveJobBtn event listener added');
        saveJobBtn.addEventListener('click', saveJob);
    }
    
    // File input change event
    if (elements.fileInput && handleFileSelect) {
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
        if (handleDragOver) elements.uploadDropArea.addEventListener('dragover', handleDragOver);
        if (handleDragLeave) elements.uploadDropArea.addEventListener('dragleave', handleDragLeave);
        if (handleFileDrop) elements.uploadDropArea.addEventListener('drop', handleFileDrop);
        if (triggerFileInput) elements.uploadDropArea.addEventListener('click', triggerFileInput);
    }
    
    // Add to timeline button
    if (elements.addToTimelineBtn) {
        elements.addToTimelineBtn.addEventListener('click', handleAddToTimeline);
    }
    
    // Frame edit modal events
    if (elements.replaceImageBtn && triggerFileInput) {
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
            generateBtn.parentNode.appendChild(clearBtn);
        }
    }
}

// Function to add timeline drop zone for drag & drop from desktop
function initTimelineDropZone() {
    if (!elements.timelineContainer) return;
    
    // Initialize empty timeline UI
    updateTimelineStatus();
    
    // Ensure horizontal layout
    enforceHorizontalLayout();
    
    elements.timelineContainer.addEventListener('dragover', (e) => {
        // Only handle file drops when the timeline is empty
        if (timeline.length === 0) {
            e.preventDefault();
            e.stopPropagation();
            
            // If we're over the empty state dropzone
            const dropzone = elements.timelineContainer.querySelector('.timeline-dropzone');
            if (dropzone) {
                dropzone.classList.add('active');
            }
        }
    });
    
    elements.timelineContainer.addEventListener('dragleave', (e) => {
        // Only handle file drops when the timeline is empty
        if (timeline.length === 0) {
            e.preventDefault();
            e.stopPropagation();
            
            // If we're over the empty state dropzone
            const dropzone = elements.timelineContainer.querySelector('.timeline-dropzone');
            if (dropzone) {
                dropzone.classList.remove('active');
            }
        }
    });
    
    elements.timelineContainer.addEventListener('drop', (e) => {
        // Only handle file drops when the timeline is empty
        if (timeline.length === 0) {
            e.preventDefault();
            e.stopPropagation();
            
            // Remove active classes
            const dropzone = elements.timelineContainer.querySelector('.timeline-dropzone');
            if (dropzone) {
                dropzone.classList.remove('active');
            }
            
            // Handle files dropped directly from the desktop
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                // Instead of processing immediately, show the upload modal first
                if (clearSelectedFiles) clearSelectedFiles(); // Clear any previously selected files
                if (processSelectedFiles) processSelectedFiles(e.dataTransfer.files);
                
                // Show the upload modal to allow user to confirm/adjust before adding to timeline
                if (uploadModal) {
                    uploadModal.show();
                }
            }
        }
    });
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
    
    e.stopPropagation(); // Stop propagation to prevent container handlers from firing
    
    e.dataTransfer.dropEffect = 'move';
    
    // Add visual indicator that this is a valid drop target
    this.classList.add('drag-over');
    
    return false;
}

function handleTimelineDragLeave(e) {
    e.stopPropagation(); // Stop propagation
    this.classList.remove('drag-over');
}

function handleTimelineDrop(e) {
    if (e.preventDefault) {
        e.preventDefault();
    }
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

// Function to update timeline status
function updateTimelineStatus() {
    // Enable/disable generate button based on timeline
    if (elements.generateVideoBtn) {
        // Need at least 1 frame to generate a video
        elements.generateVideoBtn.disabled = timeline.length < 1;
    }
    
    // Also update Save Job button if it exists
    const saveJobBtn = document.getElementById('saveJobBtn');
    if (saveJobBtn) {
        saveJobBtn.disabled = timeline.length < 1;
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
            
            // Remove include last frame checkbox from non-last items
            const existingCheckbox = item.querySelector('.include-last-frame-container');
            if (existingCheckbox && idx !== lastIndex) {
                existingCheckbox.remove();
            }
        });
        
        // Add note and checkbox to the last item
        if (lastIndex >= 0) {
            const lastItem = timelineItems[lastIndex];
            const durationSection = lastItem.querySelector('.timeline-item-duration');
            
            
            if (durationSection) {
                // First check if there's already a checkbox
                let includeLastFrameCheckbox = lastItem.querySelector('.include-last-frame-checkbox');
                let checkboxContainer = lastItem.querySelector('.include-last-frame-container');
                let isChecked = includeLastFrameCheckbox ? includeLastFrameCheckbox.checked : false;
                
                // If no checkbox exists, create it
                if (!checkboxContainer) {
                    checkboxContainer = document.createElement('div');
                    checkboxContainer.className = 'include-last-frame-container mt-2';
                    checkboxContainer.innerHTML = `
                        <div class="form-check">
                            <input class="form-check-input include-last-frame-checkbox" type="checkbox" id="includeLastFrame_${lastIndex}" ${isChecked ? 'checked' : ''}>
                            <label class="form-check-label small" for="includeLastFrame_${lastIndex}">
                                Include as segment
                            </label>
                        </div>
                    `;
                    durationSection.appendChild(checkboxContainer);
                    
                    // Get the newly created checkbox
                    includeLastFrameCheckbox = checkboxContainer.querySelector('.include-last-frame-checkbox');
                }
                
                // Add event listener to checkbox
                if (includeLastFrameCheckbox) {
                    // First remove any existing listeners
                    const newCheckbox = includeLastFrameCheckbox.cloneNode(true);
                    includeLastFrameCheckbox.parentNode.replaceChild(newCheckbox, includeLastFrameCheckbox);
                    includeLastFrameCheckbox = newCheckbox;
                    
                    includeLastFrameCheckbox.addEventListener('change', (e) => {
                        const durationInput = lastItem.querySelector('.duration-input');
                        const note = lastItem.querySelector('.last-frame-info');
                        
                        if (e.target.checked) {
                            // Enable duration input when checked
                            if (durationInput) durationInput.disabled = false;
                            if (note) note.innerHTML = '<i class="bi bi-info-circle"></i> Duration used for this frame';
                        } else {
                            // Disable duration input when unchecked
                            if (durationInput) durationInput.disabled = true;
                            if (note) note.innerHTML = '<i class="bi bi-info-circle"></i> Duration not used for last frame';
                        }
                    });
                }
                
                // Add or update the note
                let note = lastItem.querySelector('.last-frame-info');
                if (!note) {
                    note = document.createElement('div');
                    note.className = 'last-frame-info';
                    durationSection.appendChild(note);
                }
                
                // Set the note text based on checkbox state
                if (includeLastFrameCheckbox && includeLastFrameCheckbox.checked) {
                    note.innerHTML = '<i class="bi bi-info-circle"></i> Duration used for this frame';
                } else {
                    note.innerHTML = '<i class="bi bi-info-circle"></i> Duration not used for last frame';
                }
                
                // Update the duration input based on checkbox state
                const durationInput = lastItem.querySelector('.duration-input');
                if (durationInput) {
                    durationInput.disabled = !(includeLastFrameCheckbox && includeLastFrameCheckbox.checked);
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

// Function to sort timeline by filename
function sortTimeline(direction) {
    if (timeline.length <= 1) {
        // Nothing to sort if there's only one item or none
        return;
    }
    
    console.log(`Sorting timeline ${direction}`);
    
    // First get all elements
    const items = Array.from(elements.timelineContainer.querySelectorAll('.timeline-item'));
    
    // Get the current order of items
    const originalOrder = items.map(item => {
        const img = item.querySelector('img');
        return img ? img.title || img.alt : '';
    });
    
    // Sort the timeline array by filename
    timeline.sort((a, b) => {
        const aName = a.serverPath ? a.serverPath.split('/').pop() : a.name;
        const bName = b.serverPath ? b.serverPath.split('/').pop() : b.name;
        
        if (direction === 'asc') {
            return aName.localeCompare(bName);
        } else {
            return bName.localeCompare(aName);
        }
    });
    
    // Clear the timeline container
    elements.timelineContainer.innerHTML = '';
    
    // Re-add items in sorted order
    for (const item of timeline) {
        // Create a file object for addItemToTimeline
        const fileObj = {
            src: item.src,
            file: item.file,
            serverPath: item.serverPath,
            duration: item.duration,
            prompt: item.prompt,
            name: item.name || (item.serverPath ? item.serverPath.split('/').pop() : ''),
            valid: item.valid
        };
        
        addItemToTimeline(fileObj);
    }
    
    // Update the status of the timeline
    updateTimelineStatus();
    
    // Log the sorted order
    console.log('Timeline sorted successfully');
    
    // Show a message to the user
    showMessage(`Timeline sorted ${direction === 'asc' ? 'ascending' : 'descending'} by filename`, 'success');
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
    
    // Generate a job ID
    const timestamp = Math.floor(Date.now() / 1000);
    let jobId = `${timestamp}_${Math.floor(Math.random() * 1000)}`;
    // Use the current job ID if it exists
    if (window.currentJobId) {
        jobId = window.currentJobId;
        console.log('Using existing job ID:', jobId);
    } else {
        window.currentJobId = jobId;
    }
    
    // Set this as the current active job for UI updates
    window.currentActiveJobId = jobId;
    
    // Show progress UI
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressStatus = document.getElementById('progressStatus');
    
    if (progressContainer && progressBar && progressStatus) {
        progressContainer.classList.remove('d-none');
        progressBar.style.width = '0%';
        progressBar.setAttribute('aria-valuenow', 0);
        progressBar.textContent = '0%';
        progressStatus.textContent = 'Preparing generation request...';
        
        // Add animation classes to progress bar
        progressBar.classList.add('progress-bar-animated', 'progress-bar-striped', 'bg-primary');
        progressBar.classList.remove('bg-success', 'bg-danger');
    }
    
    // Use common function to prepare the job payload
    const payload = prepareJobPayload();
    
    // Ensure job_settings has the proper ModuleJobSettings structure
    payload.job_settings = prepareModuleJobSettings(payload.job_settings);
    
    console.log('Preparing job with settings:', JSON.stringify(payload, null, 2));
    
    // First save the job
    const savePayload = {
        job_id: jobId,
        status: "saved",
        progress: 0,
        message: "Job saved",
        result_video: "",
        segments: timeline.map(item => item.serverPath),
        is_valid: true,
        missing_images: [],
        job_settings: payload.job_settings,
        queue_position: -1,
        created_timestamp: timestamp
    };
    
    // Save the job first, then run it
    fetch(`/api/save_job/${jobId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(savePayload)
    })
    .then(response => {
        if (!response.ok) {
            return response.text().then(text => {
                throw new Error(`Failed to save job: ${text}`);
            });
        }
        return response.json();
    })
    .then(data => {
        console.log('Job saved:', data);
        
        // Clear the timeline after successfully saving a job
        // Clear UI
        elements.timelineContainer.innerHTML = '';
        
        // Clear timeline array
        timeline.length = 0;
        
        // Update status
        updateTimelineStatus();
        
        const originalConfirm = window.confirm;
        window.confirm = () => true;
        
        try {
            runJob(jobId);
            console.log(`Job ${jobId} started using job_queue.runJob`);
        } catch (err) {
            console.error('Error running job from job_queue module:', err);
            showMessage(`Error starting job: ${err.message}`, 'error');
        } finally {
            // Restore original confirm function
            window.confirm = originalConfirm;
        }
    })
    .catch(error => {
        console.error('Error starting generation:', error);
        
        if (progressStatus) {
            progressStatus.textContent = 'Error: ' + error.message;
        }
        if (progressBar) {
            progressBar.classList.add('bg-danger');
        }
    });
}

// Helper function to convert job settings to ModuleJobSettings format
function prepareModuleJobSettings(settings) {
    // Create a dictionary with module name as key and settings as value
    // Always use "framepack" as the module name
    return {
        "framepack": settings
    };
}

// Function to save job without starting generation or clearing timeline
function saveJob() {
    if (timeline.length === 0) {
        alert('Please add at least one image to the timeline before saving a job.');
        return;
    }
    
    // Check for invalid images in the timeline
    const invalidImages = timeline.filter(item => item.valid === false);
    if (invalidImages.length > 0) {
        alert(`Cannot save job: ${invalidImages.length} image(s) are missing or invalid. Please replace them before saving.`);
        return;
    }
    
    // Generate a job ID if not provided
    const timestamp = Math.floor(Date.now() / 1000);
    let jobId = `${timestamp}_${Math.floor(Math.random() * 1000)}`;
    // Use the current job ID if it exists
    if (window.currentJobId) {
        jobId = window.currentJobId;
    } else {
        window.currentJobId = jobId;
    }
    
    // Use common function to prepare the job settings
    const jobSettings = prepareJobPayload();
    
    // Prepare the segment paths for the status object
    const segmentPaths = timeline.map(item => item.serverPath);
    
    // Create a SaveJobRequest structure matching the backend requirements
    const payload = {
        job_id: jobId,
        status: "saved",
        progress: 0,
        message: "Job saved",
        result_video: "",
        segments: segmentPaths,
        is_valid: true,
        missing_images: [],
        job_settings: {framepack: jobSettings},
        queue_position: -1,
        created_timestamp: timestamp
    };
    
    console.log('Saving job with payload:', JSON.stringify(payload, null, 2));
    showMessage('Saving job...', 'info');
    
    // Make API call to save job at /api/save_job/{job_id}
    fetch(`/api/save_job/${jobId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    })
    .then(response => {
        if (!response.ok) {
            return response.text().then(text => {
                try {
                    // Try to parse the error as JSON
                    const errorObj = JSON.parse(text);
                    throw new Error(`Failed to save job: ${JSON.stringify(errorObj)}`);
                } catch (e) {
                    throw new Error(`Failed to save job: ${text}`);
                }
            });
        }
        showMessage('Job saved successfully!', 'success');
        return response.json();
    })
    .then(data => {
        console.log('Job saved:', data);
        
        // Show success message
        showMessage('Job saved successfully! Job ID: ' + jobId, 'success');
        
        // Try to refresh the job queue, but handle if the function isn't available
        loadJobQueue();
    })
    .catch(error => {
        console.error('Error saving job:', error);
        showMessage('Error saving job: ' + error.message, 'error');
    });
}

// Function to prepare job payload from timeline
function prepareJobPayload() {
    // Collect segments data
    const segments = [];
    let includeLastFrame = false;
    
    // For a single image, we need special handling to create a self-transition
    if (timeline.length === 1) {
        const singleFrame = timeline[0];
        const promptInput = elements.timelineContainer.querySelector('.prompt-text');
        
        // Use the full server path directly
        const imagePath = singleFrame.serverPath;
        if (!imagePath) {
            console.error('No server path found for the single image:', singleFrame);
            throw new Error('Missing server path for the image. Please try re-uploading.');
        }
        
        console.log(`Single image path: ${imagePath}`);
        
        // Add single segment with longer duration for self-transition
        segments.push({
            image_path: imagePath,
            prompt: promptInput ? promptInput.value : '',
            duration: singleFrame.duration || 3.0
        });
    } else {
        // Process multiple images: Create transitions between each pair of frames
        for (let i = 0; i < timeline.length; i++) {
            const currentFrame = timeline[i];
            const promptInput = Array.from(elements.timelineContainer.children)
                .find((_, idx) => idx === i)
                ?.querySelector('.prompt-text');
            
            // Use the full server path directly
            const imagePath = currentFrame.serverPath;
            if (!imagePath) {
                console.error(`No server path found for image at index ${i}:`, currentFrame);
                throw new Error(`Missing server path for image ${i + 1}. Please try re-uploading.`);
            }
            
            // Log each path for debugging
            console.log(`Segment ${i + 1} path: ${imagePath}`);
            
            segments.push({
                image_path: imagePath,
                prompt: promptInput ? promptInput.value : '',
                duration: currentFrame.duration
            });
        }
        
        // Add the last frame with its prompt
        const lastIndex = timeline.length - 1;
        const lastFrame = timeline[lastIndex];
        const lastPromptInput = Array.from(elements.timelineContainer.children)
            .find((_, idx) => idx === lastIndex)
            ?.querySelector('.prompt-text');
            
        // Check if the includeAsSegment property exists, fall back to the checkbox if not
        includeLastFrame = lastFrame.includeAsSegment;
        if (includeLastFrame === undefined) {
            const includeLastFrameCheckbox = Array.from(elements.timelineContainer.children)
                .find((_, idx) => idx === lastIndex)
                ?.querySelector('.include-last-frame-checkbox');
            includeLastFrame = includeLastFrameCheckbox ? includeLastFrameCheckbox.checked : false;
        }
        
        // Check if the last frame has a valid path
        if (!lastFrame.serverPath) {
            console.error(`No server path found for the last image:`, lastFrame);
            throw new Error(`Missing server path for the last image. Please try re-uploading.`);
        }
        
        // Log the last frame for debugging
        console.log(`Last frame path: ${lastFrame.serverPath}`);

    }
    
    if (segments.length === 0) {
        throw new Error('No valid segments found in timeline.');
    }
    
    // Generate a simple job name based on first image filename and timestamp
    const currentDate = new Date();
    const timestamp = currentDate.toISOString().slice(0, 16).replace('T', ' ');
    let firstImageName = timestamp;
    if (segments[0].image_path.indexOf('/') !== -1) {
        firstImageName = segments[0].image_path.split('/').pop().split('.')[0];
    } else {
        firstImageName = segments[0].image_path.split('\\').pop().split('.')[0];
    }

    // Create the payload with all form settings
    return {
        global_prompt: elements.globalPrompt ? elements.globalPrompt.value : "",
        negative_prompt: elements.negativePrompt ? elements.negativePrompt.value : "",
        segments: segments,
        seed: Math.floor(Math.random() * 100000),
        steps: elements.steps ? parseInt(elements.steps.value) : 25,
        guidance_scale: elements.guidanceScale ? parseFloat(elements.guidanceScale.value) : 10.0,
        use_teacache: elements.useTeacache ? elements.useTeacache.checked : true,
        enable_adaptive_memory: elements.enableAdaptiveMemory ? elements.enableAdaptiveMemory.checked : true,
        resolution: elements.resolution ? parseInt(elements.resolution.value) : 640,
        mp4_crf: 16,
        gpu_memory_preservation: 6.0,
        include_last_frame: includeLastFrame || false
    };
}

// Function to connect to job websocket and handle updates
// NOTE: This function is now simplified to use the central WebSocket handler in job_queue.js
function setupJobWebsocketConnection(jobId) {
    try {
        // Set the current active job ID
        window.currentActiveJobId = jobId;
        
        // The job_queue.js module will handle all the WebSocket updates
        // via its central listener
        console.log(`Setting up job connection for job ${jobId} (using central WebSocket handler)`);
        
        // Just connect to the WebSocket - no need to register our own handler
        connectJobWebsocket(jobId);
        
        return true;
    } catch (error) {
        console.error("Error connecting to job websocket:", error);
        
        return null;
    }
}

// Update the job UI based on status data
// This function is now primarily for backward compatibility and polling
function updateJobUI(data) {
    // If we have a global function, use that instead for consistency
    if (typeof window.updateEditorProgress === 'function') {
        window.updateEditorProgress(
            data.job_id, 
            data.status, 
            data.progress, 
            data.message,
            data // Pass whole data object as eventData
        );
        return;
    }
    
    // Fallback implementation for backwards compatibility
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
        // Use the most recent segment (avoid duplicates)
        const latestSegment = data.segments[data.segments.length - 1];
        
        currentJobImage.src = latestSegment;
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
                jobQueueModule.loadJobDetails(data.job_id);
                
                // Highlight this job in the list
                const jobItems = document.querySelectorAll('.job-item');
                jobItems.forEach(item => {
                    item.classList.remove('active');
                    if (item.dataset.jobId === data.job_id) {
                        item.classList.add('active');
                    }
                });
            });
        };
    } else {
        // Hide the thumbnail if no segments are available yet
        currentJobThumbnail.classList.add('d-none');
    }
}

// Handle job completion (success or failure)
function handleJobCompletion(data) {
    const progressStatus = document.getElementById('progressStatus');
    const progressContainer = document.getElementById('progressContainer');
    const generateBtn = document.getElementById('generateVideoBtn');
    
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

        // Clear the current job ID
        window.currentJobId = null;
        
        // Show confirmation message
        showMessage('Timeline cleared', 'info');
    }
}

// Function to handle adding to timeline
function handleAddToTimeline() {
    // Get selected files using the getter function
    let files = [];
    
    // Try various ways to get the files, in order of preference
    if (getSelectedFiles) {
        files = getSelectedFiles();
    } else if (window.filesModule && window.filesModule.getSelectedFiles) {
        files = window.filesModule.getSelectedFiles();
    }
    
    if (!files || files.length === 0) {
        alert('Please select at least one image to add to the timeline.');
        return;
    }
    
    console.log(`Adding ${files.length} files to timeline`);
    
    // Show loading indicator
    const addToTimelineBtn = document.getElementById('addToTimelineBtn');
    if (addToTimelineBtn) {
        addToTimelineBtn.disabled = true;
        addToTimelineBtn.innerHTML = '<i class="bi bi-hourglass-split me-2"></i> Uploading...';
    }
    
    // First, upload all files to the server
    const uploadPromises = files.map(fileObj => uploadFileToServer(fileObj.file));
    
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
                            timeline[currentEditIndex].src = files[0].src;
                            timeline[currentEditIndex].file = files[0].file;
                            timeline[currentEditIndex].valid = true; // Mark as valid since it's new
                            
                            // Update the DOM
                            const imgElement = timelineItems[currentEditIndex].querySelector('img');
                            if (imgElement) {
                                imgElement.src = files[0].src;
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
                        }
                    }
                }
                
                // Reset edit mode
                keepCurrentImage = false;
                currentEditIndex = -1;
            } else {
                // Normal mode - add each file to the timeline with its server path
                files.forEach((fileObj, index) => {
                    if (serverPaths[index]) {
                        // Use the server path instead of local file reference
                        // This is the EXACT path returned from the server
                        fileObj.serverPath = serverPaths[index];
                        console.log(`Adding file ${index + 1}/${files.length}: ${fileObj.name} (${serverPaths[index]})`);
                        addItemToTimeline(fileObj);
                    }
                });
                
                // Make sure timeline UI is updated
                updateTimelineStatus();
            }
            
            // Reset selected files
            if (clearSelectedFiles) {
                clearSelectedFiles();
            } else if (window.filesModule && window.filesModule.clearSelectedFiles) {
                window.filesModule.clearSelectedFiles();
            }
            
            // Close the upload modal 
            closeUploadModal();
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
    // Remove the main dropzone if this is the first item and it's actually a dropzone
    // We identify actual dropzones to avoid removing real frames
    const mainDropZone = elements.timelineContainer.querySelector('.timeline-item.dropzone-item');
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
    
    // Generate a unique ID for this timeline item
    const itemId = `timeline-item-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
    timelineItem.setAttribute('data-item-id', itemId);
    
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
    
    // Store the item ID in the timeline object for later reference
    const timelineObj = {
        itemId: itemId,
        src: displaySrc,
        file: fileObj.file,
        serverPath: serverPath,
        duration: frameDuration,
        prompt: fileObj.prompt || '',
        valid: imageExists
    };
    
    timeline.push(timelineObj);
    
    elements.timelineContainer.appendChild(timelineItem);
    
    // After adding the timelineItem to the container, check if it's the last item and update any checkboxes
    const timelineItems = elements.timelineContainer.querySelectorAll('.timeline-item');
    if (timelineItems.length > 0 && timelineItem === timelineItems[timelineItems.length - 1]) {
        // Call updateTimelineStatus to update the UI with checkboxes, notes, etc.
        updateTimelineStatus();
        
        // Find the checkbox if it was added by updateTimelineStatus and attach a listener
        const checkbox = timelineItem.querySelector('.include-last-frame-checkbox');
        if (checkbox) {
            checkbox.addEventListener('change', (e) => {
                const index = Array.from(elements.timelineContainer.children).indexOf(timelineItem);
                if (index >= 0 && index < timeline.length) {
                    // Save the checkbox state to the timeline array
                    timeline[index].includeAsSegment = e.target.checked;
                }
            });
        }
    }
    
    return timelineItem;
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

// Function to update the timeline array based on the DOM order
function updateTimelineArray() {
    const newTimeline = [];
    const items = Array.from(elements.timelineContainer.querySelectorAll('.timeline-item'));
    
    // Log the number of items found in the DOM for debugging
    console.log(`Found ${items.length} items in the timeline DOM`);
    
    items.forEach((item, index) => {
        // Get the item ID from the data attribute
        const itemId = item.getAttribute('data-item-id');
        const promptText = item.querySelector('.prompt-text');
        const durationInput = item.querySelector('.duration-input');
        const includeLastCheckbox = item.querySelector('.include-last-frame-checkbox');
        
        // Find matching item in original timeline by item ID
        const originalItem = timeline.find(t => t.itemId === itemId);
        
        if (originalItem) {
            // Push a copy of the original with updated values
            newTimeline.push({
                ...originalItem,
                prompt: promptText ? promptText.value : originalItem.prompt,
                duration: durationInput ? parseFloat(durationInput.value) : originalItem.duration,
                includeAsSegment: includeLastCheckbox ? includeLastCheckbox.checked : originalItem.includeAsSegment
            });
        } else {
            // Fallback to img src if we can't find by ID
            const img = item.querySelector('img');
            if (img) {
                const imgSrc = img.src;
                const serverPath = img.title;
                
                // Try to find by image source
                const srcMatch = timeline.find(t => t.src === imgSrc);
                
                if (srcMatch) {
                    newTimeline.push({
                        ...srcMatch,
                        prompt: promptText ? promptText.value : srcMatch.prompt,
                        duration: durationInput ? parseFloat(durationInput.value) : srcMatch.duration,
                        includeAsSegment: includeLastCheckbox ? includeLastCheckbox.checked : srcMatch.includeAsSegment
                    });
                } else {
                    // Last resort: create a new item based on what we can extract
                    newTimeline.push({
                        itemId: itemId || `fallback-${Date.now()}-${index}`,
                        src: imgSrc,
                        serverPath: serverPath,
                        duration: durationInput ? parseFloat(durationInput.value) : 3.0,
                        prompt: promptText ? promptText.value : '',
                        valid: !img.classList.contains('invalid-image'),
                        includeAsSegment: includeLastCheckbox ? includeLastCheckbox.checked : false
                    });
                }
            }
        }
    });
    
    // Clear the timeline array
    timeline.length = 0;
    
    // Add the new items in the updated order
    newTimeline.forEach(item => timeline.push(item));
    
    // Log for debugging
    console.log('Updated timeline array:', timeline.map(item => ({
        itemId: item.itemId,
        serverPath: item.serverPath,
        duration: item.duration,
        includeAsSegment: item.includeAsSegment
    })));
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
    saveJob,
    uploadModal,
    setupJobWebsocketConnection,
    updateJobUI,
    handleJobCompletion
};
