// Batch Processing Module
// Handles batch image uploads and processing

// DOM Elements
let batchDropArea;
let batchFileInput;
let batchImagesContainer;
let batchProcessBtn;
let uploadBatchImagesBtn;
let clearBatchBtn;
let batchSettingsForm;

import {
    runJob
} from './job_queue.js';

import {
    uploadFileToServer,
    getSettingsFromForm,
    prepareJobPayload,
    saveAndProcessJob
} from './job_utils.js';


// State
const batchImages = [];

// Initialize batch processing module
export function initBatch() {
    console.log('Initializing batch processing module');
    
    // Get DOM elements
    batchFileInput = document.getElementById('batchFileInput');
    batchImagesContainer = document.getElementById('batchImagesContainer');
    batchProcessBtn = document.getElementById('batchProcessBtn');
    uploadBatchImagesBtn = document.getElementById('uploadBatchImagesBtn');
    clearBatchBtn = document.getElementById('clearBatchBtn');
    batchSettingsForm = document.getElementById('batchSettingsForm');
    
    // Set up event listeners
    if (batchFileInput) {
        batchFileInput.addEventListener('change', handleBatchFileSelect);
    }
    
    if (uploadBatchImagesBtn) {
        uploadBatchImagesBtn.addEventListener('click', () => batchFileInput.click());
    }
    
    if (clearBatchBtn) {
        clearBatchBtn.addEventListener('click', clearBatchImages);
    }
    
    if (batchProcessBtn) {
        batchProcessBtn.addEventListener('click', processBatch);
    }
    
    // Add module selection change event listener
    const batchVideoModuleSelect = document.getElementById('batchVideoModule');
    if (batchVideoModuleSelect) {
        batchVideoModuleSelect.addEventListener('change', handleBatchModuleSelection);
        // Initialize settings visibility based on current selection
        handleBatchModuleSelection({ target: batchVideoModuleSelect });
    }
    
    // Add drag/drop handling to the batch container
    if (batchImagesContainer) {
        batchImagesContainer.addEventListener('dragover', (e) => {
            // Only handle file drops when the container has the dropzone or is empty
            e.preventDefault();
            e.stopPropagation();
            
            // If we're over the empty state dropzone
            const dropzone = batchImagesContainer.querySelector('.timeline-dropzone');
            if (dropzone) {
                dropzone.classList.add('active');
            }
        });
        
        batchImagesContainer.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            // If we're over the empty state dropzone
            const dropzone = batchImagesContainer.querySelector('.timeline-dropzone');
            if (dropzone) {
                dropzone.classList.remove('active');
            }
        });
        
        batchImagesContainer.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            // Remove active classes
            const dropzone = batchImagesContainer.querySelector('.timeline-dropzone');
            if (dropzone) {
                dropzone.classList.remove('active');
            }
            
            // Handle files dropped directly from the desktop
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                handleBatchFiles(e.dataTransfer.files);
            }
        });
    }
    
    // Initialize batch container with empty state
    updateBatchStatus();
}

// Handle file selection
function handleBatchFileSelect(e) { 
    console.log('handleBatchFileSelect');
    if (e.target.files && e.target.files.length > 0) {
        handleBatchFiles(e.target.files);
        // Update batch display and status
        updateBatchStatus();
    }
}

// Process the selected files for batch
function handleBatchFiles(files) {
    console.log('handleBatchFiles');
    const imageFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
    
    if (imageFiles.length === 0) {
        // Show error or notification
        console.error('No valid image files selected');
        return;
    }
    
    // Add image files to batch
    imageFiles.forEach(file => {
        addImageToBatch(file);
    });
}

// Add an image to the batch
function addImageToBatch(file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
        const imageId = `batch-image-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        
        // Create batch image entry
        const imageEntry = {
            id: imageId,
            file: file,
            dataUrl: e.target.result
        };
        
        // Add to batch images array
        batchImages.push(imageEntry);
        
        // Create thumbnail element
        const thumbnailCol = document.createElement('div');
        thumbnailCol.className = 'col-6 col-md-4 col-lg-3';
        thumbnailCol.dataset.imageId = imageId;
        
        thumbnailCol.innerHTML = `
            <div class="card h-100">
                <img src="${e.target.result}" class="card-img-top" alt="${file.name}">
                <div class="card-body p-2">
                    <p class="card-text small text-truncate">${file.name}</p>
                </div>
                <div class="card-footer p-1 text-end">
                    <button class="btn btn-sm btn-outline-danger remove-batch-image" data-image-id="${imageId}">
                        <i class="bi bi-x"></i>
                    </button>
                </div>
            </div>
        `;
        
        // Add thumbnail to container
        batchImagesContainer.appendChild(thumbnailCol);
        
        // Add click handler for remove button
        const removeBtn = thumbnailCol.querySelector('.remove-batch-image');
        if (removeBtn) {
            removeBtn.addEventListener('click', (e) => {
                e.preventDefault();
                removeImageFromBatch(imageId);
            });
        }
        
        // Enable process and clear buttons
        updateButtonStates();
        updateBatchStatus();
    };
    
    reader.readAsDataURL(file);
}

// Update button states
function updateButtonStates() {
    // Enable/disable process and clear buttons based on whether there are images
    batchProcessBtn.disabled = batchImages.length === 0;
    clearBatchBtn.disabled = batchImages.length === 0;
}

// Remove an image from the batch
function removeImageFromBatch(imageId) {
    // Remove from array
    const index = batchImages.findIndex(img => img.id === imageId);
    if (index !== -1) {
        batchImages.splice(index, 1);
    }
    
    // Remove from DOM
    const thumbnailElement = batchImagesContainer.querySelector(`[data-image-id="${imageId}"]`);
    if (thumbnailElement) {
        thumbnailElement.remove();
    }
    
    // Update batch status (will show empty state if no images left)
    updateBatchStatus();
}

// Function to update batch display status based on content
function updateBatchStatus() {
    // Enable/disable buttons based on whether there are images
    updateButtonStates();
    console.log('Batch images:', batchImages);
    // Show message when batch is empty
    if (batchImages.length === 0) {
        // Clear the container and add the primary dropzone
        batchImagesContainer.innerHTML = `
            <div class="timeline-dropzone">
                <div class="text-center py-5">
                    <i class="bi bi-images fs-1 mb-3"></i>
                    <h5 class="fw-bold mb-3">Add Images to Batch</h5>
                    <p class="text-muted mb-4">Upload one or more images to process with the same settings</p>
                    <button class="btn btn-primary mt-2 px-4 py-2" id="dropzoneBatchUploadBtn">
                        <i class="bi bi-upload me-2"></i> Browse for Images
                    </button>
                </div>
            </div>
        `;
        
        // Add click event for the upload button
        const dropzoneUploadBtn = document.getElementById('dropzoneBatchUploadBtn');
        if (dropzoneUploadBtn) {
            dropzoneUploadBtn.addEventListener('click', () => batchFileInput.click());
        }
        
        // Remove any secondary drop zone when empty
        const secondaryDropZone = document.getElementById('batchSecondaryDropZone');
        if (secondaryDropZone) {
            secondaryDropZone.remove();
        }
    } else {
        // Clear the container and re-add all images
        batchImagesContainer.innerHTML = '';
        
        // Re-add all batch images to the container
        batchImages.forEach(image => {
            const thumbnailCol = document.createElement('div');
            thumbnailCol.className = 'col-6 col-md-4 col-lg-3';
            thumbnailCol.dataset.imageId = image.id;
            
            thumbnailCol.innerHTML = `
                <div class="card h-100">
                    <img src="${image.dataUrl}" class="card-img-top" alt="${image.file.name}">
                    <div class="card-body p-2">
                        <p class="card-text small text-truncate">${image.file.name}</p>
                    </div>
                    <div class="card-footer p-1 text-end">
                        <button class="btn btn-sm btn-outline-danger remove-batch-image" data-image-id="${image.id}">
                            <i class="bi bi-x"></i>
                        </button>
                    </div>
                </div>
            `;
            
            batchImagesContainer.appendChild(thumbnailCol);
            
            // Add click handler for remove button
            const removeBtn = thumbnailCol.querySelector('.remove-batch-image');
            if (removeBtn) {
                removeBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    removeImageFromBatch(image.id);
                });
            }
        });
        
        // Add a secondary drop zone AFTER the container when items exist
        if (!document.getElementById('batchSecondaryDropZone')) {
            const secondaryDropZone = document.createElement('div');
            secondaryDropZone.id = 'batchSecondaryDropZone';
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
                    handleBatchFiles(e.dataTransfer.files);
                }
            });
            
            secondaryDropZone.addEventListener('click', () => {
                batchFileInput.click();
            });
            
            // Append the secondary drop zone AFTER the batch images container
            batchImagesContainer.after(secondaryDropZone);
        }
    }
}

// Clear all batch images
function clearBatchImages() {
    // Clear array
    batchImages.length = 0;
    
    // Update batch status to show empty state
    updateBatchStatus();
}

// Process the batch
async function processBatch() {
    if (batchImages.length === 0) {
        alert('Please add at least one image before processing.');
        return;
    }
    
    // Get selected module
    const batchVideoModuleSelect = document.getElementById('batchVideoModule');
    const selectedModule = batchVideoModuleSelect ? batchVideoModuleSelect.value : 'framepack';
    
    // Get settings from form using our common utility
    const settings = getSettingsFromForm('batch');
    
    console.log('Batch settings:', settings);
    console.log('Processing batch images:', batchImages.length);
    console.log('Selected module:', selectedModule);
    
    // Disable the process button while batch is processing
    batchProcessBtn.disabled = true;
    batchProcessBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> Processing...';
    
    try {
        // Generate a job ID
        const timestamp = Math.floor(Date.now() / 1000);
        const jobId = `${timestamp}_${Math.floor(Math.random() * 1000)}`;
        
        // Prepare segments for job
        const segments = [];
        let includeLastFrame = false;
        
        // Get the "Include last frame as segment" checkbox state for WAN module
        if (selectedModule === 'wan') {
            const includeLastFrameElement = document.getElementById('batchIncludeLastFrame');
            includeLastFrame = includeLastFrameElement && includeLastFrameElement.checked;
        }
        
        // Add each image as a segment
        for (let i = 0; i < batchImages.length; i++) {
            const image = batchImages[i];
            const isLastFrame = (i === batchImages.length - 1);
            
            // Skip the last frame for WAN if we're going to use it as a separate segment
            if (selectedModule === 'wan' && includeLastFrame && isLastFrame) {
                continue;
            }
            
            segments.push({
                image_path: image.dataUrl,
                prompt: image.prompt || '',
                duration: parseFloat(settings.batchDuration) || 3.0,
                use_last_frame: includeLastFrame && isLastFrame
            });
        }
        
        // Add the last frame back in if we're including it as a separate segment
        if (selectedModule === 'wan' && includeLastFrame && batchImages.length > 0) {
            const lastImage = batchImages[batchImages.length - 1];
            segments.push({
                image_path: lastImage.dataUrl,
                prompt: lastImage.prompt || '',
                duration: parseFloat(settings.batchDuration) || 3.0,
                use_last_frame: true
            });
        }
        
        console.log('Batch segments:', segments);
        
        // Create job settings object based on selected module
        const jobSettings = {};
        
        // Special handling for WAN with "include last frame" option
        if (selectedModule === 'wan' && includeLastFrame && batchImages.length > 1) {
            // If the last frame is marked as a segment, we need to create a multi-job setup
            // 1. Create a standard job with all but the last frame
            // 2. Create a second job that just processes the last frame
            showMessage("Multi-frame WAN batch job with last frame included - creating two jobs", "info");
            
            // First, create a WAN job for all frames except the last
            const segmentsWithoutLast = segments.slice(0, -1);
            const mainWanPayload = prepareJobPayload(settings, segmentsWithoutLast, 'wan', 'batch');
            jobSettings.wan = mainWanPayload;
            
            // Create a second job for the last frame only
            const lastSegment = [segments[segments.length - 1]];
            const lastFrameJobId = `${jobId}_lastframe`;
            const lastFramePayload = prepareJobPayload(settings, lastSegment, 'wan', 'batch');
            
            // Save and start both jobs
            try {
                // Save the main job first (but don't start it)
                console.log('Saving main job:', jobId, jobSettings);
                await saveAndProcessJob(jobId, jobSettings, segmentsWithoutLast, 'wan', false);
                
                // Then save and start the last frame job
                console.log('Saving and starting last frame job:', lastFrameJobId, lastFramePayload);
                const lastFrameJobSettings = {
                    wan: lastFramePayload
                };
                await saveAndProcessJob(lastFrameJobId, lastFrameJobSettings, lastSegment, 'wan', true);
                
                // Now start the main job
                console.log('Starting main job:', jobId);
                fetch(`/api/jobs/${jobId}/run`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        console.log('Main job started:', data);
                        showMessage("Batch jobs started successfully", "success");
                        
                        // Clear the batch after successfully saving
                        clearBatchImages();
                    } else {
                        console.error('Failed to start main job:', data);
                        showMessage(`Failed to start main job: ${data.error || 'Unknown error'}`, "error");
                    }
                    
                    // Re-enable the process button
                    batchProcessBtn.disabled = false;
                    batchProcessBtn.innerHTML = '<i class="bi bi-play-fill"></i> Process Batch';
                })
                .catch(error => {
                    console.error('Error starting main job:', error);
                    showMessage(`Error starting main job: ${error.message}`, "error");
                    
                    // Re-enable the process button
                    batchProcessBtn.disabled = false;
                    batchProcessBtn.innerHTML = '<i class="bi bi-play-fill"></i> Process Batch';
                });
                
                return { jobId, lastFrameJobId };
            } catch (error) {
                console.error('Error processing multi-job WAN with last frame:', error);
                showMessage(`Error processing jobs: ${error.message}`, "error");
                
                // Re-enable the process button
                batchProcessBtn.disabled = false;
                batchProcessBtn.innerHTML = '<i class="bi bi-play-fill"></i> Process Batch';
                
                return null;
            }
        } else {
            // Normal job processing
            const modulePayload = prepareJobPayload(settings, segments, selectedModule, 'batch');
            jobSettings[selectedModule] = modulePayload;
            
            // Save and process the job
            console.log('Processing job:', jobId, jobSettings);
            const result = await saveAndProcessJob(jobId, jobSettings, segments, selectedModule);
            
            // Clear the batch after successfully saving
            clearBatchImages();
            
            // Re-enable the process button
            batchProcessBtn.disabled = false;
            batchProcessBtn.innerHTML = '<i class="bi bi-play-fill"></i> Process Batch';
            
            return result;
        }
    } catch (error) {
        console.error('Error processing batch:', error);
        showMessage(`Error processing batch: ${error.message}`, "error");
        
        // Re-enable the process button
        batchProcessBtn.disabled = false;
        batchProcessBtn.innerHTML = '<i class="bi bi-play-fill"></i> Process Batch';
        
        return null;
    }
}

// Function to handle batch module selection and toggle related settings
function handleBatchModuleSelection(event) {
    const selectedModule = event.target.value;
    
    // Get all settings elements
    const commonSettings = [
        'batchGlobalPrompt', 'batchNegativePrompt', 'batchResolution', 'batchFps', 'batchDuration'
    ];
    
    // FramePack-specific settings
    const framepackSettings = [
        'batchAutoCaptionImage', 'batchFaceRestoration', 'batchSteps', 'batchGuidanceScale', 
        'batchUseTeacache', 'batchEnableAdaptiveMemory', 'batchLoraModel', 'batchLoraScale'
    ];
    
    // WAN-specific settings
    const wanSettings = [
        'batchWanSize', 'batchWanFrameNum', 'batchWanSampleSteps', 
        'batchWanSampleShift', 'batchWanSampleGuideScale'
    ];
    
    // Show/hide settings based on selected module
    if (selectedModule === 'wan') {
        // Show WAN-specific settings
        document.getElementById('batchWanSettings').style.display = 'block';
        
        // Hide FramePack-specific settings that don't apply to WAN
        framepackSettings.forEach(setting => {
            const element = document.getElementById(setting);
            if (element) {
                const container = element.closest('.mb-3, .form-check');
                if (container) container.style.display = 'none';
            }
        });
        
        // Make sure common settings are visible
        commonSettings.forEach(setting => {
            const element = document.getElementById(setting);
            if (element) {
                const container = element.closest('.mb-3, .form-check');
                if (container) container.style.display = 'block';
            }
        });
        
        // Hide the WAN task selector as it will be determined automatically
        const wanTaskContainer = document.getElementById('batchWanTask')?.closest('.mb-3');
        if (wanTaskContainer) wanTaskContainer.style.display = 'none';
        
    } else {
        // Hide WAN-specific settings
        document.getElementById('batchWanSettings').style.display = 'none';
        
        // Show FramePack-specific settings
        framepackSettings.forEach(setting => {
            const element = document.getElementById(setting);
            if (element) {
                const container = element.closest('.mb-3, .form-check');
                if (container) container.style.display = 'block';
            }
        });
        
        // Make sure common settings are visible
        commonSettings.forEach(setting => {
            const element = document.getElementById(setting);
            if (element) {
                const container = element.closest('.mb-3, .form-check');
                if (container) container.style.display = 'block';
            }
        });
    }
} 