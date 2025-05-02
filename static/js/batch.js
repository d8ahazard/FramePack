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
function processBatch() {
    if (batchImages.length === 0) {
        alert('Please add at least one image before processing.');
        return;
    }
    
    // Get settings from form
    const settings = {
        autoCaptionImage: document.getElementById('batchAutoCaptionImage').checked,
        globalPrompt: document.getElementById('batchGlobalPrompt').value,
        negativePrompt: document.getElementById('batchNegativePrompt').value,
        resolution: document.getElementById('batchResolution').value,
        steps: document.getElementById('batchSteps').value,
        guidanceScale: document.getElementById('batchGuidanceScale').value,
        useTeacache: document.getElementById('batchUseTeacache').checked,
        enableAdaptiveMemory: document.getElementById('batchEnableAdaptiveMemory').checked,
        outputFormat: document.getElementById('batchOutputFormat').value,
        duration: document.getElementById('batchDuration').value
    };
    
    console.log('Batch settings:', settings);
    console.log('Processing batch images:', batchImages.length);
    
    // Disable the process button while batch is processing
    batchProcessBtn.disabled = true;
    batchProcessBtn.innerHTML = '<i class="bi bi-hourglass-split me-2"></i> Uploading...';
    
    // First, upload all images to the server
    const uploadPromises = batchImages.map(image => {
        // Return a promise for each file upload
        return uploadFileToServer(image.file)
            .then(serverPath => {
                // Store the server path with the image entry for later use
                return {
                    id: image.id,
                    file: image.file,
                    serverPath: serverPath
                };
            });
    });
    
    // Process all uploads
    Promise.all(uploadPromises)
        .then(uploadedImages => {
            console.log('All batch images uploaded successfully:', uploadedImages);
            
            // Update status message
            batchProcessBtn.innerHTML = '<i class="bi bi-hourglass-split me-2"></i> Creating Jobs...';
            
            // Process each uploaded image as a separate job
            const jobPromises = uploadedImages.map(image => {
                // Generate a job ID
                const timestamp = Math.floor(Date.now() / 1000);
                const jobId = `batch_${timestamp}_${Math.floor(Math.random() * 1000)}`;
                
                // Each image is a single segment in the job
                const imageDuration = parseFloat(settings.duration) || 3.0;
                
                // Create the job payload similar to editor.js's prepareJobPayload
                const jobPayload = {
                    global_prompt: settings.globalPrompt || "",
                    negative_prompt: settings.negativePrompt || "",
                    segments: [
                        {
                            // Now use the server path instead of data URL
                            image_path: image.serverPath,
                            prompt: "",  // Individual image prompts not supported in batch mode
                            duration: imageDuration
                        }
                    ],
                    seed: Math.floor(Math.random() * 100000),
                    steps: parseInt(settings.steps) || 25,
                    guidance_scale: parseFloat(settings.guidanceScale) || 10.0,
                    use_teacache: settings.useTeacache,
                    enable_adaptive_memory: settings.enableAdaptiveMemory,
                    resolution: parseInt(settings.resolution) || 640,
                    mp4_crf: 16,
                    gpu_memory_preservation: 6.0,
                    include_last_frame: false,
                    auto_prompt: settings.autoCaptionImage
                };

                let jobSettings = {
                    framepack: jobPayload
                }

                if (document.getElementById('batchFaceRestoration').checked) {
                    jobSettings.facefusion = {
                        source_image_path: image.serverPath,
                        target_video_path: `${jobId}_final.mp4`,
                        output_path: `${jobId}_final_restored.mp4`
                    }
                }
                // Create save payload similar to startGeneration in editor.js
                const savePayload = {
                    job_id: jobId,
                    status: "saved",
                    progress: 0,
                    message: "Batch job saved",
                    result_video: "",
                    segments: [image.serverPath], // Now use actual server path for segments
                    is_valid: true,
                    missing_images: [],
                    job_settings: jobSettings,
                    queue_position: -1,
                    created_timestamp: timestamp
                };
                
                // Return a promise for saving and running the job
                return submitBatchJob(jobId, savePayload);
            });
            
            // Wait for all jobs to be submitted
            return Promise.all(jobPromises);
        })
        .then(jobResults => {
            console.log('All batch jobs submitted:', jobResults);
            
            // Show feedback that batch has been submitted
            alert(`Submitted ${batchImages.length} image(s) for processing. You can monitor progress in the Job Queue tab.`);
            
            // Switch to the job queue tab to show progress
            const queueTab = document.getElementById('queue-tab');
            if (queueTab) {
                bootstrap.Tab.getOrCreateInstance(queueTab).show();
            }
            
            // Try to refresh the job queue
            try {
                import('./job_queue.js').then(jobQueueModule => {
                    if (typeof jobQueueModule.loadJobQueue === 'function') {
                        jobQueueModule.loadJobQueue();
                    }
                });
            } catch (err) {
                console.error('Error refreshing job queue:', err);
            }
            
            // Clear the batch images since they've been processed
            clearBatchImages();
        })
        .catch(error => {
            console.error('Error processing batch:', error);
            alert('Error processing batch: ' + error.message);
        })
        .finally(() => {
            // Re-enable the process button
            batchProcessBtn.disabled = false;
            batchProcessBtn.innerHTML = 'Process Batch';
        });
}

// Helper function to upload a file to the server
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

// Helper function to submit a batch job
function submitBatchJob(jobId, savePayload) {
    return new Promise((resolve, reject) => {
        // Save the job
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
            console.log('Batch job saved:', data);
            
            try {
                // Run the job
                runJob(jobId, true); // Pass true to indicate this is a batch job
                console.log(`Batch job ${jobId} started using job_queue.runJob`);
                resolve(jobId); // Resolve with the job ID
            } catch (err) {
                console.error('Error running batch job:', err);
                reject(err);
            }
        })
        .catch(error => {
            console.error('Error submitting batch job:', error);
            reject(error);
        });
    });
} 