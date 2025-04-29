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
    batchDropArea = document.getElementById('batchDropArea');
    batchFileInput = document.getElementById('batchFileInput');
    batchImagesContainer = document.getElementById('batchImagesContainer');
    batchProcessBtn = document.getElementById('batchProcessBtn');
    uploadBatchImagesBtn = document.getElementById('uploadBatchImagesBtn');
    clearBatchBtn = document.getElementById('clearBatchBtn');
    batchSettingsForm = document.getElementById('batchSettingsForm');
    
    // Set up event listeners
    if (batchDropArea) {
        batchDropArea.addEventListener('dragover', handleDragOver);
        batchDropArea.addEventListener('dragleave', handleDragLeave);
        batchDropArea.addEventListener('drop', handleBatchDrop);
        batchDropArea.addEventListener('click', () => batchFileInput.click());
    }
    
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
}

// Event Handlers
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    this.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    this.classList.remove('drag-over');
}

function handleBatchDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    this.classList.remove('drag-over');
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleBatchFiles(e.dataTransfer.files);
    }
}

function handleBatchFileSelect(e) {
    if (e.target.files && e.target.files.length > 0) {
        handleBatchFiles(e.target.files);
    }
}

// Process the selected files for batch
function handleBatchFiles(files) {
    const imageFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
    
    if (imageFiles.length === 0) {
        // Show error or notification
        console.error('No valid image files selected');
        return;
    }
    
    // Clear the alert if it exists
    const alertElement = batchImagesContainer.querySelector('.alert');
    if (alertElement) {
        alertElement.remove();
    }
    
    // Add image files to batch
    imageFiles.forEach(file => {
        addImageToBatch(file);
    });
    
    // Enable process and clear buttons
    batchProcessBtn.disabled = batchImages.length === 0;
    clearBatchBtn.disabled = batchImages.length === 0;
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
        batchProcessBtn.disabled = false;
        clearBatchBtn.disabled = false;
    };
    
    reader.readAsDataURL(file);
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
    
    // Disable buttons if no images
    batchProcessBtn.disabled = batchImages.length === 0;
    clearBatchBtn.disabled = batchImages.length === 0;
    
    // Show info message if no images
    if (batchImages.length === 0) {
        batchImagesContainer.innerHTML = `
            <div class="alert alert-info">
                <i class="bi bi-info-circle me-2"></i> Upload one or more images to process in batch. Each image will be processed with the same settings.
            </div>
        `;
    }
}

// Clear all batch images
function clearBatchImages() {
    // Clear array
    batchImages.length = 0;
    
    // Clear DOM
    batchImagesContainer.innerHTML = `
        <div class="alert alert-info">
            <i class="bi bi-info-circle me-2"></i> Upload one or more images to process in batch. Each image will be processed with the same settings.
        </div>
    `;
    
    // Disable buttons
    batchProcessBtn.disabled = true;
    clearBatchBtn.disabled = true;
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
                    job_settings: {'framepack': jobPayload},
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