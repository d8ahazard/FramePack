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
    // Implement batch processing logic here
    console.log('Processing batch of', batchImages.length, 'images');
    
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
        outputFormat: document.getElementById('batchOutputFormat').value
    };
    
    // For now, just log the settings and images
    console.log('Batch settings:', settings);
    console.log('Batch images:', batchImages);
    
    for (const image of batchImages) {
    }
} 