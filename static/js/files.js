// Files module for FramePack
// Handles file uploads and processing

import { 
    elements, 
    keepCurrentImage,
    showMessage
} from './common.js';

import { uploadModal } from './editor.js';

// Initialize module variables
let selectedFiles = [];

// Module initialization function
function initFiles() {
    console.log('Files module initialized');
    
    // Initialize upload modal
    const uploadModalElement = document.getElementById('uploadImagesModal');
    if (uploadModalElement) {
        uploadModal = new bootstrap.Modal(uploadModalElement);
    }
    
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
    filesArray.forEach(file => {
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
            
            // Create thumbnail
            const thumbnailHtml = `
                <div class="col-4 col-md-3 mb-3">
                    <div class="card">
                        <img src="${imgSrc}" class="card-img-top" alt="${fileName}">
                        <div class="card-body p-2">
                            <p class="card-text small text-muted text-truncate">${fileName}</p>
                            <div class="input-group input-group-sm">
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
            const allDurationInputs = elements.imageUploadContainer.querySelectorAll('.image-duration');
            const lastInput = allDurationInputs[allDurationInputs.length - 1];
            if (lastInput) {
                lastInput.addEventListener('change', (e) => {
                    const index = Array.from(allDurationInputs).indexOf(e.target);
                    if (index >= 0 && index < selectedFiles.length) {
                        selectedFiles[index].duration = parseFloat(e.target.value);
                    }
                });
            }
            
            console.log(`Processed file: ${fileName}, total files: ${selectedFiles.length}`);
        };
        
        reader.readAsDataURL(file);
    });
    
    // The upload modal will be shown by the caller
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
                const serverPath = data.path;
                
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

// Export module functions
export {
    initFiles,
    handleDragOver,
    handleDragLeave,
    handleFileDrop,
    handleFileSelect,
    triggerFileInput,
    processSelectedFiles,
    selectedFiles,
    uploadFileToServer,
    showUploadModal,
    clearSelectedFiles
};
