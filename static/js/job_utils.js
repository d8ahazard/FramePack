// Job Utilities Module for FramePack
// Shared job-related functions used by editor and batch modules

import { showMessage } from './common.js';
// Remove direct import to avoid circular dependency
// import { runJob } from './job_queue.js';

// Track jobs that are currently in the saving process
const savingJobs = new Set();

/**
 * Upload a file to the server
 * @param {File} file - The file to upload
 * @returns {Promise<string>} - Promise resolving to the server file path
 */
export function uploadFileToServer(file) {
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

/**
 * Prepare job payload based on settings and module type
 * @param {Object} settings - Form settings
 * @param {Array} segments - Array of segment objects with image_path, prompt, and duration
 * @param {string} moduleType - 'framepack' or 'wan'
 * @param {boolean} includeLastFrame - Whether to include the last frame for FramePack
 * @returns {Object} - Job payload for the selected module
 */
export function prepareJobPayload(settings, segments, moduleType = 'framepack', includeLastFrame = false) {
    // If WAN module is selected, prepare a WAN payload
    if (moduleType === 'wan') {
        return prepareWanJobPayload(settings, segments);
    }
    
    // Otherwise, prepare a FramePack payload (default)
    return prepareFramePackJobPayload(settings, segments, includeLastFrame);
}

/**
 * Prepare FramePack job payload
 * @param {Object} settings - Form settings
 * @param {Array} segments - Array of segment objects with image_path, prompt, and duration
 * @param {boolean} includeLastFrame - Whether to include the last frame
 * @returns {Object} - FramePack job payload
 */
function prepareFramePackJobPayload(settings, segments, includeLastFrame = false) {
    if (!segments || segments.length === 0) {
        throw new Error('No segments provided for job payload');
    }
    
    return {
        global_prompt: settings.globalPrompt || "",
        negative_prompt: settings.negativePrompt || "",
        segments: segments,
        seed: Math.floor(Math.random() * 100000),
        steps: parseInt(settings.steps) || 25,
        guidance_scale: parseFloat(settings.guidanceScale) || 10.0,
        use_teacache: settings.useTeacache !== false,
        enable_adaptive_memory: settings.enableAdaptiveMemory !== false,
        resolution: parseInt(settings.resolution) || 640,
        mp4_crf: 16,
        gpu_memory_preservation: 6.0,
        include_last_frame: includeLastFrame,
        auto_prompt: settings.autoCaptionImage || false,
        lora_model: settings.loraModel || null,
        lora_scale: parseFloat(settings.loraScale) || 1.0,
        fps: parseInt(settings.fps) || 30
    };
}

/**
 * Prepare WAN job payload
 * @param {Object} settings - Form settings
 * @param {Array} segments - Array of segment objects with image_path
 * @returns {Object} - WAN job payload
 */
function prepareWanJobPayload(settings, segments) {
    if (!segments || segments.length === 0) {
        throw new Error('No segments provided for job payload');
    }
    
    // Automatically determine task based on number of segments
    // - If only one segment: use i2v-14B (Image to Video)
    // - If two or more segments: use flf2v-14B (First-Last Frame to Video)
    const task = segments.length === 1 ? "i2v-14B" : "flf2v-14B";
    console.log(`Auto-selected WAN task: ${task} based on ${segments.length} segments`);
    
    // Create the basic payload object
    const payload = {
        prompt: settings.globalPrompt || "",
        negative_prompt: settings.negativePrompt || "",
        task: task, // Use auto-determined task
        size: settings.wanSize || "1280*720",
        frame_num: parseInt(settings.wanFrameNum) || 81,
        sample_solver: "unipc",
        sample_steps: parseInt(settings.wanSampleSteps) || 40,
        sample_shift: parseFloat(settings.wanSampleShift) || 5.0,
        sample_guide_scale: parseFloat(settings.wanSampleGuideScale) || 5.0,
        base_seed: Math.floor(Math.random() * 100000),
        fps: parseInt(settings.fps) || 30,
        segments: segments // Pass all segments to the backend for consistent handling
    };
    
    // For backwards compatibility, also include the explicit image paths
    // based on the task type (the backend will prioritize using segments)
    if (task === 'i2v-14B') {
        // For i2v-14B, use the first image
        payload.image = segments[0].image_path;
    } else if (task === 'flf2v-14B') {
        // For flf2v-14B, use the first and last images
        payload.first_frame = segments[0].image_path;
        payload.last_frame = segments[segments.length - 1].image_path;
    }
    
    return payload;
}

/**
 * Get settings from form elements for either editor or batch
 * @param {string} prefix - Prefix for form element IDs ('batch' or '')
 * @returns {Object} - Object containing all form settings
 */
export function getSettingsFromForm(prefix = '') {
    // Create prefix with separator if provided
    const p = prefix ? prefix + ((prefix.endsWith('-') || prefix.length === 0) ? '' : '-') : '';
    
    return {
        // Common settings
        autoCaptionImage: document.getElementById(`${p}autoCaptionImage`)?.checked || false,
        globalPrompt: document.getElementById(`${p}globalPrompt`)?.value || "",
        negativePrompt: document.getElementById(`${p}negativePrompt`)?.value || "",
        resolution: document.getElementById(`${p}resolution`)?.value || "640",
        steps: document.getElementById(`${p}steps`)?.value || "25",
        guidanceScale: document.getElementById(`${p}guidanceScale`)?.value || "10.0",
        useTeacache: document.getElementById(`${p}useTeacache`)?.checked !== false,
        enableAdaptiveMemory: document.getElementById(`${p}enableAdaptiveMemory`)?.checked !== false,
        outputFormat: document.getElementById(`${p}outputFormat`)?.value || "mp4",
        duration: document.getElementById(`${p}duration`)?.value || "3.0",
        loraModel: document.getElementById(`${p}loraModel`)?.value || "",
        loraScale: document.getElementById(`${p}loraScale`)?.value || "1.0",
        fps: document.getElementById(`${p}fps`)?.value || "30",
        
        // WAN specific settings
        wanTask: document.getElementById(`${p}wanTask`)?.value || "i2v-14B",
        wanSize: document.getElementById(`${p}wanSize`)?.value || "1280*720",
        wanFrameNum: document.getElementById(`${p}wanFrameNum`)?.value || "81",
        wanSampleSteps: document.getElementById(`${p}wanSampleSteps`)?.value || "40",
        wanSampleShift: document.getElementById(`${p}wanSampleShift`)?.value || "5.0",
        wanSampleGuideScale: document.getElementById(`${p}wanSampleGuideScale`)?.value || "5.0"
    };
}

/**
 * Save and run/queue a job
 * @param {string} jobId - Job ID
 * @param {Object} jobSettings - Job settings object with module-specific settings
 * @param {Array} segments - Array of segment paths
 * @param {string} moduleType - Module type ('framepack' or 'wan')
 * @param {boolean} startJob - Whether to start the job immediately (true) or just save it (false)
 * @returns {Promise<Object>} - Promise resolving to job result
 */
export async function saveAndProcessJob(jobId, jobSettings, segments, moduleType = 'framepack', startJob = true) {
    // Validate inputs
    if (!jobId) throw new Error('Job ID is required');
    if (!jobSettings) throw new Error('Job settings are required');
    if (!segments || segments.length === 0) throw new Error('At least one segment is required');
    
    // Check if we're already saving this job to prevent duplicate submissions
    if (savingJobs.has(jobId)) {
        console.log(`Job ${jobId} is already being saved, ignoring duplicate request`);
        return { success: true, jobId, message: 'Duplicate job save request ignored' };
    }
    
    // Mark this job as being saved
    savingJobs.add(jobId);
    
    try {
        const timestamp = Math.floor(Date.now() / 1000);
        
        // Create save payload
        const savePayload = {
            job_id: jobId,
            status: "saved",
            progress: 0,
            message: `Job saved (${moduleType})`,
            result_video: "",
            segments: segments,
            is_valid: true,
            missing_images: [],
            job_settings: jobSettings,
            queue_position: -1,
            created_timestamp: timestamp
        };
        
        // Save the job
        const saveResponse = await fetch(`/api/save_job/${jobId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(savePayload)
        });
        
        if (!saveResponse.ok) {
            const errorText = await saveResponse.text();
            throw new Error(`Failed to save job: ${errorText}`);
        }
        
        const saveResult = await saveResponse.json();
        console.log('Job saved:', saveResult);
        
        // Start the job if requested
        if (startJob) {
            try {
                // Make direct API call to run the job
                const runResponse = await fetch(`/api/run_job/${jobId}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        module: moduleType
                    })
                });
                
                if (!runResponse.ok) {
                    const errorText = await runResponse.text();
                    throw new Error(`Failed to run job: ${errorText}`);
                }
                
                const runResult = await runResponse.json();
                console.log('Server response for run job:', runResult);
                
                if (!runResult.success) {
                    throw new Error(runResult.message || 'Server failed to start the job');
                }
                
                return { success: true, jobId, ...runResult };
            } catch (err) {
                console.error('Error running job:', err);
                return { success: false, error: err.message, jobId };
            }
        }
        
        return { success: true, jobId };
    } catch (error) {
        console.error('Error saving job:', error);
        throw error;
    } finally {
        // Always remove from tracking when done
        savingJobs.delete(jobId);
    }
} 