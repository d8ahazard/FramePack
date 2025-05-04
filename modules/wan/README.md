# Wan2.1 Module

## Overview

The Wan2.1, a module implementing Wan2.1 video generation capabilities. It supports:

1. **Image-to-Video (I2V)**: Generate videos from a single input image
2. **First-Last-Frame to Video (FLF2V)**: Generate videos from two images (first and last frames)

The module is designed to automatically detect the appropriate task type based on provided inputs and gracefully handle missing components with fallback implementations.

## Architecture

The module consists of the following key components:

- **module.py**: Main entry point implementing the task detection and processing logic
- **datatypes.py**: Data type definitions for job settings and parameters
- **first_last_frame2video.py**: Implementation of the FLF2V functionality
- **configs/**: Configuration settings for Wan models
- **utils/**: Utility functions for prompt extension, image processing, etc.
- **modules/**: Custom model implementations
- **distributed/**: Distributed training and inference support

## Usage

The module handles tasks through the `process()` function, which takes a `WanJobSettings` instance. The system will automatically determine whether to use I2V or FLF2V based on the provided parameters:

- If `first_frame` and `last_frame` are provided, it will use FLF2V
- If `image` is provided, it will use I2V
- Otherwise, it will use the explicitly specified `task` parameter

### WanJobSettings Parameters

- **Required Parameters**:
  - `job_id`: Unique identifier for the job
  - `prompt`: Text description of the desired video content

- **Task Selection** (one of the following):
  - `image`: Path to input image for I2V generation
  - `first_frame` + `last_frame`: Paths to first and last frames for FLF2V generation
  - `task`: Explicit task type ("i2v-14B" or "flf2v-14B")

- **Common Parameters**:
  - `size`: Output video dimensions in format "width*height" (e.g., "1280*720")
  - `frame_num`: Number of frames to generate (default: 81)
  - `fps`: Frames per second in output video (default: 16)
  - `negative_prompt`: Text description of elements to avoid in generation
  - `base_seed`: Random seed for generation (-1 for random)

- **Sampling Parameters**:
  - `sample_steps`: Number of diffusion steps (default: 50)
  - `sample_guide_scale`: Guidance scale (default: 5.0)
  - `sample_shift`: Flow matching shift parameter (default: 5.0)

- **Prompt Extension**:
  - `use_prompt_extend`: Whether to use prompt extension (default: false)
  - `prompt_extend_method`: Method for prompt extension ("dashscope" or "local_qwen")
  - `prompt_extend_model`: Model to use for prompt extension (optional)
  - `prompt_extend_target_lang`: Target language for prompt extension (default: "zh")

- **Advanced Settings**:
  - `t5_cpu`: Whether to place T5 model on CPU (default: false)
  - `offload_model`: Whether to offload model to CPU during generation (default: true)

## Implementation Notes

### FLF2V Implementation

The First-Last-Frame to Video implementation includes:

1. A robust component loading system with detailed error reporting
2. A fallback implementation that creates a simple interpolation when full implementation components are unavailable
3. Proper handling of different size inputs, automatically resizing the last frame to match the first frame
4. Memory management for efficient operation on consumer hardware

### Automatic Task Detection

The module's `process()` function automatically detects the appropriate task based on provided inputs:

```python
if hasattr(request, 'first_frame') and hasattr(request, 'last_frame') and request.first_frame and request.last_frame:
    task = "flf2v-14B"
elif hasattr(request, 'image') and request.image:
    task = "i2v-14B"
else:
    # Fallback to explicitly requested task
    task = request.task
```

### Error Handling

The module implements comprehensive error handling with:

1. Detailed error messages for missing inputs or components
2. Status updates throughout the generation process
3. Exception handling with proper status updates
4. Path validation for input images

## Model Information

The Wan2.1 models used by this module are from Wan-AI:

- I2V-14B-480P: "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
- I2V-14B-720P: "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
- FLF2V-14B: "Wan-AI/Wan2.1-FLF2V-14B-720P"

The FLF2V model works best with Chinese prompts due to its training data.

## Tips for Best Results

1. **Resolution Selection**:
   - For I2V, use "832*480" for better performance on limited hardware
   - For FLF2V, "1280*720" generally produces the best results

2. **Prompt Design**:
   - Use detailed descriptions for better results
   - For FLF2V, consider using Chinese prompts for best results
   - Enable prompt extension for more detailed generation

3. **Parameter Tuning**:
   - Adjust `sample_shift` based on resolution (3.0 for 480p, 5.0+ for 720p)
   - Increase `sample_guide_scale` for more faithful adherence to prompts
   - For more dynamic motion, try increasing the `sample_shift` parameter 