# FramePack Development Guidelines

## Project Overview
FramePack is an image-to-video generation tool that uses various AI models including Hunyuan's AI model and Wan2.1 to create smooth video transitions.

## Tech Stack
- **Python 3.8+**: Core programming language
- **PyTorch/CUDA**: Deep learning framework for model operations
- **Diffusers**: Library for diffusion models (HunyuanVideo, Wan2.1)
- **Transformers**: Hugging Face models (CLIP, LLaMA, SiglipVision, T5)
- **FastAPI**: Backend API framework
- **FFmpeg**: Video processing and concatenation

## Project Structure
```
FramePack/
├── modules/            # Core functionality modules
│   ├── framepack/      # Main Hunyuan video generation module
│   └── wan/            # Wan2.1 video generation module
├── handlers/           # Various service handlers
│   ├── job_queue.py    # Job queue management
│   ├── model.py        # Model loading/management
│   ├── path.py         # Path utilities
│   └── vram.py         # VRAM/memory management
├── datatypes/          # Data type definitions
├── static/             # Frontend static assets
├── templates/          # HTML templates
├── uploads/            # Uploaded user images
├── outputs/            # Generated video outputs
├── models/             # Downloaded model files
├── jobs/               # Job data storage
└── infer.py            # Main application entry point
```

## Development Workflow

### Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Server
```bash
python infer.py --host 0.0.0.0 --port 8000
```

## Code Architecture Requirements

### Handlers
- Every handler MUST implement a `register_api_endpoints` function
- These endpoints are auto-detected and registered in the main application
- Handlers should focus on specific functional areas (jobs, models, etc.)

### Modules
- Every module MUST contain:
  - `datatypes.py` file defining the request structures
  - `module.py` file with a `process()` function as the entry point
- The `process()` function is the standard interface for calling the module

### Module Specifics

#### FramePack Module (Hunyuan)
- Handles image-to-video generation using Hunyuan's model
- Supports timeline-based generation with multiple segments

#### Wan Module
- Supports multiple Wan2.1 models:
  - **Image-to-Video (I2V)** - Generates video from a single image
  - **First-Last-Frame to Video (FLF2V)** - Generates video from first and last frames
- Automatically detects the appropriate task type based on input parameters
- Includes fallback mechanisms for when components are unavailable
- Supports prompt extension for better generation results

### Job Settings
- Every process function should have one `JobSettings` class (e.g., `FramePackJobSettings`, `WanJobSettings`)
- This class MUST inherit from `ModuleJobSettings`
- Required fields:
  - `job_id`: Unique identifier for the job
  - Module-specific required fields (see respective module documentation)
- Other parameters are optional and module-specific

### Memory Management
- The system uses adaptive memory management to handle models efficiently
- Models are loaded/unloaded based on VRAM availability
- `high_vram` mode keeps models in memory for faster processing

### Best Practices
1. **Model Handling**: 
   - Use the existing model loading/unloading infrastructure
   - Check memory after operations with `get_cuda_free_memory_gb()`

2. **Code Organization**:
   - Keep model pipelines in the appropriate module directories
   - Separate UI logic from generation logic

3. **Error Handling**:
   - Always update job status on failures
   - Use thorough exception handling and logging

4. **Performance**:
   - Batch operations where possible
   - Use PyTorch's memory optimization features
   - Consider precision options (bf16/fp16) for memory savings

5. **Module Development**:
   - Ensure graceful fallbacks for missing components
   - Maintain consistent input validation
   - Use proper status updates throughout processing
   - Implement clean path handling for inputs/outputs

## Testing
Run manual testing through the web interface at http://localhost:8000 