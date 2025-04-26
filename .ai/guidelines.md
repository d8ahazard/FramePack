# FramePack Development Guidelines

## Project Overview
FramePack is an image-to-video generation tool that uses Hunyuan's AI model to create smooth video transitions between images.

## Tech Stack
- **Python 3.8+**: Core programming language
- **PyTorch/CUDA**: Deep learning framework for model operations
- **Diffusers**: Library for diffusion models (HunyuanVideo)
- **Transformers**: Hugging Face models (CLIP, LLaMA, SiglipVision)
- **FastAPI**: Backend API framework
- **FFmpeg**: Video processing and concatenation

## Project Structure
```
FramePack/
├── modules/            # Core functionality modules
│   └── framepack/      # Main video generation module
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

### Job Settings
- Every process function should have one `JobSettings` class (e.g., `FramePackJobSettings`)
- This class MUST inherit from `ModuleJobSettings`
- Required fields:
  - `job_id`: Unique identifier for the job
  - `segments`: List of `SegmentConfig` objects
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
   - Keep model pipelines in the `modules/framepack/diffusers_helper/` directory
   - Separate UI logic from generation logic

3. **Error Handling**:
   - Always update job status on failures
   - Use thorough exception handling and logging

4. **Performance**:
   - Batch operations where possible
   - Use PyTorch's memory optimization features
   - Consider precision options (bf16/fp16) for memory savings

## Testing
Run manual testing through the web interface at http://localhost:8000 