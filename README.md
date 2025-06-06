# FramePack+ - Image to Video Generation

FramePack+ is a powerful tool that converts images into videos with smooth transitions using various AI models, including Hunyuan's image-to-video and Wan2.1's models.

## Features

- **Timeline-based interface**: Upload multiple images and organize them in a timeline
- **Individual segment prompts**: Set unique text prompts for each segment
- **Customizable durations**: Control the length of each segment
- **Real-time progress tracking**: Monitor the generation process with live preview updates
- **Advanced settings**: Fine-tune the generation process with detailed parameters
- **Multiple Video Generation Modes**:
  - **Image-to-Video (I2V)**: Generate videos from single images using Hunyuan or Wan2.1
  - **First-Last-Frame to Video (FLF2V)**: Generate videos between two keyframes using Wan2.1

## Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU with at least 8GB VRAM (16GB+ recommended)
- FFmpeg installed and available in your system PATH

### Installation

1. Clone this repository:
   ```
   git clone https://your-repo-url/FramePack.git
   cd FramePack
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create required directories:
   ```
   mkdir -p static/css static/js templates uploads outputs
   ```

## Usage

1. Start the server:
   ```
   python infer.py --host 0.0.0.0 --port 8000
   ```
   
   Optional arguments:
   - `--preload`: Preload all models at startup
   - `--hf_token`: Provide a Hugging Face authentication token

2. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```

3. Using the interface:
   - Enter a global prompt that describes the overall video
   - Click "Add Frame" to upload images to the timeline
   - For each image, you can specify a custom prompt and duration
   - Arrange your images in the desired sequence
   - Adjust generation settings (steps, guidance scale, etc.)
   - Click "Generate Video" to start the process
   - Monitor progress and preview the generation in real-time
   - Download the final video when complete

## Generation Parameters

- **Global Prompt**: Text description that guides the overall video generation
- **Negative Prompt**: Features to avoid in the generation
- **Seed**: Random seed for reproducible results
- **Steps**: Number of diffusion steps (higher = better quality but slower)
- **Guidance Scale**: How closely to follow the prompt (higher = more faithful to prompt)
- **Resolution**: Output video resolution (higher = better quality but slower)
- **TeaCache**: Speed optimization (faster but slightly lower detail)
- **Adaptive Memory Management**: Better memory handling for longer videos

## Available Models

### Hunyuan Image-to-Video
The default model that creates smooth transitions between images.

### Wan2.1 Models
FramePack now supports Wan2.1 models which provide additional capabilities:

1. **Image-to-Video (I2V)**: Generate videos from a single image, similar to Hunyuan but with Wan2.1's unique visual quality.

2. **First-Last-Frame to Video (FLF2V)**: Generate a video that smoothly transitions between two keyframes. This is especially useful for:
   - Creating controlled camera movements between two points
   - Animating a scene with defined start and end states
   - Generating videos where precise beginning and ending frames are important

The system will automatically detect which model to use based on the inputs provided:
- If a single image is provided, it will use I2V
- If both first and last frames are provided, it will use FLF2V

## Tips for Best Results

1. **Image Selection**:
   - Choose images with similar compositions for smoother transitions
   - Images with clear subjects work best
   - Avoid overly complex or busy scenes

2. **Prompts**:
   - Be specific but concise in your descriptions
   - Mention movements or actions you want to see
   - Include style descriptions for more control

3. **Parameters**:
   - Start with default settings, then experiment
   - Higher guidance scale (10-15) for more faithful prompt following
   - 25-30 steps provides a good balance between quality and speed

4. **Using FLF2V**:
   - Ensure first and last frames have similar content but different positions
   - For optimal results with Wan2.1 FLF2V, consider using Chinese language prompts

## Auto-LLM Captioning

FramePack supports automatic caption generation using various LLM providers:

1. **API Configuration**:
   - Edit the `apikeys.json` file.
   - Populate the API key for your preferred provider (OpenAI, Anthropic, DeepSeek, Gemini, Groq, or OpenWebUI)
   - Alternatively, set environment variables using the pattern: `PROVIDER_API_KEY` (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
   - For OpenWebUI users, you can specify the LLM endpoint URL with the `OPENWEBUI_API_URL` environment variable

2. **Using Auto-Captioning**:
   - Enable the auto-captioning feature in the interface (Only visible when you have provided a valid API key.)
   - The system will analyze your images and generate appropriate prompts
   - The quality of captions may depend on the LLM provider used

## Troubleshooting

- **Out of Memory Errors**: Reduce resolution or disable adaptive memory management
- **Black Frames**: Try increasing the MP4 compression quality
- **Poor Transitions**: Use more similar images or increase steps

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on Tencent's Hunyuan Video model
- Uses the lllyasviel/FramePackI2V_HY model for interpolation
- Wan2.1 models from Wan-AI
- Frontend built with Bootstrap 5

## FramePack+WAN Integration Improvements (June 2025)

- **Input Validation:** The UI and backend now enforce required fields for each mode. FramePack mode requires an initial image and non-empty prompts; WAN mode allows text-only jobs.
- **Multi-Prompt Support:** Prompts can be provided as a list for multi-segment videos. The backend and UI both support this format.
- **Optional Fields:** Fields like final image are now truly optional and handled gracefully in both UI and backend.
- **User-Friendly Errors:** Validation errors are now mapped to clear, actionable messages in the UI, highlighting the specific field to fix.
- **Mode Awareness:** The UI dynamically shows/hides required fields based on whether FramePack or WAN mode is selected.
- **Robustness:** Improved error handling, job queue management, and progress feedback for a smoother user experience.

See the audit summary for more details on these changes.
