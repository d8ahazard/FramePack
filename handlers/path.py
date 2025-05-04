import os
import sys
import logging

logger = logging.getLogger(__name__)

# Use infer.py as the reference file instead of app.py
app_path = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__))))
model_path = os.path.join(app_path, "models")
output_path = os.path.join(app_path, "outputs")
cache_path = os.path.join(app_path, ".cache")
thumbnail_path = os.path.join(app_path, "thumbnails")
job_path = os.path.join(app_path, "jobs")
upload_path = os.path.join(app_path, "uploads")
lora_path = os.path.join(model_path, "loras")

# Ensure the project root (app_path) is in the Python path first
if app_path not in sys.path:
    sys.path.insert(0, app_path)

# Create all required directories
for path in [model_path, output_path, cache_path, job_path, upload_path, thumbnail_path, lora_path]:
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
    except OSError as e:
        pass
