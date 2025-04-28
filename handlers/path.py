import os
import sys
import logging

logger = logging.getLogger(__name__)

# Use infer.py as the reference file instead of app.py
app_path = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "infer.py")))
model_path = os.path.join(app_path, "models")
output_path = os.path.join(app_path, "outputs")
cache_path = os.path.join(app_path, ".cache")
thumbnail_path = os.path.join(app_path, "thumbnails")
job_path = os.path.join(app_path, "jobs")
upload_path = os.path.join(app_path, "uploads")

# Ensure the project root (app_path) is in Python path first
if app_path not in sys.path:
    sys.path.insert(0, app_path)

modules_path = os.path.join(app_path, "modules")
# Properly use os.walk to find all subdirectories in the modules path
for root, dirs, files in os.walk(modules_path):
    for dir_name in dirs:
        module_path = os.path.join(root, dir_name)
        if os.path.isdir(module_path) and not module_path.endswith("__pycache__"):
            print(f"Adding module path: {module_path}")
            sys.path.insert(0, module_path)

# Create all required directories
for path in [model_path, output_path, cache_path, job_path, upload_path, thumbnail_path]:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        pass
