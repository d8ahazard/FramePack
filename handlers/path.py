import os

app_path = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py")))
model_path = os.path.join(app_path, "models")
output_path = os.path.join(app_path, "outputs")
cache_path = os.path.join(app_path, ".cache")
thumbnail_path = os.path.join(app_path, "thumbnails")
job_path = os.path.join(app_path, "jobs")
upload_path = os.path.join(app_path, "uploads")

for path in [model_path, output_path, cache_path, job_path, upload_path, thumbnail_path]:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        pass
