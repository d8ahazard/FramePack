import argparse
import hashlib
import importlib
import json
import logging
import os
import time

import uvicorn
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import modules
from datatypes.datatypes import UploadResponse, ConnectionManager, EndpointFilter
from handlers.job_queue import job_statuses
from handlers.model import preload_all_models
from handlers.path import thumbnail_path, upload_path, output_path


logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--preload", action="store_true", help="Preload all models at startup")
parser.add_argument("--hf_token", type=str, help="Hugging Face authentication token")
args = parser.parse_args()

print(args)

os.makedirs("static/images", exist_ok=True)

# Create a connection manager instance
manager = ConnectionManager()

# ----------------------------------------
# FastAPI app setup
# ----------------------------------------
app = FastAPI(
    title="FramePack API",
    description="API for image to video generation using FramePack",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Configure logging to suppress job status logs
# Apply the filter to the uvicorn access logger

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory=output_path), name="outputs")

# Keep this for backward compatibility but it's no longer primary access method
app.mount("/uploads", StaticFiles(directory=upload_path), name="uploads")
app.mount("/thumbnails", StaticFiles(directory=thumbnail_path), name="thumbnails")

# Templates setup
templates = Jinja2Templates(directory="templates")




@app.middleware("http")
async def add_response_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


def resolve_upload_path(path):
    """
    Resolves path to a valid file path in the uploads directory.

    Args:
        path: A full server path to an uploaded file

    Returns:
        tuple: (resolved_path, exists) where resolved_path is the full filesystem path
               and exists is a boolean indicating if the file exists
    """
    if not path:
        return None, False

    # Just check if the provided path exists directly
    exists = os.path.isfile(path)

    return path, exists


def cleanup_thumbnail_cache(max_age_days=30):
    """
    Clean up old thumbnails from the cache directory

    Args:
        max_age_days: Maximum age of thumbnails to keep
    """
    try:
        now = time.time()
        count = 0

        for filename in os.listdir(thumbnail_path):
            file_path = os.path.join(thumbnail_path, filename)
            if os.path.isfile(file_path):
                file_age = now - os.path.getmtime(file_path)
                if file_age > max_age_days * 86400:  # Convert days to seconds
                    os.remove(file_path)
                    count += 1

        print(f"Cleaned up {count} old thumbnails")
    except Exception as e:
        print(f"Error cleaning up thumbnails: {e}")


@app.get("/favicon.ico")
async def favicon():
    """Serve the favicon"""
    return FileResponse("static/images/favicon.ico")


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.get("/api")
async def api_docs_redirect():
    """Redirect to API documentation"""
    return RedirectResponse(url="/api/docs")




@app.websocket("/ws/job/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await manager.connect(websocket, job_id)

    # Send initial job status on connection
    try:
        status_obj = job_statuses.get(job_id)
        if status_obj:
            await websocket.send_text(json.dumps(status_obj.to_dict()))
    except Exception as e:
        print(f"Error sending initial job status: {e}")

    try:
        while True:
            # Wait for any client messages (mostly ping/pong)
            data = await websocket.receive_text()
            # Just echo back to confirm connection is alive
            await websocket.send_text(data)
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, job_id)


def register_all_endpoints():
    # Enumerate all modules in handlers. If they have a register_api_endpoints function, call it
    import handlers
    import pkgutil

    def try_register(module_name):
        try:
            module = __import__(module_name, fromlist=[''])
            if hasattr(module, "register_api_endpoints"):
                print(f"Registering API endpoints from {module_name}")
                module.register_api_endpoints(app)
        except Exception as e:
            print(f"Error registering API endpoints from {module_name}: {e}")

    for _, module_name, _ in pkgutil.walk_packages(handlers.__path__, handlers.__name__ + "."):
        try_register(module_name)

    # Check every subfolder in modules, find any API classes with register_api_endpoints, call them
    for _, module_name, _ in pkgutil.walk_packages(modules.__path__, modules.__name__ + "."):
        # Need to find the api module in the subfolder
        api_module_name = module_name + ".api"
        # Check if the api module is a valid import
        try:
            importlib.import_module(api_module_name)
        except ImportError:
            continue
        try_register(api_module_name)


# Register job queue routes

# Run cleanup on startup with a max age of 30 days

# Run the application
if __name__ == "__main__":
    # Preload models if requested
    register_all_endpoints()
    cleanup_thumbnail_cache(30)

    if args.preload:
        model_paths = preload_all_models(use_auth_token=args.hf_token)

    # Start the server
    uvicorn.run(app, host=args.host, port=args.port)
