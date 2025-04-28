import argparse
import asyncio
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from datatypes.datatypes import EndpointFilter
from handlers.file import cleanup_thumbnail_cache
from handlers.job_queue import clear_running_jobs
from handlers.model import preload_all_models
from handlers.path import thumbnail_path, upload_path, output_path, app_path
from handlers.socket import process_broadcasts

# Set default logging level to INFO
logging.basicConfig(level=logging.INFO)

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--preload", action="store_true", help="Preload all models at startup")
parser.add_argument("--hf_token", type=str, help="Hugging Face authentication token")
args = parser.parse_args()

print(args)

os.makedirs("static/images", exist_ok=True)


# ----------------------------------------
# FastAPI app setup
# ----------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    # Start the broadcast processing task
    asyncio.create_task(process_broadcasts())
    # Run cleanup on startup with a max age of 30 days
    cleanup_thumbnail_cache(30)
    clear_running_jobs()
    yield
    # Run cleanup on startup with a max age of 30 days
    cleanup_thumbnail_cache(30)
    clear_running_jobs()


app = FastAPI(
    title="FramePack API",
    description="API for image to video generation using FramePack",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
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

static_path = os.path.join(app_path, "static")
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

    # # Check every subfolder in modules, find any API classes with register_api_endpoints, call them
    # for _, module_name, _ in pkgutil.walk_packages(modules.__path__, modules.__name__ + "."):
    #     # Need to find the api module in the subfolder
    #     api_module_name = module_name + ".api"
    #     # Check if the api module is a valid import
    #     try:
    #         importlib.import_module(api_module_name)
    #     except ImportError:
    #         continue
    #     try_register(api_module_name)




# Run the application
if __name__ == "__main__":
    # Register API endpoints
    register_all_endpoints()

    if args.preload:
        model_paths = preload_all_models(use_auth_token=args.hf_token)

    # Start the server
    uvicorn.run(app, host=args.host, port=args.port)
