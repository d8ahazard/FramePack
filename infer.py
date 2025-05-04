import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
import shutil
import socket

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

startup_events = []
shutdown_events = []
background_tasks = set()
server_should_exit = False

os.makedirs("static/images", exist_ok=True)
# If apikeys.json doesn't exist, copy apikeys_sample.json to it
if not os.path.exists("apikeys.json"):
    shutil.copy("apikeys_sample.json", "apikeys.json")


# Helper to keep track of background tasks
def create_background_task(coroutine):
    """Create a background task that automatically removes itself when done"""
    task = asyncio.create_task(coroutine)
    background_tasks.add(task)
    
    # Add a callback to remove the task when done
    def _remove_task(_):
        background_tasks.discard(task)
    
    task.add_done_callback(_remove_task)
    return task


# Graceful shutdown handler
async def shutdown(signal_type=None):
    """Cleanup tasks and perform graceful shutdown"""
    global server_should_exit
    server_should_exit = True
    
    if signal_type:
        logging.info(f"Received exit signal {signal_type.name}")
    
    logging.info("Cancelling all running tasks...")
    
    # Cancel our tracked background tasks first
    bg_tasks = list(background_tasks)
    if bg_tasks:
        logging.info(f"Cancelling {len(bg_tasks)} tracked background tasks...")
        for task in bg_tasks:
            if not task.done():
                task.cancel()
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*bg_tasks, return_exceptions=True), 
                timeout=3.0
            )
        except asyncio.TimeoutError:
            logging.warning("Some tracked background tasks didn't respond to cancellation")
    
    # Then handle any other tasks
    tasks = [t for t in asyncio.all_tasks() 
             if t is not asyncio.current_task() and t not in bg_tasks]
    
    # Cancel remaining tasks
    if tasks:
        logging.info(f"Cancelling {len(tasks)} other active tasks...")
        for task in tasks:
            task.cancel()
        
        # Then wait for them to complete
        logging.info(f"Waiting for tasks to finish cancelling...")
        try:
            # Give tasks time to handle cancellation, but don't wait forever
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
        except asyncio.TimeoutError:
            logging.warning("Timed out waiting for tasks to cancel, some tasks may not have terminated properly")
    
    # Call all shutdown events with a timeout
    if shutdown_events:
        logging.info(f"Running {len(shutdown_events)} shutdown events...")
        try:
            await asyncio.wait_for(
                asyncio.gather(*[event() for event in shutdown_events], return_exceptions=True),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            logging.warning("Timed out waiting for shutdown events, some resources may not be released properly")
    
    # Ensure job queue is cleared
    try:
        await clear_running_jobs()
    except Exception as e:
        logging.error(f"Error clearing job queue during shutdown: {e}")
    
    # Final cleanup
    logging.info("Shutdown complete")


# ----------------------------------------
# FastAPI app setup
# ----------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the broadcast processing task
    for event in startup_events:
        logging.info(f"Running startup event: {event}")
        create_background_task(event())
    
    # Start the broadcast processor if needed
    create_background_task(process_broadcasts())
    
    # Run cleanup on startup with a max age of 30 days
    cleanup_thumbnail_cache(30)
    
    logging.info("Application startup complete")
    try:
        yield
    finally:
        # Start the full shutdown process when the lifespan context exits
        logging.info("Lifespan context exiting, initiating shutdown...")
        await shutdown()
        
        # Ensure any remaining background tasks are explicitly cancelled
        remaining_tasks = [t for t in background_tasks if not t.done()]
        if remaining_tasks:
            logging.info(f"Cancelling {len(remaining_tasks)} remaining background tasks...")
            for task in remaining_tasks:
                task.cancel()
            
            try:
                await asyncio.wait_for(
                    asyncio.gather(*remaining_tasks, return_exceptions=True),
                    timeout=3.0
                )
            except asyncio.TimeoutError:
                logging.warning("Some background tasks could not be cancelled gracefully")
        
        # Make sure we signal server exit clearly
        global server_should_exit
        server_should_exit = True
        logging.info("Application shutdown complete")


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
    global startup_events, shutdown_events
    def try_register(module_name):
        try:
            module = __import__(module_name, fromlist=[''])
            if hasattr(module, "register_api_endpoints"):
                print(f"Registering API endpoints from {module_name}")
                module.register_api_endpoints(app)
            if hasattr(module, "startup_event"):
                startup_events.append(module.startup_event)
            if hasattr(module, "shutdown_event"):
                shutdown_events.append(module.shutdown_event)
        except Exception as e:
            print(f"Error registering API endpoints from {module_name}: {e}")

    for _, module_name, _ in pkgutil.walk_packages(handlers.__path__, handlers.__name__ + "."):
        try_register(module_name)


# Function to handle signals for the main process
def handle_signal(sig, frame):
    logging.info(f"Received signal {signal.Signals(sig).name}")
    # Setting this to True tells the main loop to exit
    global server_should_exit
    server_should_exit = True
    
    # Schedule the shutdown coroutine to run in the event loop
    if asyncio.get_event_loop().is_running():
        asyncio.create_task(shutdown(signal.Signals(sig)))
    else:
        # If no event loop is running, we're likely in the main thread
        # Set the exit flag and let the main process handle cleanup
        logging.info("Signal received outside event loop, flagging for exit")


# Check if a port is in use
def is_port_in_use(port, host='0.0.0.0'):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except socket.error:
            return True


# Custom server with more robust cleanup
class CustomUvicornServer(uvicorn.Server):
    def install_signal_handlers(self):
        # Do nothing here - we handle signals our own way
        pass

    async def serve(self, sockets=None):
        process_id = os.getpid()
        logging.info(f"Started server process [{process_id}]")
        
        await self.startup(sockets=sockets)
        if self.should_exit:
            return
        
        # Main server loop
        await self._main_loop()
        
        await self.shutdown(sockets=sockets)
        logging.info("Server process exiting")
    

# Run the application
if __name__ == "__main__":
    # Register API endpoints
    register_all_endpoints()

    # Preload models if needed
    if args.preload:
        model_paths = preload_all_models(use_auth_token=args.hf_token)

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # Check if port is in use
    if is_port_in_use(args.port, args.host):
        logging.warning(f"Port {args.port} is already in use. Waiting for it to be released...")
        retry_count = 0
        while is_port_in_use(args.port, args.host) and retry_count < 5:
            time.sleep(1)
            retry_count += 1
        
        if is_port_in_use(args.port, args.host):
            logging.error(f"Port {args.port} is still in use after waiting. Please choose a different port or close the application using this port.")
            sys.exit(1)
        
        logging.info(f"Port {args.port} is now available.")
    
    # Create a custom server config
    config = uvicorn.Config(
        app, 
        host=args.host, 
        port=args.port, 
        log_level="info",
        loop="asyncio"
    )
    
    server = uvicorn.Server(config)
    
    # Run the server
    try:
        server.run()
    except KeyboardInterrupt:
        # This should be handled by our signal handler
        pass
    except OSError as e:
        if e.errno == 10048:  # Socket address already in use
            logging.error(f"Port {args.port} is already in use. Please choose a different port or close the application using this port.")
        else:
            logging.error(f"Server error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Server error: {e}")
        sys.exit(1)
    
    # Make sure everything is shut down before exit
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(shutdown())
    except Exception as e:
        logging.error(f"Error during final shutdown: {e}")
    finally:
        loop.close()
    
    logging.info("Server process exited cleanly")
    sys.exit(0)
