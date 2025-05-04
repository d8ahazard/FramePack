# Websocket handler
import asyncio
import json
import queue
from typing import Dict, Any
from typing import List
import logging

from fastapi import WebSocket, WebSocketDisconnect

from datatypes.datatypes import ConnectionManager
logger = logging.getLogger(__name__)

# Create connection manager instances
global_manager = ConnectionManager()  # For broadcasts to all clients
job_manager = ConnectionManager()  # For job-specific broadcasts
terminate_process = False

# Store active websocket connections
active_connections: List[WebSocket] = []
# Message queue to broadcast updates to clients
message_queue = asyncio.Queue()

# Create a connection manager instance
manager = ConnectionManager()

# Create a queue for broadcasting job updates
broadcast_queue = queue.Queue()
job_statuses = {}

async def startup_event():
    await process_broadcasts()

async def shutdown_event():
    global broadcaster_task
    global terminate_process
    terminate_process = True
    if broadcaster_task:
        broadcaster_task.cancel()
        try:
            await broadcaster_task
        except asyncio.CancelledError:
            pass
    logger.info("Shutdown event complete for socket")

async def process_broadcasts():
    while not terminate_process:
        try:
            # Check if there are any pending broadcasts
            if not broadcast_queue.empty():
                job_id, data = broadcast_queue.get()
                # Send to both old and new systems during transition
                await manager.broadcast(job_id, data)
                broadcast_queue.task_done()
            # Short sleep to avoid high CPU usage
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error processing broadcast: {e}")
            await asyncio.sleep(1)  # Wait longer on error


async def websocket_broadcaster():
    """
    Background task to broadcast messages to all connected websocket clients.
    """
    while not terminate_process:
        # Get message from queue
        message = await message_queue.get()

        # Broadcast to all connected clients
        for connection in active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending message to client: {e}")
                # Connection might be closed, but we'll handle removal in the main handler

        # Mark task as done
        message_queue.task_done()


# Global broadcaster task
broadcaster_task = None


def update_status_sync(job_id: str, status: str = None, progress: int = None, message: str = None,
                       data: Dict[str, Any] = None):
    """
    Update job status and broadcast to connected clients - works in synchronous context
    
    Args:
        job_id: The unique job identifier
        status: Current job status (e.g., "queued", "running", "completed", "failed")
        progress: Progress percentage (0-100)
        message: Status message to display to user
        data: Additional data to include in the update
    """
    update_data = {
        "job_id": job_id,
        "type": "status_update"
    }

    # Add optional fields if provided
    if status is not None:
        update_data["status"] = status

    if progress is not None:
        update_data["progress"] = progress

    if message is not None:
        update_data["message"] = message

    # Add any additional data
    if data:
        update_data.update(data)

    # Queue the update for broadcast using the synchronous method
    # Add to queue in thread-safe manner
    try:
        queue_broadcast(job_id, update_data)
    except Exception as e:
        print(f"Error queuing status update: {e}")


async def update_status(job_id: str, status: str = None, progress: int = None, message: str = None,
                        data: Dict[str, Any] = None):
    """
    Update job status and broadcast to connected clients
    
    Args:
        job_id: The unique job identifier
        status: Current job status (e.g., "queued", "running", "completed", "failed")
        progress: Progress percentage (0-100)
        message: Status message to display to user
        data: Additional data to include in the update
    """
    # Create update data with common fields
    update_data = {
        "job_id": job_id,
        "type": "status_update"
    }

    # Add optional fields if provided
    if status is not None:
        update_data["status"] = status

    if progress is not None:
        update_data["progress"] = progress

    if message is not None:
        update_data["message"] = message

    # Add any additional data
    if data:
        update_data.update(data)

    # Broadcast to job-specific clients
    await job_manager.broadcast(job_id, update_data)

    # Also broadcast to global clients with job_id in the message
    # This allows clients listening to the general websocket to get all updates
    await global_manager.broadcast("global", update_data)


def queue_broadcast(job_id: str, data: Dict[str, Any]):
    """
    Queue a broadcast for job updates (compatible with infer.py)
    
    Args:
        job_id: The unique job identifier
        data: The data to broadcast
    """
    try:
        # Try to use a running event loop if one exists
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(update_status(job_id, data=data))
                return
        except RuntimeError:
            # No event loop, continue with synchronous approach
            pass
            
        # Use a new event loop for this operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(update_status(job_id, data=data))
        loop.close()
    except Exception as e:
        print(f"Error queuing broadcast: {e}")


def register_api_endpoints(app):
    """
    Register API endpoints for WebSocket communication.
    """
    api_tag = __name__.split(".")[-1].title().replace("_", " ")

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """
        Global WebSocket endpoint for real-time updates across all jobs
        """
        await global_manager.connect(websocket, "global")

        try:
            # Send initial connection confirmation
            await websocket.send_json({
                "type": "connection_established",
                "message": "WebSocket connection established"
            })

            # Keep connection open and handle incoming messages
            while True:
                # Receive and parse message
                message = await websocket.receive_text()

                try:
                    data = json.loads(message)

                    # Handle different message types
                    if "type" in data:
                        if data["type"] == "ping":
                            # Simple ping/pong for connection health check
                            await websocket.send_json({"type": "pong"})
                        else:
                            # Unknown message type
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Unknown message type: {data['type']}"
                            })
                    else:
                        # Message missing type
                        await websocket.send_json({
                            "type": "error",
                            "message": "Message missing 'type' field"
                        })

                except json.JSONDecodeError:
                    # Invalid JSON
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON message"
                    })

        except WebSocketDisconnect:
            # Client disconnected
            pass
        except Exception as e:
            # Other error occurred
            print(f"WebSocket error: {e}")
        finally:
            # Remove connection from active list
            global_manager.disconnect(websocket, "global")

    @app.websocket("/ws/job/{job_id}")
    async def websocket_job_endpoint(websocket: WebSocket, job_id: str):
        """
        Job-specific WebSocket endpoint for targeted updates
        """
        global job_statuses
        # If job_statuses is not defined, initialize it
        if job_statuses is None:
            job_statuses = {}

        # If job_id is 'undefined', get the first running job if any
        if job_id == "undefined":
            statuses = job_statuses.values()
            running_jobs = [status for status in statuses if status.status == "running"]
            if running_jobs:
                job_id = running_jobs[0].job_id
                logger.info(f"No job_id provided, using first running job: {job_id}")

        await job_manager.connect(websocket, job_id)

        try:
            # Send initial connection confirmation
            await websocket.send_json({
                "type": "connection_established",
                "job_id": job_id,
                "message": f"WebSocket connection established for job {job_id}"
            })

            # Send initial job status on connection if available
            from handlers.job_queue import job_statuses
            status_obj = job_statuses.get(job_id)
            if status_obj:
                await websocket.send_text(json.dumps(status_obj.to_dict()))

            # Keep connection open and handle incoming messages
            while True:
                # Receive message - most likely ping/pong for health check
                data = await websocket.receive_text()
                # Echo back to confirm connection is alive
                await websocket.send_text(data)

        except WebSocketDisconnect:
            # Client disconnected
            pass
        except Exception as e:
            # Other error occurred
            print(f"WebSocket error: {e}")
        finally:
            # Remove connection
            job_manager.disconnect(websocket, job_id)

    # Start the broadcaster task when the app starts
    @app.on_event("startup")
    async def startup_broadcaster():
        global broadcaster_task
        broadcaster_task = asyncio.create_task(websocket_broadcaster())

    @app.on_event("shutdown")
    async def shutdown_broadcaster():
        # Cancel the broadcaster task
        if broadcaster_task:
            broadcaster_task.cancel()
            try:
                await broadcaster_task
            except asyncio.CancelledError:
                pass

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """
        WebSocket endpoint for real-time updates
        """
        await websocket.accept()
        active_connections.append(websocket)

        try:
            # Send initial connection confirmation
            await websocket.send_json({
                "type": "connection_established",
                "message": "WebSocket connection established"
            })

            # Keep connection open and handle incoming messages
            while True:
                # Receive and parse message
                message = await websocket.receive_text()

                try:
                    data = json.loads(message)

                    # Handle different message types
                    if "type" in data:
                        if data["type"] == "ping":
                            # Simple ping/pong for connection health check
                            await websocket.send_json({"type": "pong"})
                        else:
                            # Unknown message type
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Unknown message type: {data['type']}"
                            })
                    else:
                        # Message missing type
                        await websocket.send_json({
                            "type": "error",
                            "message": "Message missing 'type' field"
                        })

                except json.JSONDecodeError:
                    # Invalid JSON
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON message"
                    })

        except WebSocketDisconnect:
            # Client disconnected
            pass
        except Exception as e:
            # Other error occurred
            print(f"WebSocket error: {e}")
        finally:
            # Remove connection from active list
            if websocket in active_connections:
                active_connections.remove(websocket)
