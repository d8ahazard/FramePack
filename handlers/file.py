import hashlib
import hashlib
import logging
import os
import time
from typing import List, Union

from fastapi import HTTPException
from fastapi import UploadFile, File
from starlette.responses import JSONResponse, FileResponse

from datatypes.datatypes import ErrorResponse, DeleteVideoRequest, UploadResponse, FileExistsResponse, \
    PathsExistResponse
from handlers.path import output_path, thumbnail_path, upload_path

logger = logging.getLogger(__name__)


def generate_thumbnail(video_path):
    """
    Generate a thumbnail for the given video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Path to the generated thumbnail image.
    """
    # Placeholder for actual thumbnail generation logic
    # For example, using ffmpeg or any other library
    thumbnail_file = os.path.join(thumbnail_path, f"{os.path.basename(video_path)}.jpg")
    if not os.path.exists(thumbnail_file):
        try:
            # Run ffmpeg
            os.system(f"ffmpeg -i {video_path} -ss 00:00:01.000 -vframes 1 {thumbnail_file}")
        except Exception as e:
            print(f"Error generating thumbnail: {e}")
            return None
    # Check if thumbnail generation was successful
    if not os.path.exists(thumbnail_file):
        print(f"Thumbnail generation failed for {video_path}")
        return None
    return thumbnail_file


def register_api_endpoints(app):
    """Register file handling API endpoints"""
    api_tag = __name__.split(".")[-1].title().replace("_", " ")

    @app.get("/api/check_file_exists", response_model=FileExistsResponse, tags=[api_tag])
    async def check_file_exists(path: str):
        """
        Check if a file exists on the server
        
        Args:
            path: The file path to check
            
        Returns:
            A dictionary with exists: True/False and the normalized path
        """
        # Clean the path to prevent directory traversal attacks
        clean_path = os.path.normpath(path)

        # If path starts with file:// protocol, remove it
        if clean_path.startswith('file://'):
            clean_path = clean_path[7:]

        # Check if the file exists
        exists = os.path.exists(clean_path) and os.path.isfile(clean_path)

        return FileExistsResponse(exists=exists, path=clean_path)

    @app.post("/api/check_multiple_files", response_model=PathsExistResponse, tags=[api_tag])
    async def check_multiple_files(paths: List[str]):
        """
        Check if multiple files exist on the server
        
        Args:
            paths: List of file paths to check
            
        Returns:
            List of results with exists: True/False for each path
        """
        results = []

        for path in paths:
            # Clean the path to prevent directory traversal attacks
            clean_path = os.path.normpath(path)

            # If path starts with file:// protocol, remove it
            if clean_path.startswith('file://'):
                clean_path = clean_path[7:]

            # Check if the file exists
            exists = os.path.exists(clean_path) and os.path.isfile(clean_path)

            results.append(FileExistsResponse(exists=exists, path=clean_path))

        return PathsExistResponse(results=results)

    @app.get("/api/serve_file", tags=[api_tag])
    async def serve_file(path: str, download: bool = False):
        """
        Serve a file from the server with proper headers
        
        Args:
            path: The file path to serve
            download: Whether to serve as an attachment (download)
            
        Returns:
            File response
        """
        # Clean the path to prevent directory traversal attacks
        clean_path = os.path.normpath(path)

        # If path starts with file:// protocol, remove it
        if clean_path.startswith('file://'):
            clean_path = clean_path[7:]

        # Check if the file exists
        if not os.path.exists(clean_path) or not os.path.isfile(clean_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Get filename from path
        filename = os.path.basename(clean_path)

        # Determine media type based on extension
        _, ext = os.path.splitext(filename)
        ext = ext.lower()

        media_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp',
            '.mp4': 'video/mp4',
            '.webm': 'video/webm',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.json': 'application/json',
            '.txt': 'text/plain',
        }

        media_type = media_types.get(ext, 'application/octet-stream')

        return FileResponse(
            path=clean_path,
            filename=filename,
            media_type=media_type,
            content_disposition_type="attachment" if download else "inline"
        )

    @app.get("/api/list_outputs", tags=[api_tag])
    async def list_outputs():
        """
        List all generated output videos
        
        Returns:
            A list of all output videos with metadata
        """
        outputs = []
        try:
            for filename in os.listdir(output_path):
                if filename.endswith('.mp4'):
                    file_path = os.path.join(output_path, filename)
                    # Skip temp directories and check if it's a file
                    if not os.path.isfile(file_path) or '_temp' in filename:
                        continue

                    # Get file stats
                    stats = os.stat(file_path)
                    timestamp = stats.st_mtime

                    # Generate thumbnail
                    thumbnail_path = generate_thumbnail(file_path)
                    thumbnail_url = f"/thumbnails/{os.path.basename(thumbnail_path)}" if thumbnail_path else None

                    # Create output entry
                    output = {
                        "name": filename,
                        "path": f"/outputs/{filename}",
                        "timestamp": timestamp,
                        "size": stats.st_size,
                        "thumbnail": thumbnail_url
                    }
                    outputs.append(output)

            # Sort by timestamp (newest first)
            outputs.sort(key=lambda x: x["timestamp"], reverse=True)

        except Exception as e:
            print(f"Error listing outputs: {e}")
            import traceback
            traceback.print_exc()

        return outputs

    @app.post("/api/delete_video", response_model=Union[dict, ErrorResponse], tags=[api_tag])
    async def delete_video(request: DeleteVideoRequest):
        """
        Delete an output video file

        Args:
            request: DeleteVideoRequest containing the video path

        Returns:
            Success message or error
        """
        try:
            # Normalize video path
            video_path = request.video_path

            # If path starts with /outputs/, remove that prefix and join with output_path
            if video_path.startswith('/outputs/'):
                video_path = os.path.join(output_path, video_path[9:])  # Remove '/outputs/' prefix
            # If path doesn't start with the outputs folder, join it
            elif not video_path.startswith(output_path):
                video_path = os.path.join(output_path, video_path)

            # Get the basename for logging and response
            video_basename = os.path.basename(video_path)

            # Print info for debugging
            print(f"Attempting to delete video: {video_path}")

            # Check if file exists
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                return ErrorResponse(error=f"Video file not found: {video_basename}")

            # Delete associated thumbnail if it exists
            try:
                video_hash = hash(video_basename + str(os.path.getmtime(video_path)))
                thumbnail_name = f"{video_hash}.jpg"
                thumbnail_file = os.path.join(thumbnail_path, thumbnail_name)

                if os.path.exists(thumbnail_file):
                    os.remove(thumbnail_file)
                    print(f"Deleted thumbnail: {thumbnail_file}")
            except Exception as e:
                print(f"Error deleting thumbnail: {e}")
                # Continue with video deletion even if thumbnail deletion fails

            # Delete the video file
            os.remove(video_path)
            print(f"Successfully deleted video: {video_path}")

            return {
                "success": True,
                "message": f"Video {video_basename} deleted successfully"
            }

        except Exception as e:
            print(f"Error deleting video: {e}")
            return ErrorResponse(error=f"Failed to delete video: {str(e)}")

    @app.post("/api/upload_image", response_model=UploadResponse, tags=[api_tag])
    async def upload_image(file: UploadFile = File(...)):
        """
        Upload an image file to the server

        Returns:
            success: Whether the upload was successful
            filename: The filename on the server
            path: The full server path to the file (for internal use)
            url: The API URL to access the file (for frontend use)
        """
        try:
            # Get file content hash to ensure uniqueness
            file_content = await file.read()

            # Create a hash of the file content
            file_hash = hashlib.md5(file_content).hexdigest()

            # Return file pointer to start for later use
            await file.seek(0)

            # Get original filename and extension
            original_filename = file.filename
            base_name, ext = os.path.splitext(original_filename)

            # Create a new filename with the hash
            filename = f"{file_hash}_{original_filename}"
            file_path = os.path.join(upload_path, filename)

            # Check if a file with this hash already exists
            if os.path.exists(file_path):
                print(f"File with hash {file_hash} already exists, reusing existing file")
            else:
                # Save the uploaded file
                with open(file_path, "wb") as f:
                    f.write(file_content)
                print(f"File uploaded successfully: {file_path}")

            # Create API URL for frontend use
            api_url = f"/api/serve_file?path={file_path}"
            
            # For backwards compatibility, also create a /uploads/ URL
            upload_url = f"/uploads/{filename}"

            # Return both the full server path (for job processing) and the API URL (for frontend)
            return UploadResponse(
                success=True,
                filename=filename,
                path=file_path,  # Return the full server path for internal use
                url=api_url,     # Return the API URL for frontend use
                upload_url=upload_url  # Return the /uploads/ URL for backward compatibility
            )
        except Exception as e:
            print(f"Error uploading file: {str(e)}")
            import traceback
            traceback.print_exc()
            return UploadResponse(
                success=False,
                error=f"Upload failed: {str(e)}"
            )

    @app.get("/api/video_thumbnail", tags=[api_tag])
    async def get_video_thumbnail(video: str):
        """
        Generate and return a thumbnail for a video file

        Args:
            video: Path to the video file

        Returns:
            The thumbnail image
        """
        try:
            # Ensure the path is inside the outputs folder for security
            if video.startswith('/outputs/'):
                video_path = os.path.join(output_path, os.path.basename(video))
            elif not video.startswith(output_path):
                video_path = os.path.join(output_path, video)
            else:
                video_path = video

            # Check if file exists
            if not os.path.exists(video_path):
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Video file not found: {os.path.basename(video_path)}"}
                )

            # Generate thumbnail
            thumbnail_file = generate_thumbnail(video_path)

            if thumbnail_file and os.path.exists(thumbnail_file):
                return FileResponse(thumbnail_file)
            else:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Failed to generate thumbnail"}
                )

        except Exception as e:
            print(f"Error generating thumbnail: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to generate thumbnail: {str(e)}"}
            )


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
