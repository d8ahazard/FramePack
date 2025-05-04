import os
import time

from huggingface_hub import snapshot_download
from handlers.path import model_path, lora_path


def check_download_model(repo_id, module=None, subfolder=None, retries=3, use_auth_token=None):
    """
    Downloads a model from Hugging Face Hub to a local models directory.

    Args:
        repo_id: The Hugging Face model repository ID
        module: Optional module name for the model
        subfolder: Optional subfolder within the model repository
        retries: Number of download retries on failure
        use_auth_token: Optional auth token for private models

    Returns:
        str: Path to the downloaded model for use with from_pretrained
    """
    # Define the models directory

    # Create a sanitized directory name for this specific model
    model_name = repo_id.replace('/', '_')
    models_dir = model_path
    if module:
        models_dir = os.path.join(models_dir, module)
    if subfolder:
        models_dir = os.path.join(models_dir, subfolder)
    model_dir = os.path.join(models_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Check if the model is already downloaded
    if os.path.exists(model_dir) and os.listdir(model_dir):
        print(f"Using cached model {repo_id} from {model_dir}")
    else:
        print(f"Downloading {repo_id} to {model_dir}...")

        for attempt in range(retries):
            try:
                # First try with local_files_only=True
                try:
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=model_dir,
                        local_files_only=False,
                        token=use_auth_token,
                        max_workers=4  # Limit concurrent downloads
                    )
                    print(f"Model {repo_id} found in local cache.")
                    break
                except Exception as local_err:
                    print(f"Model not found locally, downloading from HuggingFace Hub: {str(local_err)}")
                    # If local download fails, try downloading from the internet
                    snapshot_download(
                        repo_id=repo_id,
                        local_dir=model_dir,
                        local_files_only=False,  # Allow internet downloads
                        token=use_auth_token,
                        max_workers=4  # Limit concurrent downloads
                    )
                    print(f"Successfully downloaded model {repo_id} from HuggingFace Hub.")
                    break
            except Exception as e:
                if attempt < retries - 1:
                    print(f"Download attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1)  # Wait before retry
                else:
                    print(f"All {retries} download attempts failed for {repo_id}")
                    raise RuntimeError(f"Failed to download model {repo_id}: {e}")

    # Verify the model directory has content
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        raise RuntimeError(f"Model directory {model_dir} is empty after download")

    # Return the appropriate path
    if subfolder:
        subfolder_path = os.path.join(model_dir, subfolder)
        if not os.path.exists(subfolder_path):
            raise ValueError(f"Subfolder '{subfolder}' doesn't exist in {model_dir}")
        return subfolder_path

    return model_dir


def list_loras(include_extension=False):
    """
    List all available LoRA models.
    
    Args:
        include_extension: Whether to include file extensions in returned names
        
    Returns:
        list: List of available LoRA model names
    """
    os.makedirs(lora_path, exist_ok=True)
    
    lora_files = []
    valid_extensions = ['.pt', '.pth', '.bin', '.safetensors']
    
    for file in os.listdir(lora_path):
        file_path = os.path.join(lora_path, file)
        if os.path.isfile(file_path) and any(file.endswith(ext) for ext in valid_extensions):
            if include_extension:
                lora_files.append(file)
            else:
                # Remove extension and return base filename
                lora_files.append(os.path.splitext(file)[0])
    
    return lora_files


def get_lora_full_path(lora_name):
    """
    Get full path to a LoRA model file.
    
    Args:
        lora_name: Name of the LoRA model (with or without extension)
        
    Returns:
        str: Full path to the LoRA model file, or None if not found
    """
    # Check if name already has an extension
    has_ext = any(lora_name.endswith(ext) for ext in ['.pt', '.pth', '.bin', '.safetensors'])
    
    if has_ext:
        # Direct check with extension
        full_path = os.path.join(lora_path, lora_name)
        if os.path.isfile(full_path):
            return full_path
        return None
    
    # Try different extensions
    for ext in ['.safetensors', '.pt', '.bin', '.pth']:
        full_path = os.path.join(lora_path, f"{lora_name}{ext}")
        if os.path.isfile(full_path):
            return full_path
    
    return None


def preload_all_models(use_auth_token=None):
    """
    Preloads all required models to the local cache.

    Args:
        use_auth_token: Optional auth token for private models

    Returns:
        dict: Paths to all downloaded models
    """
    print("Preloading all required models...")

    model_repos = {
        "hunyuan": "hunyuanvideo-community/HunyuanVideo",
        "flux": "lllyasviel/flux_redux_bfl",
        "framepack": "lllyasviel/FramePackI2V_HY"
    }

    model_paths = {}

    for name, repo_id in model_repos.items():
        try:
            path = check_download_model(repo_id, use_auth_token=use_auth_token)
            model_paths[name] = path
            print(f"Successfully preloaded {name} model from {repo_id}")
        except Exception as e:
            print(f"Error preloading {name} model: {e}")
            raise

    print("All models preloaded successfully!")
    return model_paths


def list_models(subfolder=None, recurse=False, list_dirs=False, include_names=None, exclude_names=None):
    """
    List all models in the local models directory.

    Args:
        subfolder: Optional subfolder to list models from
        recurse: Whether to list models recursively
        list_dirs: Whether to list directories instead of files
        include_names: Optional list of names/extensions to include
        exclude_names: Optional list of names/extensions to exclude
    """
    models_dir = model_path
    if subfolder:
        models_dir = os.path.join(models_dir, subfolder)

    if not os.path.exists(models_dir):
        print(f"Models directory {models_dir} does not exist")
        return []

    model_list = []
    for root, dirs, files in os.walk(models_dir):
        if not recurse and root != models_dir:
            continue
        if list_dirs:
            model_list.extend(dirs)
        else:
            model_list.extend([f for f in files])

    if include_names:
        model_list = [f for f in model_list if any(name in f for name in include_names)]
        # Ensure we're not looking for extensions
        model_list = [f for f in model_list if not f.endswith(tuple(include_names))]

    if exclude_names:
        model_list = [f for f in model_list if not any(name in f for name in exclude_names)]
        # Ensure we're not looking for extensions
        model_list = [f for f in model_list if not f.endswith(tuple(exclude_names))]

    return model_list


def register_api_endpoints(app):
    """
    Register API endpoints for model management.
    """
    api_tag = __name__.split(".")[-1].title().replace("_", " ")

    @app.get("/api/list_models", tags=[api_tag])
    async def list_models_endpoint(subfolder: str = None, recurse: bool = False, list_dirs: bool = False,
                                   include_names: str = None, exclude_names: str = None):
        """
        List all models in the local models directory.

        Args:
            subfolder: Optional subfolder to list models from
            recurse: Whether to list models recursively
            list_dirs: Whether to list directories instead of files
            include_names: Optional list of names/extensions to include
            exclude_names: Optional list of names/extensions to exclude
        """
        include_names = include_names.split(",") if include_names else None
        exclude_names = exclude_names.split(",") if exclude_names else None

        models = list_models(subfolder=subfolder, recurse=recurse, list_dirs=list_dirs,
                             include_names=include_names, exclude_names=exclude_names)
        return {"models": models}
        
    @app.get("/api/list_loras", tags=[api_tag])
    async def list_loras_endpoint(include_extension: bool = False):
        """
        List all available LoRA models.
        
        Args:
            include_extension: Whether to include file extensions in returned names
        """
        loras = list_loras(include_extension=include_extension)
        return {"loras": loras}
        
    @app.get("/api/get_lora_path/{lora_name}", tags=[api_tag])
    async def get_lora_path_endpoint(lora_name: str):
        """
        Get the full path for a specific LoRA model.
        
        Args:
            lora_name: Name of the LoRA model (with or without extension)
        """
        full_path = get_lora_full_path(lora_name)
        if full_path:
            return {"success": True, "lora_path": full_path}
        return {"success": False, "error": f"LoRA model '{lora_name}' not found"}