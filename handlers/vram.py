import torch


def get_cuda_free_memory_gb(device=None):
    if device is None:
        device = gpu

    memory_stats = torch.cuda.memory_stats(device)
    bytes_active = memory_stats['active_bytes.all.current']
    bytes_reserved = memory_stats['reserved_bytes.all.current']
    bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
    bytes_inactive_reserved = bytes_reserved - bytes_active
    bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
    return bytes_total_available / (1024 ** 3)


def fake_diffusers_current_device(model: torch.nn.Module, target_device: torch.device):
    if hasattr(model, 'scale_shift_table'):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return

    for k, p in model.named_modules():
        if hasattr(p, 'weight'):
            p.to(target_device)
            return


def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')

    for m in model.modules():
        if get_cuda_free_memory_gb(target_device) <= preserved_memory_gb:
            torch.cuda.empty_cache()
            return

        if hasattr(m, 'weight'):
            m.to(device=target_device)

    model.to(device=target_device)
    torch.cuda.empty_cache()
    return


def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=0):
    print(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')

    for m in model.modules():
        if get_cuda_free_memory_gb(target_device) >= preserved_memory_gb:
            torch.cuda.empty_cache()
            return

        if hasattr(m, 'weight'):
            m.to(device=cpu)

    model.to(device=cpu)
    torch.cuda.empty_cache()
    return


def unload_complete_models(*args):
    for m in gpu_complete_modules + list(args):
        m.to(device=cpu)
        #print(f'Unloaded {m.__class__.__name__} as complete.')

    gpu_complete_modules.clear()
    torch.cuda.empty_cache()
    return


def load_model_as_complete(model, target_device, unload=True):
    if unload:
        unload_complete_models()

    model.to(device=target_device)
    #print(f'Loaded {model.__class__.__name__} to {target_device} as complete.')

    gpu_complete_modules.append(model)
    return


gpu = torch.device(f'cuda:{torch.cuda.current_device()}')
free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 32
adaptive_memory_management = True

print(f'Free VRAM {free_mem_gb:.2f} GB')
print(f'High-VRAM Mode: {high_vram}')
print(f'Adaptive Memory Management: {adaptive_memory_management}')

cpu = torch.device('cpu')
gpu_complete_modules = []
