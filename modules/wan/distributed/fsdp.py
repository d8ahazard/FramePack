"""Utility functions for Fully Sharded Data Parallel (FSDP) training."""
import logging
import torch

def shard_model(model, device_id=0):
    """
    Shard model with FSDP.
    
    Args:
        model: model to shard
        device_id: GPU device ID
        
    Returns:
        sharded model
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        import transformers
        
        auto_wrap_policy = transformer_auto_wrap_policy(
            transformer_layer_cls={
                transformers.models.t5.modeling_t5.T5Block,
                transformers.models.t5.modeling_t5.T5LayerSelfAttention
            })
        
        # Use mixed precision to speed up training
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
        
        return FSDP(
            model, 
            device_id=torch.device(f"cuda:{device_id}"),
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
        )
    except Exception as e:
        logging.warning(f"Failed to shard model with FSDP: {e}")
        model.cuda(device_id)
        return model 