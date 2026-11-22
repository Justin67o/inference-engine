def bytes_per_block(cfg, head_dim, dtype_bytes, block_size=16):
    """K and V, for every layer and KV head, for block_size tokens"""
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * dtype_bytes * block_size


def num_blocks_for(free_bytes, activation_bytes, block_bytes, headroom=0.1):
    """How many blocks fit in free GPU memory after reserving room for activations, 
    keeping a fraction of free memory as headroom for fragmentation and other allocations"""
    usable = free_bytes * (1 - headroom) - activation_bytes
    n = int(usable // block_bytes)
    if n < 1:
        raise ValueError(f"no room for KV blocks: {free_bytes} free, {activation_bytes} for activation")
    return n
