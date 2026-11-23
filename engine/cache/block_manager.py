from collections import deque


class BlockManager:
    """Hands out fixed-size KV blocks. Each sequence owns a block_table: its physical block ids in order"""

    def __init__(self, num_blocks, block_size=16):
        self.bs, self.num_blocks = block_size, num_blocks
        self.free = deque(range(num_blocks))


    def blocks_for(self, n_tokens):
        return -(-n_tokens // self.bs)


    def ensure(self, req, n_tokens):
        """Grow req's table to hold n_tokens. All or nothing: returns False and changes nothing if short"""
        need = self.blocks_for(n_tokens) - len(req.block_table)
        if need > len(self.free):
            return False
        req.block_table += [self.free.popleft() for _ in range(need)]   # need <= 0 adds nothing
        return True


    def release(self, req):
        self.free.extend(req.block_table)
        req.block_table = []        


def slot_ids(table, start, n, bs=16):
    """Flat cache slot for each of the n tokens starting at position start"""
    return [table[p // bs] * bs + p % bs for p in range(start, start + n)]
    

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

