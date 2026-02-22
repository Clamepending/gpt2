import torch
import numpy as np
import os
import glob

class FineWebDataLoader:
    def __init__(self, B, T, ddp_rank=0, ddp_world_size=1, 
                 dataset_path="fineweb_sample_10B",
                 buffer_size=1000000,
                 split="training"):
        self.B, self.T = B, T
        self.ddp_rank, self.ddp_world_size = ddp_rank, ddp_world_size
        self.buffer_size = buffer_size
        
        # Resolve the dataset path (handle symlink)
        if os.path.islink(dataset_path):
            dataset_path = os.path.realpath(dataset_path)
        elif not os.path.isabs(dataset_path):
            # If relative path, try to resolve from current directory
            abs_path = os.path.abspath(dataset_path)
            if os.path.exists(abs_path):
                dataset_path = abs_path
        
        self.dataset_path = dataset_path
        
        # Find all numpy shard files
        shard_pattern = os.path.join(dataset_path, f"{split}_shard_*.npy")
        all_shards = sorted(glob.glob(shard_pattern))
        
        if len(all_shards) == 0:
            raise ValueError(f"No numpy shards found in {dataset_path}")
        
        # Distribute shards across DDP ranks
        # Each rank gets a subset of shards
        
        shards_per_rank = len(all_shards) // ddp_world_size
        start_idx = ddp_rank * shards_per_rank
        if ddp_rank == ddp_world_size - 1:
            # Last rank gets any remaining shards
            self.shard_files = all_shards[start_idx:]
        else:
            self.shard_files = all_shards[start_idx:start_idx + shards_per_rank]
        
        # temporary fix because have just 1 validaiton shard and only the master needs it
        if split == "validation":
            if ddp_world_size == 1 or ddp_rank == 0:
                self.shard_files = all_shards
            else:
                self.shard_files = []
        
        if len(self.shard_files) == 0:
            raise ValueError(f"No shards assigned to rank {ddp_rank} for split {split}")
        
        self.current_shard_idx = 0
        self.current_shard_local_idx = -1
        self.current_shard_data = None
        self.current_shard_pos = 0
        self.shard_order = np.random.permutation(len(self.shard_files)) # shuffle shard order to fix U shaped loss?
        
        # The main tensor buffer
        self.token_buffer = torch.zeros(buffer_size + B * T + 1, dtype=torch.long)
        self.buffer_fill = 0 
        self.current_pos = 0 
        
        # Store tokens from a shard that haven't been put in the buffer yet
        self.remainder = torch.tensor([], dtype=torch.long)

        self._fill_buffer()
    
    def _load_next_shard(self):
        """Load the next shard in rotation."""
        # Only load if we don't have a shard or current shard is exhausted
        if self.current_shard_data is None or self.current_shard_pos >= len(self.current_shard_data):
            # Load next shard
            self.current_shard_local_idx = int(self.shard_order[self.current_shard_idx])
            shard_file = self.shard_files[self.current_shard_local_idx]
            self.current_shard_data = np.load(shard_file)
            # Convert to torch tensor and ensure it's long dtype
            self.current_shard_data = torch.from_numpy(self.current_shard_data).long()
            self.current_shard_pos = 0
            
            # Move to next shard (with wrap-around)
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shard_order)
        
    def _fill_buffer(self):
        # 1. Shift unused tokens to the start
        leftover_in_buffer = self.buffer_fill - self.current_pos
        if leftover_in_buffer > 0:
            self.token_buffer[:leftover_in_buffer] = self.token_buffer[self.current_pos:self.buffer_fill]
        
        self.buffer_fill = leftover_in_buffer
        self.current_pos = 0

        while self.buffer_fill < self.buffer_size:
            # 2. Check if we have leftover tokens from the PREVIOUS shard
            if len(self.remainder) > 0:
                t_tokens = self.remainder
                self.remainder = torch.tensor([], dtype=torch.long) # Clear it
            else:
                # 3. Otherwise, load tokens from the current shard
                self._load_next_shard()
                
                # Get tokens from current position in shard
                t_tokens = self.current_shard_data[self.current_shard_pos:]


            # 4. Determine how much fits
            space_left = self.token_buffer.size(0) - self.buffer_fill
            if len(t_tokens) <= space_left:
                # It all fits!
                self.token_buffer[self.buffer_fill : self.buffer_fill + len(t_tokens)] = t_tokens
                self.buffer_fill += len(t_tokens)
                # Update shard position if we consumed from shard
                self.current_shard_pos += len(t_tokens)
            else:
                # It doesn't all fit. Take what we can and save the rest in remainder.
                self.token_buffer[self.buffer_fill : self.buffer_fill + space_left] = t_tokens[:space_left]
                self.remainder = t_tokens[space_left:] # Save for next loop/call
                self.buffer_fill += space_left
                # Update shard position if we consumed from shard
                self.current_shard_pos += space_left

    def get_batch(self):
        req = self.B * self.T
        if self.current_pos + req + 1 > self.buffer_fill:
            self._fill_buffer()
            
        buf = self.token_buffer[self.current_pos : self.current_pos + req + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        
        self.current_pos += req
        return x, y
    
    # fraction of the current shard that has been processed
    def get_current_shard_progress(self):
        if self.current_shard_data is None or len(self.current_shard_data) == 0:
            return 0.0
        return float(self.current_shard_pos) / len(self.current_shard_data)


import time
def main():
    dataloader = FineWebDataLoader(B=4, T=1024, ddp_rank=0, ddp_world_size=1)
    # measure tokens per second
    start_time = time.time()
    for _ in range(500):
        x, y = dataloader.get_batch()
    end_time = time.time()
    print(f"Tokens per second: {500 * dataloader.B * dataloader.T / (end_time - start_time)}")


if __name__ == "__main__":
    main()