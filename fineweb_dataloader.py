import torch
import tiktoken
from datatrove.pipeline.readers import ParquetReader

class FineWebDataLoader:
    def __init__(self, B, T, ddp_rank=0, ddp_world_size=1, 
                 dataset_path="hf://datasets/HuggingFaceFW/fineweb/data",
                 buffer_size=1000000, limit=None):
        self.B, self.T = B, T
        self.ddp_rank, self.ddp_world_size = ddp_rank, ddp_world_size
        self.buffer_size = buffer_size
        self.dataset_path = dataset_path
        
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.eot_token = self.tokenizer.eot_token
        
        # ParquetReader expects limit to be an int or -1, not None
        self.limit = -1
        self.data_reader = ParquetReader(dataset_path, limit=self.limit)
        self.data_iterator = iter(self.data_reader(rank=ddp_rank, world_size=ddp_world_size))
        
        # The main tensor buffer
        self.token_buffer = torch.zeros(buffer_size + B * T + 1, dtype=torch.long)
        self.buffer_fill = 0 
        self.current_pos = 0 
        
        # NEW: Store tokens from a document that haven't been put in the buffer yet
        self.remainder = torch.tensor([], dtype=torch.long)

        self._fill_buffer()
        
    def _fill_buffer(self):
        # 1. Shift unused tokens to the start
        leftover_in_buffer = self.buffer_fill - self.current_pos
        if leftover_in_buffer > 0:
            self.token_buffer[:leftover_in_buffer] = self.token_buffer[self.current_pos:self.buffer_fill]
        
        self.buffer_fill = leftover_in_buffer
        self.current_pos = 0

        while self.buffer_fill < self.buffer_size:
            # 2. Check if we have leftover tokens from the PREVIOUS document
            if len(self.remainder) > 0:
                t_tokens = self.remainder
                self.remainder = torch.tensor([], dtype=torch.long) # Clear it
            else:
                # 3. Otherwise, fetch a NEW document
                try:
                    doc = next(self.data_iterator)
                    text = doc.text if hasattr(doc, 'text') else str(doc)
                    tokens = self.tokenizer.encode(text)
                    tokens.append(self.eot_token)
                    t_tokens = torch.tensor(tokens, dtype=torch.long)
                except StopIteration:
                    # Recreate the reader and iterator when dataset is exhausted
                    self.data_reader = ParquetReader(self.dataset_path, limit=self.limit)
                    self.data_iterator = iter(self.data_reader(rank=self.ddp_rank, world_size=self.ddp_world_size))
                    if self.buffer_fill == 0: break
                    continue

            # 4. Determine how much fits
            space_left = self.token_buffer.size(0) - self.buffer_fill
            if len(t_tokens) <= space_left:
                # It all fits!
                self.token_buffer[self.buffer_fill : self.buffer_fill + len(t_tokens)] = t_tokens
                self.buffer_fill += len(t_tokens)
            else:
                # It doesn't all fit. Take what we can and save the rest in remainder.
                self.token_buffer[self.buffer_fill : self.buffer_fill + space_left] = t_tokens[:space_left]
                self.remainder = t_tokens[space_left:] # Save for next loop/call
                self.buffer_fill += space_left

    def get_batch(self):
        req = self.B * self.T
        if self.current_pos + req + 1 > self.buffer_fill:
            self._fill_buffer()
            
        buf = self.token_buffer[self.current_pos : self.current_pos + req + 1]
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        
        self.current_pos += req
        return x, y


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