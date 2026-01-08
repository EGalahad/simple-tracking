import logging
from typing import Dict
import torch
from torch import Tensor


class ReplayBuffer:
    def __init__(self, capacity, device="cuda"):
        super().__init__()

        self.device = device
        self.capacity = capacity
        self.buffer: Dict[str, Tensor] = {}
        self.buffer_idx = 0
        self.buffer_size = 0

        self._initialized = False
        self._logger = logging.getLogger(__name__)

    def insert(self, random_replace=False, **data):
        """Insert a new data point into the buffer."""
        if not self._initialized:
            for key, value in data.items():
                value_shape = value.shape[1:]
                self.buffer[key] = torch.zeros(
                    (self.capacity,) + value_shape, device=self.device
                )
            self._initialized = True

            # Calculate and log memory usage details
            total_bytes = 0
            for key, tensor in self.buffer.items():
                bytes_per_tensor = tensor.nelement() * tensor.element_size()
                total_bytes += bytes_per_tensor
                memory_mb = bytes_per_tensor / (1024 * 1024)  # Convert to MB
                self._logger.info(f"Buffer tensor '{key}' size: {memory_mb:.2f} MB")

            total_memory_mb = total_bytes / (1024 * 1024)  # Convert to MB
            self._logger.info(
                f"ReplayBuffer total memory usage: {total_memory_mb:.2f} MB"
            )

        mini_batch_size = data[list(data.keys())[0]].shape[0]

        # first_part_size = min(mini_batch_size, self.capacity - self.buffer_idx)
        # second_part_size = mini_batch_size - first_part_size

        # for key, value in data.items():
        #     self.buffer[key][self.buffer_idx:self.buffer_idx+first_part_size] = value[:first_part_size]
        #     self.buffer[key][:second_part_size] = value[first_part_size:]

        # self.buffer_idx = (self.buffer_idx + mini_batch_size) % self.capacity
        # self.buffer_size = min(self.buffer_size + mini_batch_size, self.capacity)

        if self.buffer_size + mini_batch_size <= self.capacity:
            # Buffer not full, insert at the end
            start_idx = self.buffer_size
            for key, value in data.items():
                self.buffer[key][start_idx : start_idx + mini_batch_size] = value
            self.buffer_size += mini_batch_size
        else:
            # Buffer is full or will overflow, randomly replace some entries
            if mini_batch_size >= self.capacity:
                # If new data is larger than capacity, take the last `capacity` samples
                for key, value in data.items():
                    self.buffer[key] = value[-self.capacity :]
                self.buffer_size = self.capacity
            else:
                # Randomly select positions to replace
                replace_indices = torch.randint(
                    0, self.buffer_size, (mini_batch_size,), device=self.device
                )
                for key, value in data.items():
                    self.buffer[key][replace_indices] = value

                # If buffer wasn't full before, update buffer size
                if self.buffer_size < self.capacity:
                    self.buffer_size = min(
                        self.buffer_size + mini_batch_size, self.capacity
                    )
        # print(f"Inserted {mini_batch_size} samples into ReplayBuffer. Current size: {self.buffer_size}/{self.capacity}")
        # breakpoint()

    def sample(self, batch_size):
        """Sample a batch of data from the buffer."""
        if self.buffer_size == 0:
            raise ValueError("No samples in buffer")

        if self.buffer_size < batch_size:
            batch_size = self.buffer_size

        indices = torch.randint(0, self.buffer_size, (batch_size,), device=self.device)
        return {key: self.buffer[key][indices] for key in self.buffer.keys()}
