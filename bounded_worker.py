"""Local resource bound for a standard vLLM worker; model math is unchanged."""
import json
import os

import torch
from vllm.v1.worker.gpu_worker import Worker


class BoundedWorker(Worker):
    def load_model(self, *args, **kwargs):
        fraction = float(os.environ["GEMMA4_JEV_TORCH_FRACTION"])
        if not 0 < fraction <= 0.5:
            raise ValueError("Unsafe allocator fraction")
        torch.cuda.set_per_process_memory_fraction(fraction, self.device)
        result = super().load_model(*args, **kwargs)
        print(json.dumps({"event": "allocator_after_load",
                          "fraction": fraction,
                          "allocated_gib": torch.cuda.memory_allocated(self.device) / 1024**3,
                          "reserved_gib": torch.cuda.memory_reserved(self.device) / 1024**3}), flush=True)
        return result
