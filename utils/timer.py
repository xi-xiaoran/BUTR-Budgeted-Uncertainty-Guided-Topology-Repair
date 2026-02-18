import time
import torch

class CUDATimer:
    def __init__(self, use_cuda: bool=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.t0 = None

    def __enter__(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.use_cuda:
            torch.cuda.synchronize()

    def ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000.0
