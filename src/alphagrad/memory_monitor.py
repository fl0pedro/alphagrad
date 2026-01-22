import jax
import jax.numpy as jnp
import threading
from typing import Optional, List, Any
from jax.profiler import device_memory_profile
import gzip

import profile_pb2 
# `profile_pb2` generated via `protoc --python_out=. profile.proto`
# `profile.proto` downloaded from: https://raw.githubusercontent.com/google/pprof/main/proto/profile.proto

def mem_diff(prof1: profile_pb2.Profile, prof2: profile_pb2.Profile | None = None) -> int:
    prof_set1 = set((tuple(x.location_id), x.value[1]) for x in prof1.sample)
    if prof2 is None:
        return sum(x[1] for x in prof_set1)
    prof_set2 = set((tuple(x.location_id), x.value[1]) for x in prof2.sample)
    return sum(-x[1] if x in prof_set2 else x[1] for x in prof_set2|prof_set1)

def decode_pprof(data: bytes) -> profile_pb2.Profile:
    try:
        data = gzip.decompress(data)
    except (TypeError, EOFError):
        data = data

    profile = profile_pb2.Profile()
    profile.ParseFromString(data)
    return profile

def device_memory() -> int:
    try:
        pprof = device_memory_profile()
        prof = decode_pprof(pprof)
        return mem_diff(prof)
    except Exception as e:
        print(f"Error: {e}\nfailed to read memory, returning 0")
        return 0

class PeakMemoryMonitor:
    def __init__(
        self,
        interval: float = 0.01,
        base_memory: int = 0
    ):
        self.interval = interval
        self._base_memory = base_memory
        self._samples: List[float] = []
        self._stop_flag = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._peak = 0

    def _monitor(self):
        while not self._stop_flag.is_set():
            try:
                current_usage = device_memory() + self._base_memory
                self._samples.append(current_usage)
                self._peak = max(self._peak, current_usage)
                if self.interval:
                    self._stop_flag.wait(self.interval)
            except Exception as e:
                print(f"Memory monitoring error: {e}")
                self._stop_flag.set() # Stop if error is persistent

    def __enter__(self) -> 'PeakMemoryMonitor':
        self._peak = device_memory() + self._base_memory
        self._samples = [self._peak]
        self._stop_flag.clear()
        self._monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self._monitor_thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> float:
        self._stop_flag.set()
        if self._monitor_thread:
            self._monitor_thread.join()
        
        try:
            # synchronize async jax calls
            jax.device_get(jnp.array(0.0)).block_until_ready()
        except Exception:
            pass # ignore case where .block_until_read() fails
            
        if not self._samples:
            final_peak = device_memory() + self._base_memory
            self._peak = max(self._peak, final_peak)

        self._base_memory = self._peak
        
        return self._peak

    @property
    def peak(self) -> float:
        return self._peak
