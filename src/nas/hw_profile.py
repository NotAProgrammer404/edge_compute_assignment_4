"""Real hardware measurements used by the constrained NAS objective."""
import io
import time
import tracemalloc

import torch


def measure_latency(model, input_size=(1, 3, 32, 32), device="cpu", reps=30, warmup=5):
    """Mean inference latency in milliseconds on the requested device."""
    model = model.to(device).eval()
    x = torch.randn(*input_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                model(x)
            torch.cuda.synchronize()
        else:
            t0 = time.perf_counter()
            for _ in range(reps):
                model(x)
        elapsed = time.perf_counter() - t0
    return 1000.0 * elapsed / reps


def measure_size_kb(model):
    """Serialized state-dict size in KiB (model storage size)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue()) / 1024.0


def measure_peak_kb(model, input_size=(1, 3, 32, 32)):
    """Peak Python-side allocator usage during a forward pass."""
    model = model.to("cpu").eval()
    x = torch.randn(*input_size)
    tracemalloc.start()
    with torch.no_grad():
        model(x)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024.0
