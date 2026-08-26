markdown

[![Python Version](https://img.shields.io/badge/python-3.7.16-blue.svg)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-Numba-green.svg)](https://pydata.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

---

## 🚀 Key Architectural Features

* **Zero Heavy Dependencies:** Built strictly on `numpy`, `matplotlib`, `exiftool`, and `lensfunpy`. Completely decoupled from heavyweight libraries like OpenCV or PyTorch to eliminate runtime overhead and deployment friction.
* **Deterministic VRAM Management:** Implements aggressive tile-based memory containment loop logic. It prevents out-of-memory (OOM) errors on consumer hardware by enforcing precise hardware pointer deletion (`del`) and triggering asynchronous device memory garbage collection via Numba's `defer_cleanup()`.
* **Zero Host-Device Divergence:** All invariant scalars and normalization constraints are pre-computed on the host CPU. This leaves the GPU Streaming Multiprocessors (SM) free to execute raw arithmetic without register spilling or instruction redundancy.
* **3D Pipeline Continuity:** Operates entirely on non-interleaved `float32` 3D tensor matrices `[Height, Width, Channels]` starting immediately post-demosaicing.
* **Modular Interface:** Filters can be seamlessly chained sequentially using standard idiomatic functional pipelines:
  ```python
  img = script1(img, tile_size=1024)
  img = script2(img, tile_size=1024)
  img = script3(img, tile_size=1024)
  ```

---

## 💻 Target Hardware Specification

The kernels and execution grids are mathematically tailored and optimized for the following baseline hardware profile:
* **GPU:** NVIDIA GeForce GTX 980M (Maxwell Architecture, 8 GB VRAM, 1536 CUDA Cores)
* **CPU:** Intel Core i7-6700HQ @ 2.60GHz (4 Cores, 8 Threads)
* **System RAM:** 16.0 GB
* **OS:** Windows / Linux 64-bit (Anaconda3 Environment)

---

## 📊 Performance Benchmark
<img width="2400" height="1350" alt="benchmark_s_curve" src="https://github.com/user-attachments/assets/68f30f45-bf2d-4319-9b0f-77ebfb5eab85" />

Below is the execution profile comparing the native **vectorized CPU engine (NumPy)** against our **parallelized GPU engine (Numba CUDA)** processing a `2048 x 2048 x 3` `float32` matrix with a dynamic tile allocation size of `1024`.

### Metrics Summary
* **CPU Execution Time:** ~10x to 15x slower due to sequential thread scheduling.
* **GPU Execution Time:** Sub-millisecond performance post-JIT compilation.
* **Hardware Speedup:** **>10.0x throughput acceleration** while maintaining a hard-capped VRAM overhead.

> *Note on Benchmarking Rigor: The JIT (Just-In-Time) compilation overhead of Numba is strictly excluded from our performance metrics by executing an un-timed hardware warm-up pass prior to launching `time.perf_counter()`.*

---

## 🛠️ Project Structure & Execution

Every processing node in this repository is designed as a self-contained, end-to-end executable script containing its own synthetic test generator to ensure instant portability during review.

### Running the Benchmark
To run the performance evaluation tool and verify the host-to-device streaming throughput, execute:
```bash
python Benchmark_s_curve_01.py
```

### Script Naming & Pipeline Sequence
The pipeline filters are structured numerically to track their position within the operational image flow:
1. `Cuda_s_curve_01.py` / `s_curve_01.py` — Sigmoidal midtone contrast adaptation (GPU/CPU).
2. `[Next_Script].py` — *Pipeline extension pending execution.*

---

## 🛡️ Legal Compliance & Security

* **Dependency Auditing:** This repository is compliant with software composition analysis standards. It has been validated using `scancode-toolkit`, `scanoss`, and `pip-licenses` to ensure all underlying algorithms conform strictly to permissive open-source licenses (Apache 2.0).
* **Metadata Sanitization:** All sample images used within testing modules have been programmatically scrubbed of private EXIF data, GPS geolocation parameters, and camera serial numbers using `exiftool`.

---

## 📄 License

This project is licensed under the Apache 2.0 - see the [LICENSE](LICENSE) file for details.
