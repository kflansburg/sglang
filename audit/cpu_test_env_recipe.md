# Pure-CPU recipe to run HiCache unit tests (used for C4 repro)

1. pip install --user torch --index-url https://download.pytorch.org/whl/cpu
2. pip install --user pytest orjson msgspec pybase64 fastapi uvicorn pydantic psutil \
   zmq aiohttp requests tqdm pillow cachetools cloudpickle blake3 nvidia-ml-py \
   setproctitle triton IPython "transformers==5.19.0" torchvision --index-url .../cpu \
   dill blobfile einops tiktoken sentencepiece openai partial_json_parser jsonlines \
   py-spy scipy compressed_tensors gguf datasets
3. pip install --user -e python/ --no-deps --no-build-isolation   # registers sglang
4. Create stub ~/.local/.../site-packages/sgl_kernel.py (lazy-raising stub; see git
   history of agent/prefetch-ctl-audit audit/ if needed) — PyPI sgl-kernel is
   CUDA-only and its import fails without libnvrtc.
5. apt-get install -y ninja-build libssl-dev python3.13-dev (native-hash JIT); ensure ~/.local/bin (pip ninja) is on PATH.
6. `hostname scpu` first if 2-rank gloo tests run (long sandbox hostname breaks gloo FileStore sockets).
7. Run tests with SGLANG_USE_CPU_ENGINE=1 (NOT SGLANG_PLATFORM=cpu; cpu platform is
   built-in behind that env var — no entry point).
