# SDV on PyTorch 2.7 + CUDA 12.8 Test

Minimal test project for SDV TVAE compatibility with PyTorch 2.7.0 and CUDA 12.8 on NVIDIA RTX 5070 Ti (sm_120).

## Purpose

Test if:
1. PyTorch 2.7.0 + CUDA 12.8 supports RTX 5070 Ti (Blackwell sm_120)
2. SDV TVAE works with PyTorch 2.7+
3. GPU acceleration works for training and generation
4. Model saving/loading works (SDV pickle format used by sdpype)

## Hardware Target

- **GPU**: NVIDIA GeForce RTX 5070 Ti
- **Compute Capability**: 12.0 (sm_120, Blackwell architecture)
- **Driver**: CUDA 13.0+ compatible

## Setup

```bash
# Install dependencies with uv
uv sync

# Verify GPU is detected
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Run Test

```bash
uv run python test_tvae.py
```

## Expected Output

- PyTorch detects RTX 5070 Ti
- No sm_120 compatibility warnings
- TVAE trains on GPU
- Model saves successfully
- Model loads successfully
- Generation works on GPU
- Synthetic data matches schema

## Dependencies

- Python 3.11
- PyTorch 2.7.0 (CUDA 12.8)
- SDV 1.0+
- pandas, numpy

## References

- [PyTorch 2.7 Release Blog](https://pytorch.org/blog/pytorch-2-7/)
- [Stack Overflow: PyTorch CUDA 12.8 Install](https://stackoverflow.com/questions/79513544/how-to-install-pytorch-2-6-with-cuda-12-8-using-pip)
