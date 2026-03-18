import torch

# Basic CUDA check
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Devices: {torch.cuda.device_count()}")

# Deep CUDA verification
if torch.cuda.is_available():
    t = torch.randn(3,3).cuda()
    print(f"\nTensor on GPU: {t.device}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Memory Usage: {torch.cuda.memory_allocated()/1e6:.2f}MB")
else:
    print("\nCUDA UNAVAILABLE - CRITICAL ISSUE")
