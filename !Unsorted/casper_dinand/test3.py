import torch
print(torch.cuda.is_available())         # True means GPU will be used
print(torch.cuda.get_device_name(0))     # e.g., "NVIDIA GeForce RTX 3060"
