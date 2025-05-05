import torch
print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.device_count())  # Should print: 1 (or more if multiple GPUs)
print(torch.cuda.get_device_name(0))  # Should print: NVIDIA GeForce GTX 1650 Ti
