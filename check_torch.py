import torch
print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None))
print("avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("dev", torch.cuda.get_device_name(0))
