import torch
f, t = torch.cuda.mem_get_info()
print("driver: free %.2f GB / total %.2f GB" % (f/1e9, t/1e9))
print("torch : allocated %.2f GB  reserved %.2f GB" % (torch.cuda.memory_allocated()/1e9, torch.cuda.memory_reserved()/1e9))
