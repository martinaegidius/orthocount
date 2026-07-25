import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

x = torch.zeros(13,13).cuda()
print(x.device)
