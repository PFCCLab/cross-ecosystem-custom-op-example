import torch
import extension

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
z = 2.0
result = extension.muladd(x, y, z)
print(result)  # Expected output: tensor([ 6., 12., 20.])