import numpy as np
import torch

a = torch.tensor([1, 2, 3],device="cuda")

print(a.cpu().numpy().tolist())