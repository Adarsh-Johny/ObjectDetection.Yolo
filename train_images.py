import torch
import utils
display = utils.notebook_init()  # checks

print(torch.__version__)
print(torch.cuda.is_available())
