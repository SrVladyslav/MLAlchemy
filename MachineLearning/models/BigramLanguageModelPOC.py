import torch 
import torch.nn as nn 
from torch.nn import functional as F

# ============================================== HYPERPARAMETERS
batch_size = 32 # How many parallel sequences will we process?
block_size = 8 # What is the max context prediction length? 
max_iters = 3000 
eval_iters = 200
eval_interval = 300
learning_rate = 1e-2 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ===============================================================

torch.manual_seed(1337)

# Obtaining the data
data_url = "./../../data/tinyshakespeare.txt" 
with open(data_url, 'r', encoding='utf-8') as f:
    text = f.read()

print(text[:100])

