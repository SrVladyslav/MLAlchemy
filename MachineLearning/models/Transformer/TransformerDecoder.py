"""
This is the implementations of a decoder block of a transformer
based on Attention Is All You Need paper. 
"""

import torch 
import torch.nn as nn 
from torch.nn import functional as F 

# ================================================== HYPERPARAMETERS
 
# ==================================================================