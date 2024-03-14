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

# Obtaining all the set of characters that are ocurring in the text
# The vocabulary are the possible characters that the model can see or emit
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Strategy to tokenize the text: convert the RAW text to some sequence 
# of integers according to some vocabulary of elements, In other words:
# mapping the characters to integers, so we can obtain a encoder and decoder
stoi = { character:i for i,character in enumerate(chars)}
itos = {i:character for i,character in enumerate(chars)}
encode = lambda s: [ stoi[c] for c in s ]
decode = lambda s: ''.join([itos[i] for i in s])

# Training and testing/validation data splits
# Our split will be 90% for training set and 10% for testing/validation
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
valid_data = data[n:]

# ============================================== Data loading 
# When fidding to the GPU, we will have many batches of chunks of text
# -> That's for take GPU busy, they're very good on parallel processing of data
# We want to proces multipole chuks, all at the same time, but they're processed completely
# independently and do not talk with each other
def get_batch(split):
    # Generates a small batch of data of inputs x and targets y
    data = train_data if split =='train' else valid_data
    # Size between zero and random block size
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # When we load data and use cuda, we need to make sure we move it to the
    # device
    x,y = x.to(device), y.to(device) # 
    return x, y



# ============================================== Estimate loss
# This noGrad says pytorch to not call .backward(), so it will be more optimized 
# because pytorch do not need to store all intermediate variables, 
# a lot more efficient this way
@torch.no_grad() 
def estimate_loss():
    # Averages the loss over multiple batches, so we make it less noisy
    out = {}
    model.eval() 
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] =  loss.item()
        out[split] = losses.mean()
    model.train() 
    return out



class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()  
        # Each token directly reads off the logits for the next token from a lookup table 
        # This creates a token embedding table of size: vocab_size x vocab_size
        # bawsically is a tensor of vocab_size x vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor integers 
        logits = self.token_embedding_table(idx) # (Batch=4, Time=8, Channel=65) tensor
        # Channel and vocab size are the same, The logits 
        # are basically the scores for the next character sequence
        # Here we predict what comes next just on individual identity of this single token
        # This tokens do not see any context yet.
        # ------------------------------
        if targets is None:
            loss = None # Will return logits in (B, T, C) form and None
        else:
            # Obtaining the loss, negative log likelihood, between preddictions and targets 
            # Quality of logits respect targets, we have the identity of the next char, so how 
            # well we can predict it.
            # We also need to reshape the logits B,T,C to B,C,T in order to fit torch function
            B,T,C = logits.shape 
            # We stretch the logits into 2 dimensional tensor to conform better pythorch def
            # hERE we can evaluate the quality of the model on some data
            logits = logits.view(B*T, C) # Becomes (B, C)
            targets = targets.view(B*T) # -1 is also valid
            loss = F.cross_entropy(logits, targets) # -ln(1/65) == 4.174 should be the ideal loss
        return logits, loss 
    
    #Generate the model 
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions by calling the forward function 
            logits, loss = self(idx)
            # focus only on the last time step 
            logits = logits[:, -1, :] # Becomes (B, C)
            # Obtaining the probabilities by applying the softmax 
            probs = F.softmax(logits, dim=1) # (B, C)
            # Sample from the distribution to obtain new characters in the sequence 
            idx_next = torch.multinomial(probs, num_samples= 1) # (B, 1)
            # Append sampled index to the current sequence, 
            # Concatenating along the first dim, which is the Time dimension (T=8)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
model = BigramLanguageModel(vocab_size)
# When we create the model and we use Cuda, we want to move 
# its parameters to the device also
m = model.to(device)    


# ============================================== PyTorch optimizer 
# Normally you will use 1e-3 o 1e-4 for networks, 
# but for smaller ones you can use bigger ones
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for step in range(max_iters):
    # Every onclein a while evaluate the loss on train and validation sets 
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {step}: train loss {losses['train']:.4f}, valid loss {losses['valid']:.4f}")
    
    # Obtain a sample of the data
    xb, yb = get_batch('train')
    
    # Evaluating the loss 
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True) # Zeroing all the gradients from the previous step
    loss.backward() # Getting the gradients for all the parameters
    optimizer.step() # Using those gradients to update the parameters

# Generate from the model 
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))