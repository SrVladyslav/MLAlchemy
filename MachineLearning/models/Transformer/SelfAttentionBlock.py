import torch 
import torch.nn as nn 
from torch.nn import functional as F

# ============================================== HYPERPARAMETERS
batch_size = 64 # How many parallel sequences will we process?
block_size = 256 # What is the max context prediction length? 
max_iters = 500 
eval_iters = 200
eval_interval = 500
learning_rate = 1e-4 # Self attention can't tolerate very hight learning rates 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 384 / 6 = 64 -> Every head has 64 dimensional
n_embd = 384 # Number of embedding directions
n_heads = 6
n_layer = 6
dropout = 0.2
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

class Head(nn.Module):
    """One head of the self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # This is not a parameter of the model, so instead of the naming it like always 
        # You need to register it as buffer
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        B, T, C = x.shape 
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # Compute attention scores (Affinities)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        # Here we make sute that future does not communicate with the past
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=1) # (B,T,T)
        # Dropout here when we calculate the affinities after the softmax,
        # So we are randomly prevent some of the nodes to communicating each other
        # Dropout: A simple way to prevent neural networks from overfitting
        wei = self.dropout(wei)
        # Perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple Heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        # We are jsut creating an array of multiple heads 
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout) 
    
    def forward(self, x):
        # The Self-attention output itself
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, C)
        # We actually want to apply the projection, that is only the linear
        # transformations of the outcome of the self attention layer
        out = self.dropout(
            self.projection(out)
        )
        return out

class FeedForward(nn.Module):
    """ a Simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4* n_embd, n_embd), # This is th eprojection layer going back to the residual pathway
            # That we can add right before the residual connection back into resiadual pathway
            # so
            nn.Dropout(dropout) 
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: comunication (self-attention)
        followed by computation (feed-forward)
    """
    def __init__(self, n_embd, n_heads):
        # n_embd: Embedding dimension, 
        # n_heads: The number of heads we had like
        super().__init__()
        head_size = n_embd // n_heads 
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.feed_fwd = FeedForward(n_embd)
        # The mean and the variance take over n_embd (32 in our case) numbers
        # So the batch in the time act as batch dimensions, both of them
        self.layer_norm_1 = nn.LayerNorm(n_embd) # 32
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x+ are the residual connections
        x = x + self.sa(
            # Per token transformation that normalizes the features and makes them unit mean,
            # Unit gaussian at initialization, but as we have gamma and beta trainable parameters 
            # in it, during training it can creates outputs that might not be unit gaussian, 
            # but the optimiaztion will determine that
            self.layer_norm_1(x) # Pre-norm: Applying the Layer norm before self-attention & ffwd
        )
        x = x + self.feed_fwd(
            self.layer_norm_2(x) # Pre-norm: Applying the Layer norm before self-attention & ffwd
        )
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()  
        # Each token directly reads off the logits for the next token from a lookup table 
        # This creates a token embedding table of size: vocab_size x n_embd
        # bawsically is a tensor of vocab_size x n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Multihead attention is just applying multiple attentions in parallel and 
        # concatenating the results
        # self.sa_head = Head(n_embd)
        # This creates multiple channels of communication, and then gather all this data to one
        # self.sa_heads = MultiHeadAttention(4, n_embd // 4) # i.e. 4 heads of 8-dimensional self-attention
        # self.feed_fwd = FeedForward(n_embd)
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_heads=4), # n_embd=32, n_head=4 --> 32/4=8
        #     Block(n_embd, n_heads=4),
        #     Block(n_embd, n_heads=4),
        #     # Distribual layer norm, tipically at the end of the transformer
        #     # For the final linear layer that decodes to the vocabulary
        #     nn.LayerNorm(n_embd) 
        # )
        self.blocks = nn.Sequential(
            # i.e. n_embd=32, n_head=4 --> 32/4=8
            *[Block(n_embd, n_heads=n_heads) for _ in range(n_layer)]
        )
        # Distribual layer norm, tipically at the end of the transformer
        # For the final linear layer that decodes to the vocabulary
        self.ln_final = nn.LayerNorm(n_embd)  
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor integers 
        # To go from token embeddings to logits we'll need a linear layer
        tok_emb = self.token_embedding_table(idx) # (Batch=4, Time=8, Channel=65) tensor
        # Integers from 0 to T-1
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C) 
        # X holds not just token identity but also the position where the token occur  
        # Sums pos_emb for each tok_emb batch
        x = tok_emb + pos_emb # (B, T, C)
        # This is on the token level, all the token do this independently, 
        # The self-attention is the communication 
        # x = self.sa_heads(x) # Apply one head os self-attention (B,T,C)
        # This data, is now gathered and this feed forward thinks about this data individually
        # x = self.feed_fwd(x) # (B, T, C)
        x = self.blocks(x)  # (B, T, C) -> Disperse communication + ffwd many times
        x = self.ln_final(x)  # (B, T, C) -> Layer normalization
        logits = self.lm_head(x) # (B,T,vocab_size) -> Finally we decode
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
            # Crops the idx to the last block_size tokens so we will not 
            # have an overflow in the embeddings table
            idx_cond = idx[:, -block_size:]
            # Get the predictions by calling the forward function 
            logits, loss = self(idx_cond)
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

# Saving the model
torch.save(model.state_dict(), "./../weights/modelWeight.pt")
