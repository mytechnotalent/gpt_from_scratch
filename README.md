# GPT From Scratch
This notebook builds a complete GPT (Generative Pre-trained Transformer) model from scratch using PyTorch. It covers tokenization, self-attention, multi-head attention, transformer blocks, and text generation and all explained step-by-step with a simple nursery rhyme corpus.
#### Author: [Kevin Thomas](mailto:ket189@pitt.edu)


## Part 1: Setup and Imports

First, let's import the libraries we need:
- **torch**: The main PyTorch library for tensor operations
- **torch.nn**: Neural network modules (layers, loss functions, etc.)
- **torch.nn.functional**: Functional interface for operations like softmax


```python
import torch                        # Main PyTorch library
import torch.nn as nn               # Neural network modules
import torch.nn.functional as F     # Functional operations (softmax, relu, etc.)

# Check our PyTorch setup
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("MPS available:", torch.backends.mps.is_available())

# Determine the best available device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("Using CPU")

print(f"\nDevice: {device}")
```


**Output:**
```
Torch version: 2.9.1
CUDA available: False
MPS available: True
Using MPS (Apple Silicon)

Device: mps

```


## Part 2: Preparing the Training Data

### 2.1 Creating a Corpus

GPT models learn from text. We'll use a small corpus of simple sentences.
In real applications, this would be millions of documents!


```python
# Our training corpus - the classic nursery rhyme "Mary Had a Little Lamb"
# In production, you'd use a massive dataset (books, websites, etc.)
corpus = [
    "mary had a little lamb",
    "little lamb little lamb",
    "mary had a little lamb",
    "its fleece was white as snow",
    "and everywhere that mary went",
    "mary went mary went",
    "everywhere that mary went",
    "the lamb was sure to go",
    "it followed her to school one day",
    "school one day school one day",
    "it followed her to school one day",
    "which was against the rules",
    "it made the children laugh and play",
    "laugh and play laugh and play",
    "it made the children laugh and play",
    "to see a lamb at school"
]

# Add an <END> token to mark sentence boundaries
# This helps the model learn when to stop generating
corpus = [sentence + " <END>" for sentence in corpus]

# Combine all sentences into one continuous text
text = " ".join(corpus)
print("Combined text:")
print(text)
```


**Output:**
```
Combined text:
mary had a little lamb <END> little lamb little lamb <END> mary had a little lamb <END> its fleece was white as snow <END> and everywhere that mary went <END> mary went mary went <END> everywhere that mary went <END> the lamb was sure to go <END> it followed her to school one day <END> school one day school one day <END> it followed her to school one day <END> which was against the rules <END> it made the children laugh and play <END> laugh and play laugh and play <END> it made the children laugh and play <END> to see a lamb at school <END>

```


### 2.2 Building the Vocabulary

**Tokenization** is the process of converting text into numbers that the model can process.
We need to:
1. Find all unique words (our vocabulary)
2. Assign each word a unique number (index)
3. Create mappings to convert between words and indices


```python
# Get all unique words in our text
# set() removes duplicates, list() converts back to a list
words = list(set(text.split()))
print(f"Unique words ({len(words)} total):")
print(words)

# Our vocabulary size is the number of unique words
vocab_size = len(words)
print(f"\nVocabulary size: {vocab_size}")
```


**Output:**
```
Unique words (35 total):
['as', 'day', 'had', 'lamb', 'that', 'play', 'the', 'mary', 'rules', 'everywhere', 'one', 'children', 'school', 'a', '<END>', 'fleece', 'made', 'little', 'and', 'her', 'against', 'its', 'at', 'was', 'went', 'to', 'go', 'it', 'see', 'sure', 'white', 'which', 'laugh', 'followed', 'snow']

Vocabulary size: 35

```


```python
# Create word-to-index mapping (word2idx)
# This dictionary maps each word to a unique integer
# Example: {"hello": 0, "friends": 1, "how": 2, ...}
word2idx = {word: idx for idx, word in enumerate(words)}
print("word2idx (word → number):")
print(word2idx)

# Create index-to-word mapping (idx2word)
# This is the reverse mapping for decoding model outputs
# Example: {0: "hello", 1: "friends", 2: "how", ...}
idx2word = {idx: word for word, idx in word2idx.items()}
print("\nidx2word (number → word):")
print(idx2word)
```


**Output:**
```
word2idx (word → number):
{'as': 0, 'day': 1, 'had': 2, 'lamb': 3, 'that': 4, 'play': 5, 'the': 6, 'mary': 7, 'rules': 8, 'everywhere': 9, 'one': 10, 'children': 11, 'school': 12, 'a': 13, '<END>': 14, 'fleece': 15, 'made': 16, 'little': 17, 'and': 18, 'her': 19, 'against': 20, 'its': 21, 'at': 22, 'was': 23, 'went': 24, 'to': 25, 'go': 26, 'it': 27, 'see': 28, 'sure': 29, 'white': 30, 'which': 31, 'laugh': 32, 'followed': 33, 'snow': 34}

idx2word (number → word):
{0: 'as', 1: 'day', 2: 'had', 3: 'lamb', 4: 'that', 5: 'play', 6: 'the', 7: 'mary', 8: 'rules', 9: 'everywhere', 10: 'one', 11: 'children', 12: 'school', 13: 'a', 14: '<END>', 15: 'fleece', 16: 'made', 17: 'little', 18: 'and', 19: 'her', 20: 'against', 21: 'its', 22: 'at', 23: 'was', 24: 'went', 25: 'to', 26: 'go', 27: 'it', 28: 'see', 29: 'sure', 30: 'white', 31: 'which', 32: 'laugh', 33: 'followed', 34: 'snow'}

```


```python
# Convert our entire text into a tensor of indices
# This is the numerical representation of our training data
data = torch.tensor([word2idx[word] for word in text.split()], dtype=torch.long)

print(f"Data tensor shape: {data.shape}")
print(f"Total tokens: {len(data)}")
print(f"\nFirst 20 tokens: {data[:20]}")
print(f"Decoded: {' '.join([idx2word[int(i)] for i in data[:20]])}")
```


**Output:**
```
Data tensor shape: torch.Size([106])
Total tokens: 106

First 20 tokens: tensor([ 7,  2, 13, 17,  3, 14, 17,  3, 17,  3, 14,  7,  2, 13, 17,  3, 14, 21,
        15, 23])
Decoded: mary had a little lamb <END> little lamb little lamb <END> mary had a little lamb <END> its fleece was

```


## Part 3: Hyperparameters

These are the key settings that control our model's architecture and training:

| Parameter | Description |
|-----------|-------------|
| `block_size` | Context window - how many tokens the model can "see" at once |
| `embedding_dim` | Size of the vector representation for each token |
| `n_heads` | Number of attention heads (parallel attention mechanisms) |
| `n_layers` | Number of transformer blocks stacked together |
| `lr` | Learning rate - how fast the model learns |
| `epochs` | Number of training iterations |


```python
# Model architecture hyperparameters
block_size = 6          # Context window: model sees 6 tokens at a time
embedding_dim = 32      # Each token represented as a 32-dimensional vector
n_heads = 2             # 2 parallel attention heads
n_layers = 2            # 2 transformer blocks stacked

# Training hyperparameters
lr = 1e-3               # Learning rate (0.001)
epochs = 1500           # Number of training steps

print("Hyperparameters set!")
print(f"Context window: {block_size} tokens")
print(f"Embedding dimension: {embedding_dim}")
print(f"Attention heads: {n_heads}")
print(f"Transformer layers: {n_layers}")
```


**Output:**
```
Hyperparameters set!
Context window: 6 tokens
Embedding dimension: 32
Attention heads: 2
Transformer layers: 2

```


## Part 4: Data Loading - Creating Training Batches

During training, we feed the model:
- **Input (x)**: A sequence of `block_size` tokens
- **Target (y)**: The same sequence shifted by 1 (next token prediction)

For example, if `block_size=6` and our text starts with "hello friends how are you doing":
- x = [hello, friends, how, are, you, doing]
- y = [friends, how, are, you, doing, <next_word>]

The model learns to predict each next token given all previous tokens!


```python
def get_batch(batch_size=16):
    """
    Create a random batch of training examples.
    
    Args:
        batch_size: Number of sequences in each batch
        
    Returns:
        x: Input sequences [batch_size, block_size]
        y: Target sequences [batch_size, block_size] (shifted by 1)
    """
    # Generate random starting positions for each sequence in the batch
    # We subtract block_size to ensure we don't go past the end
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Create input sequences: tokens from position i to i+block_size
    x = torch.stack([data[i:i+block_size] for i in ix])
    
    # Create target sequences: tokens from position i+1 to i+block_size+1
    # This is the "next token" for each position in x
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x, y

# Let's see an example batch
x_example, y_example = get_batch(batch_size=2)
print("Input (x):")
print(x_example)
print(f"\nDecoded x[0]: {' '.join([idx2word[int(i)] for i in x_example[0]])}")
print(f"\nTarget (y):")
print(y_example)
print(f"\nDecoded y[0]: {' '.join([idx2word[int(i)] for i in y_example[0]])}")
```


**Output:**
```
Input (x):
tensor([[16,  6, 11, 32, 18,  5],
        [ 7,  2, 13, 17,  3, 14]])

Decoded x[0]: made the children laugh and play

Target (y):
tensor([[ 6, 11, 32, 18,  5, 14],
        [ 2, 13, 17,  3, 14, 21]])

Decoded y[0]: the children laugh and play <END>

```


## Part 5: Self-Attention - The Heart of Transformers ❤️

### What is Self-Attention?

Self-attention allows each token to "look at" all other tokens and decide which ones are most relevant. It computes:

1. **Query (Q)**: "What am I looking for?"
2. **Key (K)**: "What do I contain?"
3. **Value (V)**: "What information do I have to share?"

The attention score between two tokens is: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$

### Causal Masking

In GPT (decoder-only), we use **causal masking** to prevent tokens from attending to future tokens. A token at position $i$ can only see tokens at positions $0, 1, ..., i$. This is done using a lower triangular mask.


```python
class SelfAttentionHead(nn.Module):
    """
    A single head of self-attention.
    
    This computes attention scores between all positions in the sequence,
    allowing each token to gather information from relevant tokens.
    """
    
    def __init__(self, embedding_dim, block_size, head_size):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            block_size: Maximum sequence length (for masking)
            head_size: Dimension of this attention head's output
        """
        super().__init__()
        
        # Linear projections for Query, Key, Value
        # These learn to extract different aspects of each token
        self.key = nn.Linear(embedding_dim, head_size, bias=False)    # What do I contain?
        self.query = nn.Linear(embedding_dim, head_size, bias=False)  # What am I looking for?
        self.value = nn.Linear(embedding_dim, head_size, bias=False)  # What info do I share?
        
        # Causal mask: lower triangular matrix
        # This prevents attending to future tokens
        # register_buffer stores this as a non-trainable parameter
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        """
        Forward pass of self-attention.
        
        Args:
            x: Input tensor of shape [Batch, Time, Channels]
               Batch = number of sequences
               Time = sequence length (number of tokens)
               Channels = embedding dimension
               
        Returns:
            Output tensor of shape [Batch, Time, head_size]
        """
        B, T, C = x.shape  # Batch, Time (sequence length), Channels (embedding dim)
        
        # Compute Key and Query projections
        k = self.key(x)    # [B, T, head_size] - What each token contains
        q = self.query(x)  # [B, T, head_size] - What each token is looking for
        
        # Compute attention scores: (Q @ K^T) / sqrt(d_k)
        # Q @ K^T gives us an [T, T] matrix of attention weights
        # Dividing by sqrt(C) prevents scores from becoming too large
        wei = q @ k.transpose(-2, -1) / (C ** 0.5)  # [B, T, T]
        
        # Apply causal mask: set future positions to -infinity
        # After softmax, -inf becomes 0, so no attention to future
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Softmax normalizes scores to probabilities (sum to 1)
        wei = F.softmax(wei, dim=-1)  # [B, T, T]
        
        # Compute Value projection and apply attention weights
        v = self.value(x)  # [B, T, head_size] - Information to aggregate
        out = wei @ v      # [B, T, head_size] - Weighted sum of values
        
        # Return the output of the attention head
        return out

# Test our attention head
test_head = SelfAttentionHead(embedding_dim, block_size, head_size=16)
test_input = torch.randn(2, block_size, embedding_dim)  # Random input
test_output = test_head(test_input)
print(f"Input shape: {test_input.shape}")
print(f"Output shape: {test_output.shape}")
```


**Output:**
```
Input shape: torch.Size([2, 6, 32])
Output shape: torch.Size([2, 6, 16])

```


### Let's Visualize the Causal Mask

The mask ensures token at position $i$ can only attend to positions $\leq i$:


```python
# Visualize the causal (triangular) mask
mask = torch.tril(torch.ones(block_size, block_size))
print("Causal Mask (1 = can attend, 0 = masked):")
print(mask)
print("\nInterpretation:")
print("- Row i shows which positions token i can attend to")
print("- Token 0 can only see itself")
print("- Token 1 can see tokens 0 and 1")
print("- Token 5 can see all 6 tokens")
```


**Output:**
```
Causal Mask (1 = can attend, 0 = masked):
tensor([[1., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1.]])

Interpretation:
- Row i shows which positions token i can attend to
- Token 0 can only see itself
- Token 1 can see tokens 0 and 1
- Token 5 can see all 6 tokens

```


## Part 6: Multi-Head Attention

**Why multiple heads?** Each attention head can learn to focus on different types of relationships:
- One head might learn syntactic relationships (subject-verb)
- Another might learn semantic relationships (synonyms)
- Another might learn positional patterns

We run multiple attention heads in parallel and concatenate their outputs!


```python
class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention running in parallel.
    
    Each head has its own Q, K, V projections and learns different patterns.
    Outputs are concatenated and projected back to embedding dimension.
    """
    
    def __init__(self, embedding_dim, block_size, num_heads):
        """
        Args:
            embedding_dim: Total embedding dimension
            block_size: Maximum sequence length
            num_heads: Number of parallel attention heads
        """
        super().__init__()
        
        # Each head gets a fraction of the total embedding dimension
        # Example: embedding_dim=32, num_heads=2 → head_size=16
        head_size = embedding_dim // num_heads
        
        # Create a list of attention heads
        # ModuleList registers these as submodules for proper parameter tracking
        self.heads = nn.ModuleList([
            SelfAttentionHead(embedding_dim, block_size, head_size) 
            for _ in range(num_heads)
        ])
        
        # Output projection: combines all head outputs back to embedding_dim
        self.proj = nn.Linear(num_heads * head_size, embedding_dim)
    
    def forward(self, x):
        """
        Run all attention heads and combine their outputs.
        
        Args:
            x: Input tensor [Batch, Time, embedding_dim]
            
        Returns:
            Output tensor [Batch, Time, embedding_dim]
        """
        # Run each head and concatenate outputs along the last dimension
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        
        # Project back to embedding dimension
        return self.proj(out)

# Test multi-head attention
test_mha = MultiHeadAttention(embedding_dim, block_size, n_heads)
test_output = test_mha(test_input)
print(f"Input shape: {test_input.shape}")
print(f"Output shape: {test_output.shape}")
print(f"\nWith {n_heads} heads, each head has dimension: {embedding_dim // n_heads}")
```


**Output:**
```
Input shape: torch.Size([2, 6, 32])
Output shape: torch.Size([2, 6, 32])

With 2 heads, each head has dimension: 16

```


## Part 7: Feed-Forward Network

After attention, each token passes through a simple feed-forward network independently. This adds non-linearity and allows the model to process the attended information.

The structure is:
1. **Linear expansion**: `embedding_dim` → `4 * embedding_dim`
2. **ReLU activation**: Adds non-linearity
3. **Linear projection**: `4 * embedding_dim` → `embedding_dim`


```python
class FeedForward(nn.Module):
    """
    A simple feed-forward network applied to each token position.
    
    This is a 2-layer MLP that expands the dimension by 4x,
    applies non-linearity, and projects back to the original dimension.
    """
    
    def __init__(self, n_embd):
        """
        Args:
            n_embd: Embedding dimension (input and output size)
        """
        super().__init__()
        
        # Sequential container for the feed-forward layers
        self.net = nn.Sequential(
            # Expand: n_embd → 4*n_embd (the "inner" dimension)
            nn.Linear(n_embd, 4 * n_embd),
            
            # Non-linearity: ReLU(x) = max(0, x)
            # This allows the network to learn non-linear patterns
            nn.ReLU(),
            
            # Project back: 4*n_embd → n_embd
            nn.Linear(4 * n_embd, n_embd)
        )
    
    def forward(self, x):
        """
        Apply feed-forward network to each position independently.
        
        Args:
            x: Input tensor [Batch, Time, n_embd]
            
        Returns:
            Output tensor [Batch, Time, n_embd]
        """
        # Return the output of the feed-forward network
        return self.net(x)

# Test feed-forward
test_ff = FeedForward(embedding_dim)
test_output = test_ff(test_input)
print(f"Input shape: {test_input.shape}")
print(f"Output shape: {test_output.shape}")
print(f"\nIntermediate dimension: {4 * embedding_dim}")
```


**Output:**
```
Input shape: torch.Size([2, 6, 32])
Output shape: torch.Size([2, 6, 32])

Intermediate dimension: 128

```


## Part 8: Transformer Block

A Transformer Block combines:
1. **Multi-Head Attention** with residual connection and layer normalization
2. **Feed-Forward Network** with residual connection and layer normalization

**Residual Connections** (the `x + ...`) help with gradient flow during training.

**Layer Normalization** stabilizes training by normalizing activations.

```
Input → LayerNorm → Multi-Head Attention → + (residual)
                                            ↓
                    LayerNorm → FeedForward → + (residual) → Output
```


```python
class Block(nn.Module):
    """
    A single Transformer block.
    
    Combines multi-head self-attention with a feed-forward network,
    using residual connections and layer normalization.
    """
    
    def __init__(self, embedding_dim, block_size, n_heads):
        """
        Args:
            embedding_dim: Embedding dimension
            block_size: Maximum sequence length
            n_heads: Number of attention heads
        """
        super().__init__()
        
        # Self-attention: allows tokens to communicate
        self.sa = MultiHeadAttention(embedding_dim, block_size, n_heads)
        
        # Feed-forward: processes each token independently
        self.ffwd = FeedForward(embedding_dim)
        
        # Layer normalization layers
        # These normalize the activations to have mean=0 and std=1
        self.ln1 = nn.LayerNorm(embedding_dim)  # Before attention
        self.ln2 = nn.LayerNorm(embedding_dim)  # Before feed-forward
    
    def forward(self, x):
        """
        Forward pass through the transformer block.
        
        Uses "pre-norm" architecture: LayerNorm is applied before each sub-layer.
        Residual connections add the input to the output of each sub-layer.
        
        Args:
            x: Input tensor [Batch, Time, embedding_dim]
            
        Returns:
            Output tensor [Batch, Time, embedding_dim]
        """
        # Self-attention with residual connection
        # x = x + attention(normalize(x))
        x = x + self.sa(self.ln1(x))
        
        # Feed-forward with residual connection
        # x = x + feedforward(normalize(x))
        x = x + self.ffwd(self.ln2(x))
        
        # Return the output of the transformer block
        return x

# Test the transformer block
test_block = Block(embedding_dim, block_size, n_heads)
test_output = test_block(test_input)
print(f"Input shape: {test_input.shape}")
print(f"Output shape: {test_output.shape}")
print("\nThe transformer block preserves the shape!")
```


**Output:**
```
Input shape: torch.Size([2, 6, 32])
Output shape: torch.Size([2, 6, 32])

The transformer block preserves the shape!

```


## Part 9: The Complete GPT Model 🎉

Now we put everything together into a complete GPT model!

**Architecture:**
1. **Token Embedding**: Convert token IDs to vectors
2. **Position Embedding**: Add positional information
3. **Transformer Blocks**: Stack of N transformer blocks
4. **Final LayerNorm**: Normalize the final output
5. **Language Model Head**: Project to vocabulary size for prediction

```
Token IDs → Token Embedding + Position Embedding
              ↓
        Transformer Block 1
              ↓
        Transformer Block 2
              ↓
          ... (N blocks)
              ↓
         Layer Norm
              ↓
         Linear Head → Logits (vocab_size)
```


```python
class TinyGPT(nn.Module):
    """
    A tiny GPT (Generative Pre-trained Transformer) model.
    
    This is a decoder-only transformer that predicts the next token
    given a sequence of previous tokens.
    """
    
    def __init__(self):
        super().__init__()
        
        # Token Embedding: maps each token ID to a dense vector
        # Shape: vocab_size → embedding_dim
        # Example: word ID 5 → [0.2, -0.1, 0.8, ...] (32 dimensions)
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Position Embedding: encodes the position of each token
        # Shape: block_size → embedding_dim
        # Position 0 has embedding [a, b, c, ...], Position 1 has [d, e, f, ...], etc.
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        
        # Stack of Transformer Blocks
        # nn.Sequential applies them one after another
        self.blocks = nn.Sequential(
            *[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]
        )
        
        # Final Layer Normalization
        self.ln_f = nn.LayerNorm(embedding_dim)
        
        # Language Model Head: projects to vocabulary size
        # Outputs "logits" (unnormalized scores) for each word in vocabulary
        self.head = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, idx, targets=None):
        """
        Forward pass of the GPT model.
        
        Args:
            idx: Input token indices [Batch, Time]
            targets: Target token indices [Batch, Time] (optional, for training)
            
        Returns:
            logits: Prediction scores [Batch, Time, vocab_size]
            loss: Cross-entropy loss (if targets provided)
        """
        # Get batch size and sequence length
        B, T = idx.shape  # Batch size, Sequence length
        
        # Get token embeddings: [B, T] → [B, T, embedding_dim]
        tok_emb = self.token_embedding(idx)
        
        # Get position embeddings: [T] → [T, embedding_dim]
        # torch.arange(T) creates [0, 1, 2, ..., T-1]
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        
        # Add token and position embeddings
        # Each token now knows both its identity and position
        x = tok_emb + pos_emb  # [B, T, embedding_dim]
        
        # Pass through transformer blocks
        x = self.blocks(x)  # [B, T, embedding_dim]
        
        # Final layer normalization
        x = self.ln_f(x)  # [B, T, embedding_dim]
        
        # Project to vocabulary size
        logits = self.head(x)  # [B, T, vocab_size]
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Reshape for cross_entropy: need [N, C] and [N]
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)  # Flatten to [B*T, vocab_size]
            targets_flat = targets.view(B * T)   # Flatten to [B*T]
            
            # Cross-entropy loss: measures how well predictions match targets
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        # Return logits and loss
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens autoregressively.
        
        Args:
            idx: Starting tokens [Batch, Time]
            max_new_tokens: Number of new tokens to generate
            
        Returns:
            Extended sequence [Batch, Time + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop to last block_size tokens (context window limit)
            idx_cond = idx[:, -block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)  # [B, T, vocab_size]
            
            # Focus on the last token's prediction
            logits = logits[:, -1, :]  # [B, vocab_size]
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)  # [B, vocab_size]
            
            # Sample from the probability distribution
            # multinomial samples indices based on their probabilities
            next_idx = torch.multinomial(probs, 1)  # [B, 1]
            
            # Append the new token to the sequence
            idx = torch.cat((idx, next_idx), dim=1)  # [B, T+1]
        
        # Return the extended sequence
        return idx

# Create the model
model = TinyGPT()

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"TinyGPT Model created!")
print(f"Total parameters: {num_params:,}")
print(f"\nModel architecture:")
print(model)
```


**Output:**
```
TinyGPT Model created!
Total parameters: 27,747

Model architecture:
TinyGPT(
  (token_embedding): Embedding(35, 32)
  (position_embedding): Embedding(6, 32)
  (blocks): Sequential(
    (0): Block(
      (sa): MultiHeadAttention(
        (heads): ModuleList(
          (0-1): 2 x SelfAttentionHead(
            (key): Linear(in_features=32, out_features=16, bias=False)
            (query): Linear(in_features=32, out_features=16, bias=False)
            (value): Linear(in_features=32, out_features=16, bias=False)
          )
        )
        (proj): Linear(in_features=32, out_features=32, bias=True)
      )
      (ffwd): FeedForward(
        (net): Sequential(
          (0): Linear(in_features=32, out_features=128, bias=True)
          (1): ReLU()
          (2): Linear(in_features=128, out_features=32, bias=True)
        )
      )
      (ln1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
      (ln2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
    )
    (1): Block(
      (sa): MultiHeadAttention(
        (heads): ModuleList(
          (0-1): 2 x SelfAttentionHead(
            (key): Linear(in_features=32, out_features=16, bias=False)
            (query): Linear(in_features=32, out_features=16, bias=False)
            (value): Linear(in_features=32, out_features=16, bias=False)
          )
        )
        (proj): Linear(in_features=32, out_features=32, bias=True)
      )
      (ffwd): FeedForward(
        (net): Sequential(
          (0): Linear(in_features=32, out_features=128, bias=True)
          (1): ReLU()
          (2): Linear(in_features=128, out_features=32, bias=True)
        )
      )
      (ln1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
      (ln2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
    )
  )
  (ln_f): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
  (head): Linear(in_features=32, out_features=35, bias=True)
)

```


## Part 10: Training the Model 🏋️

Now we train our GPT model! The training loop:
1. Get a batch of (input, target) pairs
2. Forward pass: compute predictions and loss
3. Backward pass: compute gradients
4. Optimizer step: update model weights

We're using the **AdamW optimizer**, which is the standard for transformers.


```python
# Create optimizer
# AdamW is Adam with proper weight decay (L2 regularization)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Print training configuration
print("Starting training...")
print(f"Epochs: {epochs}")
print(f"Learning rate: {lr}")
print("-" * 40)

# Training loop
for step in range(epochs):
    # Get a batch of training data
    xb, yb = get_batch()
    
    # Forward pass: compute predictions and loss
    logits, loss = model(xb, yb)
    
    # Zero out gradients from previous step
    # (PyTorch accumulates gradients by default)
    optimizer.zero_grad()
    
    # Backward pass: compute gradients
    loss.backward()
    
    # Optimizer step: update model weights
    optimizer.step()
    
    # Print progress every 300 steps
    if step % 300 == 0:
        print(f"Step {step:4d} | Loss: {loss.item():.4f}")

# Training complete
print("-" * 40)
print(f"Training complete! Final loss: {loss.item():.4f}")
```


**Output:**
```
Starting training...
Epochs: 1500
Learning rate: 0.001
----------------------------------------
Step    0 | Loss: 3.7184
Step  300 | Loss: 0.3292
Step  600 | Loss: 0.2192
Step  900 | Loss: 0.2919
Step 1200 | Loss: 0.2376
----------------------------------------
Training complete! Final loss: 0.2620

```


## Part 11: Generating Text! 🎨

Now the fun part - let's make our model generate text!

We'll give it a starting word, and it will predict the next words one at a time.


```python
# Set model to evaluation mode
# (disables dropout and other training-specific behaviors)
model.eval()

# Start with a single word
start_word = "mary"
print(f"Starting word: '{start_word}'")
print("\nGenerating text...\n")

# Convert starting word to tensor
context = torch.tensor([[word2idx[start_word]]], dtype=torch.long)

# Generate new tokens
with torch.no_grad():  # No need to track gradients during generation
    generated = model.generate(context, max_new_tokens=15)

# Convert indices back to words
generated_text = " ".join([idx2word[int(i)] for i in generated[0]])

# Print the generated text
print("=" * 50)
print("Generated Text:")
print("=" * 50)
print(generated_text)
print("=" * 50)
```


**Output:**
```
Starting word: 'mary'

Generating text...

==================================================
Generated Text:
==================================================
mary went <END> mary went mary went <END> everywhere that mary went <END> the lamb was
==================================================

```


```python
# Let's try with different starting words!
def generate_from_word(start_word, max_tokens=12):
    """Generate text starting from a given word."""
    if start_word not in word2idx:
        print(f"'{start_word}' not in vocabulary!")
        print(f"Available words: {list(word2idx.keys())}")
        return
    
    # Create context tensor
    context = torch.tensor([[word2idx[start_word]]], dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_tokens)
    
    # Decode generated indices to words
    text = " ".join([idx2word[int(i)] for i in generated[0]])
    return text

# Try different starting words from the nursery rhyme
start_words = ["mary", "the", "it", "little", "lamb"]

# Generate and print results
print("Generating from different starting words:\n")
for word in start_words:
    result = generate_from_word(word)
    if result:
        print(f"'{word}' → {result}")
    print()
```


**Output:**
```
Generating from different starting words:

'mary' → mary went mary went <END> everywhere that mary went <END> the lamb was

'the' → the lamb was sure to go <END> it followed her to school one

'it' → it made the children laugh and play <END> it made the children laugh

'little' → little lamb <END> its fleece was white as snow <END> and everywhere that

'lamb' → lamb <END> little lamb little lamb <END> mary had a little lamb <END>


```


## Part 12: Understanding What We Built 📚

### Summary

Congratulations! You've built a working GPT model from scratch! Here's what each component does:

| Component | Purpose |
|-----------|--------|
| **Token Embedding** | Converts word IDs to dense vectors that capture word meaning |
| **Position Embedding** | Adds positional information so the model knows word order |
| **Self-Attention** | Allows tokens to "look at" other tokens and gather information |
| **Multi-Head Attention** | Runs multiple attention heads to capture different patterns |
| **Feed-Forward Network** | Processes each token independently with non-linearity |
| **Residual Connections** | Helps gradient flow and training stability |
| **Layer Normalization** | Normalizes activations for stable training |
| **Language Model Head** | Converts final representations to word predictions |

### Key Differences from Production GPT

Our TinyGPT is small for educational purposes. Real GPT models have:
- **More layers**: GPT-3 has 96 layers, we have 2
- **Larger embeddings**: GPT-3 uses 12,288 dimensions, we use 32
- **More attention heads**: GPT-3 has 96 heads, we have 2
- **Bigger vocabulary**: GPT-3 has ~50,000 tokens, we have ~40 words
- **More training data**: GPT-3 trained on hundreds of billions of words!

### Next Steps

To improve this model, you could:
1. Use a larger corpus (download books, Wikipedia, etc.)
2. Increase model size (more layers, larger embeddings)
3. Use subword tokenization (BPE) instead of word-level
4. Add dropout for regularization
5. Train on a GPU with larger batch sizes
6. Implement learning rate scheduling


```python
# Final model statistics
print("=" * 50)
print("Model Statistics")
print("=" * 50)
print(f"Vocabulary size: {vocab_size} words")
print(f"Context window: {block_size} tokens")
print(f"Embedding dimension: {embedding_dim}")
print(f"Attention heads: {n_heads}")
print(f"Transformer layers: {n_layers}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print("=" * 50)
print("\n🎉 You've built a GPT from scratch! 🎉")
```


**Output:**
```
==================================================
Model Statistics
==================================================
Vocabulary size: 35 words
Context window: 6 tokens
Embedding dimension: 32
Attention heads: 2
Transformer layers: 2
Total parameters: 27,747
==================================================

🎉 You've built a GPT from scratch! 🎉

```

