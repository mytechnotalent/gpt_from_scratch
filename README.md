# GPT From Scratch

## What is GPT?

**GPT** stands for **Generative Pre-trained Transformer**. It's the technology behind ChatGPT and many other AI chatbots. But what does that mean?

- **Generative**: It can *generate* (create) new text, like writing a story or answering a question
- **Pre-trained**: It learned from reading billions of words before you use it
- **Transformer**: The type of neural network architecture it uses (we'll build this!)

## What Will You Learn?

In this notebook, we'll build a **tiny GPT model from scratch**! By the end, you'll understand:

1. **Tokenization**: How computers turn words into numbers
2. **Embeddings**: How those numbers become meaningful representations
3. **Attention**: How the model decides which words are important
4. **Neural Networks**: The basic building blocks of AI
5. **Training**: How the model learns from examples

We'll use a simple nursery rhyme ("Mary Had a Little Lamb") so you can see exactly what's happening at every step!

#### Author: [Kevin Thomas](mailto:ket189@pitt.edu)


## Part 1: Setup and Imports

Before we start building our AI, we need to import some tools. Think of this like getting your supplies ready before a science project!

### What are these libraries?

| Library | What it does | Real-world analogy |
|---------|--------------|-------------------|
| `torch` | The main AI library (PyTorch) | Your toolbox |
| `torch.nn` | Pre-built neural network pieces | LEGO blocks |
| `torch.nn.functional` | Math operations for AI | Calculator |

**PyTorch** was created by Facebook's AI Research lab and is one of the most popular libraries for building AI. It handles all the complicated math for us!


```python
# ============================================================================
# IMPORTING OUR AI TOOLS
# ============================================================================
# These three lines bring in all the tools we need to build our GPT model.
# It's like importing LEGO sets - each one has different pieces we'll use!

# torch: The main PyTorch library - handles all the math with "tensors" (fancy arrays)
import torch

# torch.nn: Pre-built neural network building blocks (like layers)
import torch.nn as nn

# torch.nn.functional: Math functions we'll use (like softmax for probabilities)
import torch.nn.functional as F

# ============================================================================
# CHECKING OUR COMPUTER'S CAPABILITIES
# ============================================================================
# Neural networks need to do LOTS of math. Let's see what our computer can do!

# Print the version of PyTorch we're using
print("=" * 50)
print("CHECKING YOUR COMPUTER'S AI CAPABILITIES")
print("=" * 50)
print(f"\nPyTorch version: {torch.__version__}")

# Check if we have a GPU (Graphics Processing Unit)
# GPUs are MUCH faster at AI math than regular CPUs!
print(f"\nNVIDIA GPU available (CUDA): {torch.cuda.is_available()}")
print(f"Apple GPU available (MPS): {torch.backends.mps.is_available()}")

# Pick the best available device
# Priority: NVIDIA GPU > Apple GPU > CPU
if torch.cuda.is_available():
    # NVIDIA GPU - fastest option!
    device = torch.device("cuda")
    print(f"\n✅ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    # Apple Silicon GPU - fast on Mac!
    device = torch.device("mps")
    print("\n✅ Using Apple Silicon GPU (MPS)")
else:
    # CPU - works on any computer, just slower
    device = torch.device("cpu")
    print("\n✅ Using CPU (this will work, just slower)")

# Show the final device choice
print(f"\n🖥️  Your AI will run on: {device}")
print("=" * 50)
```


**Output:**
```
==================================================
CHECKING YOUR COMPUTER'S AI CAPABILITIES
==================================================

PyTorch version: 2.9.1

NVIDIA GPU available (CUDA): False
Apple GPU available (MPS): True

✅ Using Apple Silicon GPU (MPS)

🖥️  Your AI will run on: mps
==================================================

```


## Part 2: Preparing the Training Data

### 2.1 What is Training Data?

**Training data** is the text that our AI will learn from. Think of it like a textbook for the AI!

**The Big Idea:** The AI will read this text over and over, learning patterns like:
- "mary" is often followed by "had" or "went"
- "little" is often followed by "lamb"
- Sentences often end with certain words

### Real GPT vs Our Tiny GPT

| | Real GPT (like ChatGPT) | Our Tiny GPT |
|---|---|---|
| **Training Data** | Billions of web pages, books, articles | One nursery rhyme |
| **Words Learned** | 50,000+ word pieces | ~25 words |
| **Training Time** | Months on supercomputers | Seconds on your laptop |

### 2.2 Creating Our Corpus

A **corpus** is just a fancy word for "a collection of text." Our corpus is the nursery rhyme, "Mary Had a Little Lamb", however simple enough to understand, but shows all the concepts!


```python
# ============================================================================
# CREATING OUR TRAINING DATA (THE CORPUS)
# ============================================================================
# This is the text our AI will learn from. We're using, "Mary Had a Little Lamb",
# because it's simple, repetitive and familiar.

# Our training corpus where each line is one sentence from the nursery rhyme.
# The AI will learn patterns from these sentences!
corpus = [
    "mary had a little lamb",              # Sentence 1
    "little lamb little lamb",             # Sentence 2
    "mary had a little lamb",              # Sentence 3
    "its fleece was white as snow",        # Sentence 4
    "and everywhere that mary went",       # Sentence 5
    "mary went mary went",                 # Sentence 6
    "everywhere that mary went",           # Sentence 7
    "the lamb was sure to go",             # Sentence 8
    "it followed her to school one day",   # Sentence 9
    "school one day school one day",       # Sentence 10
    "it followed her to school one day",   # Sentence 11
    "which was against the rules",         # Sentence 12
    "it made the children laugh and play", # Sentence 13
    "laugh and play laugh and play",       # Sentence 14
    "it made the children laugh and play", # Sentence 15
    "to see a lamb at school"              # Sentence 16
]

# ============================================================================
# ADDING A SPECIAL "END" TOKEN
# ============================================================================
# We add "<END>" to mark where each sentence stops.
# This teaches the AI to know when a thought is complete!
# Without this, it might just ramble forever.

# Add " <END>" to the end of every sentence
corpus = [sentence + " <END>" for sentence in corpus]

# Let's see what one sentence looks like now
print("Example sentence with END token:")
print(f"  '{corpus[0]}'")
print()

# ============================================================================
# COMBINING ALL SENTENCES INTO ONE BIG TEXT
# ============================================================================
# We join all sentences together with spaces between them.
# This creates one long string of text for training.

# Join all sentences with a space between each one
text = " ".join(corpus)

# Show the complete training text
print("=" * 60)
print("OUR COMPLETE TRAINING TEXT:")
print("=" * 60)
print(text)
print()
print(f"📊 Total characters: {len(text)}")
print(f"📊 Total words: {len(text.split())}")
```


**Output:**
```
Example sentence with END token:
  'mary had a little lamb <END>'

============================================================
OUR COMPLETE TRAINING TEXT:
============================================================
mary had a little lamb <END> little lamb little lamb <END> mary had a little lamb <END> its fleece was white as snow <END> and everywhere that mary went <END> mary went mary went <END> everywhere that mary went <END> the lamb was sure to go <END> it followed her to school one day <END> school one day school one day <END> it followed her to school one day <END> which was against the rules <END> it made the children laugh and play <END> laugh and play laugh and play <END> it made the children laugh and play <END> to see a lamb at school <END>

📊 Total characters: 546
📊 Total words: 106

```


### 2.2 Building the Vocabulary (Teaching the AI Words)

Before our AI can read text, we need to convert words into numbers. This is called **tokenization**.

**Why numbers?** Computers can't understand words directly as they only understand numbers! So we need to give each word a unique number (like an ID or index).

**Here's our 3-step plan:**
1. **Find all unique words** in our text (our "vocabulary")
2. **Assign each word a unique number** (like giving each word an ID badge)
3. **Create dictionaries** so we can easily convert between words ↔ numbers

**Real-world example:**
```
Word:   "mary" → Number: 0
Word:   "had"  → Number: 1  
Word:   "a"    → Number: 2
...and so on
```


```python
# ============================================================================
# STEP 1: FIND ALL UNIQUE WORDS IN OUR TEXT
# ============================================================================
# We use set() to automatically remove duplicate words.
# For example, "mary" appears many times but set() keeps only ONE copy.
# Then we convert back to a list so we can work with it.

# Split text into words, then remove duplicates using set()
words = list(set(text.split()))

# Let's see what unique words we found!
print("=" * 60)
print("ALL UNIQUE WORDS IN OUR TEXT:")
print("=" * 60)
print(words)
print()

# Count how many unique words we have - this is our "vocabulary size"
vocab_size = len(words)
print(f"📊 We found {vocab_size} unique words!")
print(f"   This means our vocabulary size = {vocab_size}")
```


**Output:**
```
============================================================
ALL UNIQUE WORDS IN OUR TEXT:
============================================================
['one', 'against', 'that', 'was', 'followed', 'day', '<END>', 'everywhere', 'its', 'white', 'which', 'snow', 'as', 'a', 'it', 'the', 'play', 'to', 'made', 'little', 'school', 'laugh', 'went', 'her', 'see', 'rules', 'fleece', 'lamb', 'had', 'and', 'at', 'go', 'mary', 'children', 'sure']

📊 We found 35 unique words!
   This means our vocabulary size = 35

```


```python
# ============================================================================
# STEP 2: CREATE WORD ↔ NUMBER DICTIONARIES
# ============================================================================
# We need TWO dictionaries:
#   1. word2idx: looks up a word, gives us its number (for ENCODING)
#   2. idx2word: looks up a number, gives us its word (for DECODING)

# Dictionary 1: Word → Number (for encoding text into numbers)
# Example: word2idx["mary"] might give us 5
word2idx = {word: idx for idx, word in enumerate(words)}

# Display the two dictionaries clearly
print("=" * 60)
print("DICTIONARY 1: word2idx (Word → Number)")
print("=" * 60)
print("Use this to ENCODE words into numbers the AI can process")
print()
for word, idx in word2idx.items():
    print(f"   '{word}' → {idx}")

# Dictionary 2: Number → Word (for decoding numbers back to text)
# Example: idx2word[5] might give us "mary"
idx2word = {idx: word for word, idx in word2idx.items()}

# Display the second dictionary clearly
print()
print("=" * 60)
print("DICTIONARY 2: idx2word (Number → Word)")
print("=" * 60)
print("Use this to DECODE numbers back into readable words")
print()
for idx, word in idx2word.items():
    print(f"   {idx} → '{word}'")
```


**Output:**
```
============================================================
DICTIONARY 1: word2idx (Word → Number)
============================================================
Use this to ENCODE words into numbers the AI can process

   'one' → 0
   'against' → 1
   'that' → 2
   'was' → 3
   'followed' → 4
   'day' → 5
   '<END>' → 6
   'everywhere' → 7
   'its' → 8
   'white' → 9
   'which' → 10
   'snow' → 11
   'as' → 12
   'a' → 13
   'it' → 14
   'the' → 15
   'play' → 16
   'to' → 17
   'made' → 18
   'little' → 19
   'school' → 20
   'laugh' → 21
   'went' → 22
   'her' → 23
   'see' → 24
   'rules' → 25
   'fleece' → 26
   'lamb' → 27
   'had' → 28
   'and' → 29
   'at' → 30
   'go' → 31
   'mary' → 32
   'children' → 33
   'sure' → 34

============================================================
DICTIONARY 2: idx2word (Number → Word)
============================================================
Use this to DECODE numbers back into readable words

   0 → 'one'
   1 → 'against'
   2 → 'that'
   3 → 'was'
   4 → 'followed'
   5 → 'day'
   6 → '<END>'
   7 → 'everywhere'
   8 → 'its'
   9 → 'white'
   10 → 'which'
   11 → 'snow'
   12 → 'as'
   13 → 'a'
   14 → 'it'
   15 → 'the'
   16 → 'play'
   17 → 'to'
   18 → 'made'
   19 → 'little'
   20 → 'school'
   21 → 'laugh'
   22 → 'went'
   23 → 'her'
   24 → 'see'
   25 → 'rules'
   26 → 'fleece'
   27 → 'lamb'
   28 → 'had'
   29 → 'and'
   30 → 'at'
   31 → 'go'
   32 → 'mary'
   33 → 'children'
   34 → 'sure'

```


```python
# ============================================================================
# STEP 3: CONVERT OUR ENTIRE TEXT INTO NUMBERS
# ============================================================================
# Now we use word2idx to convert every word in our text to its number.
# This creates a "data tensor" - a list of numbers representing our text.

# Convert each word to its index number
data = torch.tensor([word2idx[word] for word in text.split()], dtype=torch.long)

# Let's see what we created!
print("=" * 60)
print("OUR TEXT CONVERTED TO NUMBERS (DATA TENSOR)")
print("=" * 60)
print(f"Shape: {data.shape} (meaning we have {len(data)} tokens)")
print()

# Show the first 20 tokens as numbers
print("First 20 tokens (as numbers):")
print(f"   {data[:20].tolist()}")
print()

# Decode those numbers back to words to verify it worked!
decoded_words = [idx2word[int(i)] for i in data[:20]]
print("Those same 20 tokens decoded back to words:")
print(f"   {' '.join(decoded_words)}")
print()
print("✅ Perfect! We can now convert text → numbers → text!")
```


**Output:**
```
============================================================
OUR TEXT CONVERTED TO NUMBERS (DATA TENSOR)
============================================================
Shape: torch.Size([106]) (meaning we have 106 tokens)

First 20 tokens (as numbers):
   [32, 28, 13, 19, 27, 6, 19, 27, 19, 27, 6, 32, 28, 13, 19, 27, 6, 8, 26, 3]

Those same 20 tokens decoded back to words:
   mary had a little lamb <END> little lamb little lamb <END> mary had a little lamb <END> its fleece was

✅ Perfect! We can now convert text → numbers → text!

```


## Part 3: Hyperparameters (The Settings That Control Our AI)

**Hyperparameters** are like the settings on a video game as they control how the game (or in this case, our AI) behaves. These are values WE choose before training starts.

Think of building a brain:
- How many brain cells should we use?
- How should they connect?
- How fast should it learn?

Here are our key settings:

| Parameter | What It Means | Our Value | Simple Explanation |
|-----------|---------------|-----------|---------------------|
| `block_size` | Context window | 6 | AI can "remember" 6 words at a time |
| `embedding_dim` | Word vector size | 32 | Each word becomes 32 numbers |
| `n_heads` | Attention heads | 2 | AI looks at text 2 different ways |
| `n_layers` | Transformer layers | 2 | Stack 2 processing blocks together |
| `lr` | Learning rate | 0.001 | How big each learning step is |
| `epochs` | Training rounds | 1500 | Train for 1500 rounds |

**Analogy:** If the AI is a student:
- `block_size` = How many words they can read at once
- `embedding_dim` = How detailed their understanding is
- `n_heads` = How many different perspectives they consider
- `n_layers` = How many times they re-read and think
- `lr` = How quickly they adjust their understanding
- `epochs` = How many times they practice


```python
# ============================================================================
# MODEL ARCHITECTURE HYPERPARAMETERS
# ============================================================================
# These control the SIZE and STRUCTURE of our neural network

# Context window: how many tokens the model can "see" at once
# Like reading 6 words at a time through a small window
block_size = 6

# Embedding dimension: how many numbers represent each word
# Bigger = more detailed representation, but slower training
embedding_dim = 32

# Number of attention heads: parallel ways to analyze relationships
# More heads = more perspectives on the data
n_heads = 2

# Number of transformer layers: stacked processing blocks
# More layers = deeper understanding, but more computation
n_layers = 2

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
# These control HOW the model learns

# Learning rate: how big each learning step is (0.001)
# Too high = overshoots and never learns
# Too low = takes forever to learn
lr = 1e-3

# Epochs: how many times we go through training
# More epochs = more practice = better learning (up to a point)
epochs = 1500

# ============================================================================
# DISPLAY ALL OUR SETTINGS
# ============================================================================
# Let's summarize all our hyperparameters clearly
print("=" * 60)
print("HYPERPARAMETERS CONFIGURED!")
print("=" * 60)
print()
print("📐 MODEL ARCHITECTURE:")
print(f"   • Context window (block_size): {block_size} tokens")
print(f"   • Embedding dimension: {embedding_dim} numbers per word")
print(f"   • Attention heads: {n_heads}")
print(f"   • Transformer layers: {n_layers}")
print()
print("🎓 TRAINING SETTINGS:")
print(f"   • Learning rate: {lr}")
print(f"   • Training epochs: {epochs}")
print()
print("✅ Ready to build our model!")
```


**Output:**
```
============================================================
HYPERPARAMETERS CONFIGURED!
============================================================

📐 MODEL ARCHITECTURE:
   • Context window (block_size): 6 tokens
   • Embedding dimension: 32 numbers per word
   • Attention heads: 2
   • Transformer layers: 2

🎓 TRAINING SETTINGS:
   • Learning rate: 0.001
   • Training epochs: 1500

✅ Ready to build our model!

```


### 3.1 Understanding the Embedding Matrix (The Word Dictionary)

**What is an embedding matrix?** It's a big table that converts words into lists of numbers. Think of it like a secret code book!

**Imagine this:** Every word in our vocabulary gets its own "fingerprint" made of 32 numbers. These numbers describe the word in a way the computer can understand.

**Visual Example - Our Embedding Matrix:**

```
                    32 Columns (one for each "feature")
                    ↓   ↓   ↓   ↓   ↓   ↓       ↓
              ┌────────────────────────────────────┐
Token 0 (mary) →│ 0.23 -0.45 0.12 0.56 ... 0.67 │  ← "mary"'s fingerprint
Token 1 (had)  →│-0.11  0.89-0.34 0.21 ... 0.21 │  ← "had"'s fingerprint
Token 2 (a)    →│ 0.56  0.02 0.78-0.33 ...-0.45 │  ← "a"'s fingerprint
Token 3 (lamb) →│-0.22  0.33 0.44 0.11 ... 0.88 │  ← "lamb"'s fingerprint
      ...      │  ...   ...  ...  ... ...  ... │
Token 26(<END>)→│-0.33  0.44 0.11 0.67 ... 0.55 │  ← "<END>"'s fingerprint
              └────────────────────────────────────┘
                 ↑
                 27 Rows (one for each word in vocabulary)
```

**Key Points:**
- **Rows** = `vocab_size` (27 words) → Each row is ONE word's number-fingerprint
- **Columns** = `embedding_dim` (32 numbers) → Each column is a "feature" dimension

**The magic:** At first, these numbers are RANDOM! But during training, the AI adjusts them so:
- Similar words (like "lamb" and "fleece") end up with similar number patterns
- Different words (like "school" and "snow") have different patterns

**Analogy:** Imagine each word is a person, and the 32 numbers describe them:
- Number 1 might relate to "is it an animal?"
- Number 2 might relate to "is it a verb?"
- Number 3 might relate to "is it happy?"
- ...and so on (though we don't actually know what each number means!)


```python
# ============================================================================
# LET'S SEE THE EMBEDDING MATRIX IN ACTION!
# ============================================================================
# We'll create an embedding layer and look at what it contains.

# Create an embedding layer: vocab_size rows × embedding_dim columns
sample_embedding = nn.Embedding(vocab_size, embedding_dim)

# Show the shape of our embedding matrix
print("=" * 60)
print("OUR EMBEDDING MATRIX")
print("=" * 60)
print(f"Shape: {sample_embedding.weight.shape}")
print(f"  • Rows: {vocab_size} (one row per word in vocabulary)")
print(f"  • Columns: {embedding_dim} (each word becomes 32 numbers)")
print()
print(f"📊 Total numbers to learn: {vocab_size} × {embedding_dim} = {vocab_size * embedding_dim:,}")
print()

# ============================================================================
# LOOK UP SPECIFIC WORDS IN THE EMBEDDING MATRIX
# ============================================================================
# Let's see what "fingerprint" each word has!

# Token 0's word
token_0_word = idx2word[0]
print("=" * 60)
print(f"WORD AT ROW 0: '{token_0_word}'")
print("=" * 60)
print("Its 32-number embedding (currently random):")
print(sample_embedding.weight[0].data)
print()

# Token 5's word
token_5_word = idx2word[5]
print("=" * 60)
print(f"WORD AT ROW 5: '{token_5_word}'")
print("=" * 60)
print("Its 32-number embedding (currently random):")
print(sample_embedding.weight[5].data)
print()

# The embeddings are random at first
print("💡 These numbers are random NOW, but during training,")
print("   they'll adjust so similar words have similar numbers!")
```


**Output:**
```
============================================================
OUR EMBEDDING MATRIX
============================================================
Shape: torch.Size([35, 32])
  • Rows: 35 (one row per word in vocabulary)
  • Columns: 32 (each word becomes 32 numbers)

📊 Total numbers to learn: 35 × 32 = 1,120

============================================================
WORD AT ROW 0: 'one'
============================================================
Its 32-number embedding (currently random):
tensor([ 0.6332, -0.0753, -1.8653,  0.8703,  1.1888, -2.7935, -0.9989,  0.4554,
        -1.0302,  0.5497,  0.8914, -0.7058,  1.4873, -1.7920,  0.2586, -0.1602,
        -0.9965,  0.1511,  1.9502,  1.3443,  0.2510, -0.1353,  1.7262,  0.5959,
        -0.7752, -1.4495, -0.5191, -0.2409, -0.7671,  2.0413,  1.0716, -1.1712])

============================================================
WORD AT ROW 5: 'day'
============================================================
Its 32-number embedding (currently random):
tensor([-0.3783,  0.7111, -1.7393,  0.5526, -0.3928,  1.4691,  0.0308,  0.0430,
         0.1392,  2.3944,  0.9666, -0.2336,  1.2540,  0.1864,  0.1668,  0.0821,
         0.0717,  0.3131, -1.2393, -0.2884,  0.0065,  0.7896, -0.5960,  0.4357,
         1.7102,  0.5282, -0.1130,  0.6262,  0.3687, -1.7577,  0.0103, -0.9602])

💡 These numbers are random NOW, but during training,
   they'll adjust so similar words have similar numbers!

```


### 3.1.1 Neural Network Fundamentals: A Quick Primer

Before we go further, let's understand the **basic building blocks of neural networks**. This is super important for understanding how GPT learns!

---

## 🧠 What is a Neuron?

A **neuron** is the smallest unit of a brain (real or artificial). In AI, it's a simple math formula.

A single neuron can take **multiple inputs** (like 2, 3, or even 1000!), but it always produces **one output**.

```
        ┌─────────────────────────────────────────────────────────────┐
        │                    A SINGLE NEURON                          │
        │                                                             │
        │   2 INPUTS           WEIGHTS        PROCESSING              │
        │      │                  │               │                   │
        │      ▼                  ▼               ▼                   │
        │                                                             │
        │   ┌───┐                             ┌─────────┐             │
        │   │0.8│───── × 0.6 ──────┐          │         │             │
        │   └───┘                  │          │         │             │
        │   Input 1                ▼          │         │   ┌─────┐   │
        │                       ┌─────┐       │  ReLU   │──►│ 0.66│   │
        │   ┌───┐               │ SUM │──────►│         │   └─────┘   │
        │   │0.2│───── × 0.4 ──►│     │       │         │   Output!   │
        │   └───┘               │     │       │         │             │
        │   Input 2             │ +0.1│←bias  └─────────┘             │
        │                       └─────┘                               │
        │                          ▲                                  │
        │                          │                                  │
        │                   Bias is added                             │
        │                   INSIDE the neuron                         │
        │                   (not a 3rd input!)                        │
        │                                                             │
        └─────────────────────────────────────────────────────────────┘

The math: (0.8 × 0.6) + (0.2 × 0.4) + 0.1 = 0.48 + 0.08 + 0.1 = 0.66
         \_________/   \_________/   \_/
          Input 1       Input 2     Bias
          weighted      weighted    (constant)
```

**Key Point:** The neuron has 2 inputs, but the bias is NOT a third input!
- **Inputs** come from outside (data from other neurons or raw data)
- **Bias** is a constant stored INSIDE the neuron (like its personal adjustment)

---

**Parts of a Neuron:**

| Part | What It Is | Analogy |
|------|-----------|---------|
| **Inputs (x)** | Numbers fed into the neuron | Questions on a test |
| **Weights (w)** | How important each input is | How much each question is worth |
| **Bias (b)** | A constant adjustment inside the neuron | Bonus points on the test |
| **Activation** | A special function applied at the end | The grading curve |

**The Formula:**
$$\text{output} = \text{activation}(w_1 \cdot x_1 + w_2 \cdot x_2 + b)$$

**Why the activation function?** Without it, the neuron is just basic multiplication and addition. The activation function adds "curves" that let the network learn complex patterns.


```python
# ============================================================================
# EXAMPLE: Building a Single Neuron from Scratch
# ============================================================================
# Let's predict if someone will like a movie based on two scores!

print("=" * 60)
print("EXAMPLE: A Single Neuron Predicting Movie Enjoyment")
print("=" * 60)
print()

# --- STEP 1: Define our inputs ---

# These are the two features about the movie (scale 0-1)
x1 = 0.8  # Action score: 0.8 means 80% action-packed
x2 = 0.2  # Romance score: 0.2 means only 20% romance

print("INPUTS (what we know about the movie):")
print(f"  x₁ = {x1} (action score - high means lots of action)")
print(f"  x₂ = {x2} (romance score - low means little romance)")
print()

# --- STEP 2: Define our weights (learnable!) ---

# These represent what the viewer likes (positive = likes, negative = dislikes)
w1 = 0.6   # Weight for action: positive = they like action
w2 = -0.4  # Weight for romance: NEGATIVE = they dislike romance!

print("WEIGHTS (how important each feature is):")
print(f"  w₁ = {w1} (positive: this person LIKES action)")
print(f"  w₂ = {w2} (negative: this person DISLIKES romance)")
print()

# --- STEP 3: Define the bias (starting point) ---

# This is the starting tendency before considering inputs
b = 0.1  # They generally like movies (+0.1 bonus)

print("BIAS (starting tendency):")
print(f"  b = {b} (slightly positive: they generally like movies)")
print()

# --- STEP 4: Calculate the weighted sum ---

# Formula: w₁·x₁ + w₂·x₂ + b
weighted_sum = (w1 * x1) + (w2 * x2) + b

print("=" * 60)
print("THE MATH: Weighted Sum")
print("=" * 60)
print()
print("Formula: weighted_sum = w₁·x₁ + w₂·x₂ + b")
print()
print(f"  = ({w1}) × ({x1}) + ({w2}) × ({x2}) + {b}")
print(f"  = {w1*x1} + {w2*x2} + {b}")
print(f"  = {weighted_sum}")
print()

# --- STEP 5: Apply activation function (ReLU) ---

# ReLU = "Rectified Linear Unit" = max(0, x)
# If negative, make it 0. Otherwise, keep it.

def relu(x):
    """ReLU activation: returns max(0, x)."""
    return max(0, x)

# Apply ReLU to the weighted sum
output = relu(weighted_sum)

print("=" * 60)
print("THE MATH: Activation Function (ReLU)")
print("=" * 60)
print()
print("ReLU means: if negative → 0, if positive → keep it")
print(f"  ReLU({weighted_sum}) = max(0, {weighted_sum}) = {output}")
print()

# --- FINAL RESULT ---

print("=" * 60)
print(f"🎯 FINAL OUTPUT: {output:.2f}")
print("=" * 60)
print()
if output > 0.4:
    print("Interpretation: They'll probably LOVE this movie! 🎬❤️")
elif output > 0.2:
    print("Interpretation: They'll probably enjoy this movie! 🎬👍")
else:
    print("Interpretation: They might not enjoy this movie 🎬😐")
```


**Output:**
```
============================================================
EXAMPLE: A Single Neuron Predicting Movie Enjoyment
============================================================

INPUTS (what we know about the movie):
  x₁ = 0.8 (action score - high means lots of action)
  x₂ = 0.2 (romance score - low means little romance)

WEIGHTS (how important each feature is):
  w₁ = 0.6 (positive: this person LIKES action)
  w₂ = -0.4 (negative: this person DISLIKES romance)

BIAS (starting tendency):
  b = 0.1 (slightly positive: they generally like movies)

============================================================
THE MATH: Weighted Sum
============================================================

Formula: weighted_sum = w₁·x₁ + w₂·x₂ + b

  = (0.6) × (0.8) + (-0.4) × (0.2) + 0.1
  = 0.48 + -0.08000000000000002 + 0.1
  = 0.5

============================================================
THE MATH: Activation Function (ReLU)
============================================================

ReLU means: if negative → 0, if positive → keep it
  ReLU(0.5) = max(0, 0.5) = 0.5

============================================================
🎯 FINAL OUTPUT: 0.50
============================================================

Interpretation: They'll probably LOVE this movie! 🎬❤️

```


### 3.1.2 A Simple Neural Network (Connecting Neurons Together)

One neuron isn't very smart. But when we **connect many neurons together**, they can learn amazing things!

Here's a tiny neural network with **3 layers**:

```
                    SIMPLE NEURAL NETWORK
    ┌──────────────────────────────────────────────────────────────┐
    │                                                              │
    │   INPUT LAYER      HIDDEN LAYER         OUTPUT LAYER         │
    │   (what we         (does the            (gives us            │
    │    know)           thinking)            the answer)          │
    │                                                              │
    │       x₁ ─────────►[h₁]─────────────┐                        │
    │        │ ╲        ╱    ╲             ╲                       │
    │        │   ╲    ╱        ╲            ╲                      │
    │        │     ╳            ╲            ►[out]────► ŷ         │
    │        │   ╱    ╲          ╲          ╱  (answer)            │
    │        │ ╱        ╲         ╲        ╱                       │
    │       x₂ ─────────►[h₂]──────────────┘                       │
    │                                                              │
    │   (2 values)     (2 neurons)        (1 neuron)               │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
```

**What's happening:**
1. **Input Layer**: We feed in 2 numbers (like action score and romance score)
2. **Hidden Layer**: 2 neurons process those numbers (the "thinking" happens here)
3. **Output Layer**: 1 neuron combines everything into a final answer

**Each arrow = a WEIGHT (a number the AI can change)**

**Counting the learnable parameters:**
| Connection | Calculation | Result |
|-----------|-------------|--------|
| Inputs → Hidden | 2 inputs × 2 neurons = 4 weights | 4 weights |
| Hidden biases | 2 neurons × 1 bias each | 2 biases |
| Hidden → Output | 2 neurons × 1 output = 2 weights | 2 weights |
| Output bias | 1 neuron × 1 bias | 1 bias |
| **TOTAL** | | **9 learnable parameters** |

**Why "hidden"?** We can see the inputs and outputs, but the hidden layer is like the brain's inner workings - hidden from view!


```python
# ============================================================================
# BUILDING A SIMPLE NEURAL NETWORK FROM SCRATCH
# ============================================================================
# Let's build the 3-layer network from the diagram above!

print("=" * 70)
print("BUILDING A SIMPLE NEURAL NETWORK FROM SCRATCH")
print("=" * 70)
print()

# --- STEP 1: Define our inputs ---

# Same movie prediction example: action score and romance score
x1 = 0.8  # Action score (80% action)
x2 = 0.2  # Romance score (20% romance)

print("📥 STEP 1: OUR INPUTS")
print(f"   x₁ = {x1} (action score)")
print(f"   x₂ = {x2} (romance score)")
print()

# --- STEP 2: Define hidden layer weights and biases ---

# Weights going TO hidden neuron 1 (h₁)
w_h1_x1 = 0.5   # How much x₁ affects h₁
w_h1_x2 = 0.3   # How much x₂ affects h₁
b_h1 = 0.1      # Bias for h₁

# Weights going TO hidden neuron 2 (h₂)
w_h2_x1 = -0.2  # How much x₁ affects h₂ (negative = opposite effect!)
w_h2_x2 = 0.8   # How much x₂ affects h₂
b_h2 = -0.1     # Bias for h₂

print("🔷 STEP 2: HIDDEN LAYER PARAMETERS (6 total)")
print(f"   Neuron h₁: w₁={w_h1_x1}, w₂={w_h1_x2}, bias={b_h1}")
print(f"   Neuron h₂: w₁={w_h2_x1}, w₂={w_h2_x2}, bias={b_h2}")
print()

# --- STEP 3: Define output layer weights and biases ---

# Weights going TO output neuron
w_out_h1 = 0.7   # How much h₁ affects the output
w_out_h2 = 0.4   # How much h₂ affects the output
b_out = 0.05     # Bias for the output

print("🔷 STEP 3: OUTPUT LAYER PARAMETERS (3 total)")
print(f"   Output neuron: w_h1={w_out_h1}, w_h2={w_out_h2}, bias={b_out}")
print()
print(f"📊 TOTAL LEARNABLE PARAMETERS: 6 + 3 = 9")
print()

# --- STEP 4: Forward pass - hidden layer ---

print("=" * 70)
print("FORWARD PASS: Computing the Output")
print("=" * 70)
print()

# Compute hidden neuron 1's value
z_h1 = (w_h1_x1 * x1) + (w_h1_x2 * x2) + b_h1

print("🧮 HIDDEN NEURON 1 (h₁):")
print(f"   Before ReLU (z_h1):")
print(f"   = (w₁ × x₁) + (w₂ × x₂) + bias")
print(f"   = ({w_h1_x1} × {x1}) + ({w_h1_x2} × {x2}) + {b_h1}")
print(f"   = {w_h1_x1*x1} + {w_h1_x2*x2} + {b_h1}")
print(f"   = {z_h1}")

# Apply ReLU
h1 = max(0, z_h1)
print(f"   After ReLU: h₁ = max(0, {z_h1}) = {h1}")
print()

# Compute hidden neuron 2's value
z_h2 = (w_h2_x1 * x1) + (w_h2_x2 * x2) + b_h2

print("🧮 HIDDEN NEURON 2 (h₂):")
print(f"   Before ReLU (z_h2):")
print(f"   = (w₁ × x₁) + (w₂ × x₂) + bias")
print(f"   = ({w_h2_x1} × {x1}) + ({w_h2_x2} × {x2}) + {b_h2}")
print(f"   = {w_h2_x1*x1} + {w_h2_x2*x2} + {b_h2}")
print(f"   = {z_h2}")

# Apply ReLU
h2 = max(0, z_h2)
print(f"   After ReLU: h₂ = max(0, {z_h2}) = {h2}")
print()

# --- STEP 5: Forward pass - output layer ---

# Compute output neuron's value
z_out = (w_out_h1 * h1) + (w_out_h2 * h2) + b_out

print("🧮 OUTPUT NEURON:")
print(f"   Before ReLU (z_out):")
print(f"   = (w_h1 × h₁) + (w_h2 × h₂) + bias")
print(f"   = ({w_out_h1} × {h1}) + ({w_out_h2} × {h2}) + {b_out}")
print(f"   = {w_out_h1*h1:.4f} + {w_out_h2*h2:.4f} + {b_out}")
print(f"   = {z_out:.4f}")

# Apply ReLU to get final output
output = max(0, z_out)
print(f"   After ReLU: output = max(0, {z_out:.4f}) = {output:.4f}")
print()

# --- FINAL RESULT ---

print("=" * 70)
print(f"🎯 FINAL PREDICTION: {output:.4f}")
print("=" * 70)
print()
print("💡 This number is the network's answer!")
print("   In the next section, we'll see how to measure if it's right or wrong.")
```


**Output:**
```
======================================================================
BUILDING A SIMPLE NEURAL NETWORK FROM SCRATCH
======================================================================

📥 STEP 1: OUR INPUTS
   x₁ = 0.8 (action score)
   x₂ = 0.2 (romance score)

🔷 STEP 2: HIDDEN LAYER PARAMETERS (6 total)
   Neuron h₁: w₁=0.5, w₂=0.3, bias=0.1
   Neuron h₂: w₁=-0.2, w₂=0.8, bias=-0.1

🔷 STEP 3: OUTPUT LAYER PARAMETERS (3 total)
   Output neuron: w_h1=0.7, w_h2=0.4, bias=0.05

📊 TOTAL LEARNABLE PARAMETERS: 6 + 3 = 9

======================================================================
FORWARD PASS: Computing the Output
======================================================================

🧮 HIDDEN NEURON 1 (h₁):
   Before ReLU (z_h1):
   = (w₁ × x₁) + (w₂ × x₂) + bias
   = (0.5 × 0.8) + (0.3 × 0.2) + 0.1
   = 0.4 + 0.06 + 0.1
   = 0.56
   After ReLU: h₁ = max(0, 0.56) = 0.56

🧮 HIDDEN NEURON 2 (h₂):
   Before ReLU (z_h2):
   = (w₁ × x₁) + (w₂ × x₂) + bias
   = (-0.2 × 0.8) + (0.8 × 0.2) + -0.1
   = -0.16000000000000003 + 0.16000000000000003 + -0.1
   = -0.1
   After ReLU: h₂ = max(0, -0.1) = 0

🧮 OUTPUT NEURON:
   Before ReLU (z_out):
   = (w_h1 × h₁) + (w_h2 × h₂) + bias
   = (0.7 × 0.56) + (0.4 × 0) + 0.05
   = 0.3920 + 0.0000 + 0.05
   = 0.4420
   After ReLU: output = max(0, 0.4420) = 0.4420

======================================================================
🎯 FINAL PREDICTION: 0.4420
======================================================================

💡 This number is the network's answer!
   In the next section, we'll see how to measure if it's right or wrong.

```


### 3.1.3 The Loss Function: Measuring How Wrong We Are

Our network made a prediction. But how do we know if it's **good or bad**?

We use a **loss function** - a formula that measures the **difference between our prediction and the correct answer**.

---

## 📏 Think of it Like a Test Score

| What We Want | What We Got | How Wrong? |
|-------------|-------------|------------|
| 100 points | 100 points | 0 (perfect!) |
| 100 points | 90 points | 10 (pretty good) |
| 100 points | 50 points | 50 (needs work) |
| 100 points | 0 points | 100 (very wrong) |

**The loss is like the "wrongness score"** - we want it to be as LOW as possible!

---

## 🧮 Mean Squared Error (MSE)

The most common loss function is **Mean Squared Error**:

$$\text{Loss} = (\text{true value} - \text{prediction})^2$$

**Why square it?**
1. Makes all errors positive (no negative wrongness!)
2. Punishes big errors more than small ones (being off by 10 is WAY worse than being off by 2)

**Example:**
```
True value (what we wanted):     y = 1.0
Our prediction:                  ŷ = 0.6
                                    
Error:                           1.0 - 0.6 = 0.4
Squared Error (Loss):            0.4² = 0.16
```

**Our goal during training:** Adjust the weights to make this loss number SMALLER!


```python
# ============================================================================
# COMPUTING THE LOSS: How Wrong Is Our Network?
# ============================================================================
# Let's measure how far off our prediction is from what we wanted.

print("=" * 70)
print("COMPUTING THE LOSS (How Wrong Are We?)")
print("=" * 70)
print()

# --- The prediction and the truth ---

# Our network predicted this value (from the forward pass above)
y_pred = 0.5560  # What our network outputted

# This is what we WANTED the network to say (the "ground truth")
y_true = 1.0  # We wanted it to predict "definitely will like the movie"

print("📊 COMPARING PREDICTION TO TRUTH:")
print(f"   What we wanted (y_true): {y_true}")
print(f"   What we got (y_pred):    {y_pred:.4f}")
print(f"   Difference:              {y_true - y_pred:.4f}")
print()

# --- Step-by-step loss calculation ---

# Using Mean Squared Error: Loss = (y_true - y_pred)²

# Step 1: Find the difference (error)
error = y_true - y_pred

# Step 2: Square it (makes it positive and punishes big errors)
loss = error ** 2

print("🧮 LOSS CALCULATION (Mean Squared Error):")
print()
print("   Step 1: Find the difference")
print(f"           error = y_true - y_pred")
print(f"                 = {y_true} - {y_pred:.4f}")
print(f"                 = {error:.4f}")
print()
print("   Step 2: Square it")
print(f"           loss = error²")
print(f"                = ({error:.4f})²")
print(f"                = {loss:.4f}")
print()

# --- What does this loss mean? ---

print("=" * 70)
print(f"📉 FINAL LOSS: {loss:.4f}")
print("=" * 70)
print()
print("💡 What this means:")
print(f"   • Our prediction was off by {abs(error):.4f}")
print(f"   • The squared error (loss) is {loss:.4f}")
print()
print("🎯 OUR GOAL: Adjust the weights to make this loss SMALLER!")
print("   → Smaller loss = better predictions")
print("   → The next section shows HOW to adjust the weights!")
```


**Output:**
```
======================================================================
COMPUTING THE LOSS (How Wrong Are We?)
======================================================================

📊 COMPARING PREDICTION TO TRUTH:
   What we wanted (y_true): 1.0
   What we got (y_pred):    0.5560
   Difference:              0.4440

🧮 LOSS CALCULATION (Mean Squared Error):

   Step 1: Find the difference
           error = y_true - y_pred
                 = 1.0 - 0.5560
                 = 0.4440

   Step 2: Square it
           loss = error²
                = (0.4440)²
                = 0.1971

======================================================================
📉 FINAL LOSS: 0.1971
======================================================================

💡 What this means:
   • Our prediction was off by 0.4440
   • The squared error (loss) is 0.1971

🎯 OUR GOAL: Adjust the weights to make this loss SMALLER!
   → Smaller loss = better predictions
   → The next section shows HOW to adjust the weights!

```


### 3.1.4 Backpropagation: How the Network Learns

Now the big question: **How do we adjust the weights to make the loss smaller?**

The answer is **backpropagation** (short for "backward propagation of errors").

---

## 🎯 The Key Idea

Imagine you're shooting arrows at a target:
- If you miss to the LEFT, you adjust your aim to the RIGHT
- If you miss to the RIGHT, you adjust your aim to the LEFT
- The AMOUNT you adjust depends on HOW FAR you missed

Neural networks do the same thing with math!

---

## 📐 The Chain Rule (Don't Panic!)

The **chain rule** is a rule from calculus that lets us figure out how much each weight contributed to the error.

**Simple analogy:** If you're baking cookies and they taste bad:
- Was it too much sugar? Or too little flour? Or wrong temperature?
- The chain rule helps us figure out WHICH ingredient to adjust and by HOW MUCH

**The Math (simplified):**
$$\frac{\partial \text{Loss}}{\partial \text{weight}} = \text{how much this weight affected the loss}$$

---

## 🔄 Forward vs Backward Pass

```
FORWARD PASS (computing the output):
─────────────────────────────────────────────────────►
Input → Weights → Hidden → Weights → Output → Loss
                                                 │
                                                 │
BACKWARD PASS (computing gradients):             │
◄────────────────────────────────────────────────┘
∂L/∂w₁ ← ∂L/∂h₁ ← ∂L/∂out ← ∂L/∂ŷ ← Loss

First we go FORWARD to get the prediction.
Then we go BACKWARD to figure out how to fix it!
```

**The gradient (∂L/∂w) tells us:**
- **Sign**: Should we increase or decrease this weight?
- **Magnitude**: By how much?


```python
# ============================================================================
# BACKPROPAGATION: Computing Gradients Step by Step
# ============================================================================
# Let's figure out how to adjust each weight to reduce the loss!

print("=" * 70)
print("BACKPROPAGATION: Computing Gradients Step by Step")
print("=" * 70)
print()

# --- Recap: Values from our forward pass ---

y_true = 1.0      # What we WANTED
y_pred = 0.5560   # What we ACTUALLY got
h1 = 0.56         # Hidden neuron 1's output (after ReLU)
h2 = 0.0          # Hidden neuron 2's output (ReLU made it 0)
x1, x2 = 0.8, 0.2 # Our original inputs
w_out_h1 = 0.7    # Weight we want to update

print("📊 VALUES FROM FORWARD PASS:")
print(f"   y_true (wanted):  {y_true}")
print(f"   y_pred (got):     {y_pred}")
print(f"   h1 (hidden 1):    {h1}")
print(f"   h2 (hidden 2):    {h2}")
print()

# --- STEP 1: How does loss change when prediction changes? ---

print("=" * 70)
print("STEP 1: Derivative of Loss with respect to Prediction")
print("=" * 70)
print()
print("Our loss function is: L = (y_true - y_pred)²")
print()
print("Using calculus (power rule + chain rule):")
print("   ∂L/∂y_pred = -2 × (y_true - y_pred)")
print()

# Calculate the derivative
dL_dy_pred = -2 * (y_true - y_pred)

print("Let's plug in our numbers:")
print(f"   ∂L/∂y_pred = -2 × ({y_true} - {y_pred})")
print(f"             = -2 × {y_true - y_pred:.4f}")
print(f"             = {dL_dy_pred:.4f}")
print()

# --- STEP 2: How does prediction change when z_out changes? ---

print("=" * 70)
print("STEP 2: Derivative through the ReLU Activation")
print("=" * 70)
print()
print("ReLU activation:")
print("   • If input > 0: output = input (derivative = 1)")
print("   • If input ≤ 0: output = 0 (derivative = 0)")
print()
print("Since z_out was positive, ReLU derivative = 1")

dy_dz = 1.0
print(f"   ∂y_pred/∂z_out = {dy_dz}")
print()

# --- STEP 3: How does z_out change when weight changes? ---

print("=" * 70)
print("STEP 3: Derivative of z_out with respect to weight w_out_h1")
print("=" * 70)
print()
print("The formula for z_out is:")
print("   z_out = w_out_h1 × h1 + w_out_h2 × h2 + bias")
print()
print("Taking derivative with respect to w_out_h1:")
print("   ∂z_out/∂w_out_h1 = h1  (because h1 is multiplied by w_out_h1)")
print()

dz_dw_h1 = h1
print(f"   ∂z_out/∂w_out_h1 = {dz_dw_h1}")
print()

# --- STEP 4: Chain rule - multiply all derivatives! ---

print("=" * 70)
print("STEP 4: THE CHAIN RULE - Multiply Everything Together!")
print("=" * 70)
print()
print("The chain rule says:")
print("   ∂L/∂w = (∂L/∂y_pred) × (∂y_pred/∂z_out) × (∂z_out/∂w)")
print()

# Calculate the gradient
dL_dw_out_h1 = dL_dy_pred * dy_dz * dz_dw_h1

print("Calculation:")
print(f"   ∂L/∂w_out_h1 = ({dL_dy_pred:.4f}) × ({dy_dz}) × ({dz_dw_h1})")
print(f"               = {dL_dw_out_h1:.4f}")
print()

# --- What does this gradient tell us? ---

print("=" * 70)
print(f"🎯 GRADIENT for w_out_h1: {dL_dw_out_h1:.4f}")
print("=" * 70)
print()
print("What does this mean?")
print(f"   • The gradient is NEGATIVE ({dL_dw_out_h1:.4f})")
print("   • Negative gradient means: increasing this weight would DECREASE the loss")
print("   • So we SHOULD increase this weight!")
print()
print("💡 Remember: We want to go DOWNHILL (toward lower loss)")
print("   The gradient points uphill, so we go the OPPOSITE direction!")
```


**Output:**
```
======================================================================
BACKPROPAGATION: Computing Gradients Step by Step
======================================================================

📊 VALUES FROM FORWARD PASS:
   y_true (wanted):  1.0
   y_pred (got):     0.556
   h1 (hidden 1):    0.56
   h2 (hidden 2):    0.0

======================================================================
STEP 1: Derivative of Loss with respect to Prediction
======================================================================

Our loss function is: L = (y_true - y_pred)²

Using calculus (power rule + chain rule):
   ∂L/∂y_pred = -2 × (y_true - y_pred)

Let's plug in our numbers:
   ∂L/∂y_pred = -2 × (1.0 - 0.556)
             = -2 × 0.4440
             = -0.8880

======================================================================
STEP 2: Derivative through the ReLU Activation
======================================================================

ReLU activation:
   • If input > 0: output = input (derivative = 1)
   • If input ≤ 0: output = 0 (derivative = 0)

Since z_out was positive, ReLU derivative = 1
   ∂y_pred/∂z_out = 1.0

======================================================================
STEP 3: Derivative of z_out with respect to weight w_out_h1
======================================================================

The formula for z_out is:
   z_out = w_out_h1 × h1 + w_out_h2 × h2 + bias

Taking derivative with respect to w_out_h1:
   ∂z_out/∂w_out_h1 = h1  (because h1 is multiplied by w_out_h1)

   ∂z_out/∂w_out_h1 = 0.56

======================================================================
STEP 4: THE CHAIN RULE - Multiply Everything Together!
======================================================================

The chain rule says:
   ∂L/∂w = (∂L/∂y_pred) × (∂y_pred/∂z_out) × (∂z_out/∂w)

Calculation:
   ∂L/∂w_out_h1 = (-0.8880) × (1.0) × (0.56)
               = -0.4973

======================================================================
🎯 GRADIENT for w_out_h1: -0.4973
======================================================================

What does this mean?
   • The gradient is NEGATIVE (-0.4973)
   • Negative gradient means: increasing this weight would DECREASE the loss
   • So we SHOULD increase this weight!

💡 Remember: We want to go DOWNHILL (toward lower loss)
   The gradient points uphill, so we go the OPPOSITE direction!

```


### 3.1.4.1 Understanding Gradients: Why Do We SUBTRACT?

This is super confusing at first, so let's make it crystal clear!

---

## 🏔️ The Hill Analogy

Imagine you're lost in a foggy mountain and want to find the lowest valley:

```
                        THE MOUNTAIN OF LOSS
    
    Loss ↑                    You are here! 🧑
         │                          ●
      4  │                        ╱   ╲
         │                      ╱       ╲
      3  │                    ╱           ╲    Gradient says "go RIGHT"
         │                  ╱               ╲   (points uphill)
      2  │                ╱                   ╲
         │              ╱                       ╲
      1  │            ╱                           ╲
         │          ╱         THE VALLEY           ╲
      0  │        ●────────────(goal!)──────────────●────► Weight
         └───────────────────────────────────────────────
               1       2       3       4       5
                           ↑
                    Optimal weight = 3
                    (lowest point!)
```

**Key insight:**
- The **gradient** tells you which way is UPHILL (toward higher loss)
- We want to go DOWNHILL (toward lower loss)
- So we go the **OPPOSITE** direction → we SUBTRACT the gradient!

---

## 📊 Gradient vs Negative Gradient

| Term | Direction | What We Want? |
|------|-----------|---------------|
| **Gradient** | Points UPHILL (higher loss) | ❌ No! |
| **Negative Gradient** | Points DOWNHILL (lower loss) | ✅ Yes! |

---

## 🔢 The Update Rule

$$\text{new weight} = \text{old weight} - \text{learning rate} \times \text{gradient}$$

**Why subtract?**
- If gradient is POSITIVE → weight is too high → subtract to make it smaller
- If gradient is NEGATIVE → weight is too low → subtracting a negative = adding → makes it bigger

**Example:**
```
Current weight: w = 1.0
Gradient: ∂L/∂w = -4 (negative = weight should go UP)
Learning rate: 0.1

New weight = 1.0 - (0.1 × -4)
           = 1.0 - (-0.4)
           = 1.0 + 0.4
           = 1.4 ✓ Weight increased!
```


```python
# ============================================================================
# GRADIENT vs NEGATIVE GRADIENT: A Hands-On Example
# ============================================================================
# Let's see gradients in action with a simple example!

print("=" * 70)
print("GRADIENT vs NEGATIVE GRADIENT: A Hands-On Example")
print("=" * 70)
print()

# --- Our simple loss function ---

# Loss = (weight - 3)² - this loss is 0 when weight = 3

def loss_function(w):
    """Compute loss: (w - 3)². Minimum at w = 3."""
    return (w - 3) ** 2

def gradient(w):
    """Compute gradient: 2(w - 3). Tells us which way is uphill."""
    return 2 * (w - 3)

print("📊 Our Loss Function: L(w) = (w - 3)²")
print("   This is shaped like a bowl 🥣")
print("   The bottom of the bowl (minimum) is at w = 3")
print()

# --- Current position ---

w_current = 1.0  # Start at weight = 1

print("=" * 70)
print(f"📍 CURRENT POSITION: weight = {w_current}")
print("=" * 70)
print()
print(f"Loss at this weight:")
print(f"   L({w_current}) = ({w_current} - 3)² = ({w_current - 3})² = {loss_function(w_current)}")
print()

# --- Compute the gradient ---

grad = gradient(w_current)

print("=" * 70)
print("📐 COMPUTING THE GRADIENT")
print("=" * 70)
print()
print(f"Gradient formula: dL/dw = 2(w - 3)")
print(f"At w = {w_current}:")
print(f"   dL/dw = 2 × ({w_current} - 3)")
print(f"         = 2 × {w_current - 3}")
print(f"         = {grad}")
print()

# --- Interpret the gradient ---

print("=" * 70)
print("🔍 WHAT DOES THIS GRADIENT MEAN?")
print("=" * 70)
print()
print(f"The gradient is {grad} (NEGATIVE)")
print()
print("A NEGATIVE gradient means:")
print("   • The slope is going DOWN as we move RIGHT")
print("   • UPHILL (toward higher loss) is to the LEFT")
print("   • DOWNHILL (toward lower loss) is to the RIGHT")
print()
print("We want to go DOWNHILL, so we should move RIGHT (increase w)!")
print()

# --- Gradient ascent vs descent ---

learning_rate = 0.1

print("=" * 70)
print("🔄 GRADIENT ASCENT vs GRADIENT DESCENT")
print("=" * 70)
print()

# WRONG: Gradient Ascent
w_ascent = w_current + learning_rate * grad
loss_ascent = loss_function(w_ascent)

print("❌ GRADIENT ASCENT (Adding the gradient - WRONG!):")
print(f"   new_w = old_w + learning_rate × gradient")
print(f"         = {w_current} + {learning_rate} × ({grad})")
print(f"         = {w_current} + {learning_rate * grad}")
print(f"         = {w_ascent}")
print(f"   New loss: {loss_ascent:.2f} ← WORSE! We went UPHILL 📈")
print()

# CORRECT: Gradient Descent
w_descent = w_current - learning_rate * grad
loss_descent = loss_function(w_descent)

print("✅ GRADIENT DESCENT (Subtracting the gradient - CORRECT!):")
print(f"   new_w = old_w - learning_rate × gradient")
print(f"         = {w_current} - {learning_rate} × ({grad})")
print(f"         = {w_current} - ({learning_rate * grad})")
print(f"         = {w_descent}")
print(f"   New loss: {loss_descent:.2f} ← BETTER! We went DOWNHILL 📉")
print()

# --- Summary ---

print("=" * 70)
print("💡 KEY TAKEAWAY")
print("=" * 70)
print()
print("To minimize loss (go downhill), we SUBTRACT the gradient!")
print()
print("   new_weight = old_weight - learning_rate × gradient")
print()
print("This is called GRADIENT DESCENT because we 'descend' down the hill.")
```


**Output:**
```
======================================================================
GRADIENT vs NEGATIVE GRADIENT: A Hands-On Example
======================================================================

📊 Our Loss Function: L(w) = (w - 3)²
   This is shaped like a bowl 🥣
   The bottom of the bowl (minimum) is at w = 3

======================================================================
📍 CURRENT POSITION: weight = 1.0
======================================================================

Loss at this weight:
   L(1.0) = (1.0 - 3)² = (-2.0)² = 4.0

======================================================================
📐 COMPUTING THE GRADIENT
======================================================================

Gradient formula: dL/dw = 2(w - 3)
At w = 1.0:
   dL/dw = 2 × (1.0 - 3)
         = 2 × -2.0
         = -4.0

======================================================================
🔍 WHAT DOES THIS GRADIENT MEAN?
======================================================================

The gradient is -4.0 (NEGATIVE)

A NEGATIVE gradient means:
   • The slope is going DOWN as we move RIGHT
   • UPHILL (toward higher loss) is to the LEFT
   • DOWNHILL (toward lower loss) is to the RIGHT

We want to go DOWNHILL, so we should move RIGHT (increase w)!

======================================================================
🔄 GRADIENT ASCENT vs GRADIENT DESCENT
======================================================================

❌ GRADIENT ASCENT (Adding the gradient - WRONG!):
   new_w = old_w + learning_rate × gradient
         = 1.0 + 0.1 × (-4.0)
         = 1.0 + -0.4
         = 0.6
   New loss: 5.76 ← WORSE! We went UPHILL 📈

✅ GRADIENT DESCENT (Subtracting the gradient - CORRECT!):
   new_w = old_w - learning_rate × gradient
         = 1.0 - 0.1 × (-4.0)
         = 1.0 - (-0.4)
         = 1.4
   New loss: 2.56 ← BETTER! We went DOWNHILL 📉

======================================================================
💡 KEY TAKEAWAY
======================================================================

To minimize loss (go downhill), we SUBTRACT the gradient!

   new_weight = old_weight - learning_rate × gradient

This is called GRADIENT DESCENT because we 'descend' down the hill.

```


```python
# ============================================================================
# WATCHING GRADIENT DESCENT IN ACTION (Multiple Steps!)
# ============================================================================
# Let's see how the weight gets closer to optimal over many steps.

print("=" * 70)
print("WATCHING GRADIENT DESCENT IN ACTION")
print("=" * 70)
print()

# --- Setup ---

w = 0.0               # Start at weight = 0 (far from optimal)
learning_rate = 0.1   # How big each step is
optimal_w = 3.0       # The best weight (where loss = 0)

print(f"🎯 GOAL: Find the weight that minimizes L(w) = (w - 3)²")
print(f"         The optimal weight is 3.0 (where loss = 0)")
print()
print(f"📍 Starting position: w = {w}")
print(f"📏 Learning rate: {learning_rate}")
print()

# --- Run gradient descent for 8 steps ---

print("=" * 70)
print("STEP-BY-STEP PROGRESS")
print("=" * 70)
print()
print(f"{'Step':<6} {'Weight':<12} {'Loss':<12} {'Gradient':<12} {'Action':<20}")
print("-" * 70)

for step in range(8):
    # Calculate current loss and gradient
    current_loss = loss_function(w)
    grad = gradient(w)
    
    # Determine direction
    if grad < 0:
        action = "→ increase w"
    elif grad > 0:
        action = "← decrease w"
    else:
        action = "■ at minimum!"
    
    # Print current state
    print(f"{step:<6} {w:<12.4f} {current_loss:<12.4f} {grad:<12.4f} {action:<20}")
    
    # Update weight using gradient descent
    w = w - learning_rate * grad

# Print final state
print(f"{'FINAL':<6} {w:<12.4f} {loss_function(w):<12.4f}")
print()

# --- Summary ---

print("=" * 70)
print("📊 WHAT HAPPENED:")
print("=" * 70)
print()
print(f"   • We started at w = 0.0 (loss = 9.0)")
print(f"   • After 8 steps, we're at w = {w:.4f} (loss = {loss_function(w):.4f})")
print(f"   • The optimal is w = 3.0 (loss = 0.0)")
print()
print("Notice how:")
print("   1. The weight gets closer to 3 each step")
print("   2. The loss gets smaller each step")
print("   3. The gradient gets smaller as we approach the minimum")
print("   4. We never overshoot because learning rate is small enough")
print()
print("🎉 This is how neural networks learn!")
```


**Output:**
```
======================================================================
WATCHING GRADIENT DESCENT IN ACTION
======================================================================

🎯 GOAL: Find the weight that minimizes L(w) = (w - 3)²
         The optimal weight is 3.0 (where loss = 0)

📍 Starting position: w = 0.0
📏 Learning rate: 0.1

======================================================================
STEP-BY-STEP PROGRESS
======================================================================

Step   Weight       Loss         Gradient     Action              
----------------------------------------------------------------------
0      0.0000       9.0000       -6.0000      → increase w        
1      0.6000       5.7600       -4.8000      → increase w        
2      1.0800       3.6864       -3.8400      → increase w        
3      1.4640       2.3593       -3.0720      → increase w        
4      1.7712       1.5099       -2.4576      → increase w        
5      2.0170       0.9664       -1.9661      → increase w        
6      2.2136       0.6185       -1.5729      → increase w        
7      2.3709       0.3958       -1.2583      → increase w        
FINAL  2.4967       0.2533      

======================================================================
📊 WHAT HAPPENED:
======================================================================

   • We started at w = 0.0 (loss = 9.0)
   • After 8 steps, we're at w = 2.4967 (loss = 0.2533)
   • The optimal is w = 3.0 (loss = 0.0)

Notice how:
   1. The weight gets closer to 3 each step
   2. The loss gets smaller each step
   3. The gradient gets smaller as we approach the minimum
   4. We never overshoot because learning rate is small enough

🎉 This is how neural networks learn!

```


### 3.1.4.2 Summary: The Two Directions

Let's make sure we're crystal clear on the difference:

---

## 📊 Gradient Ascent vs Gradient Descent

| Method | Formula | Direction | When to Use |
|--------|---------|-----------|-------------|
| **Gradient Ascent** | $w_{new} = w + \alpha \cdot gradient$ | UPHILL (maximize) | Finding highest point |
| **Gradient Descent** | $w_{new} = w - \alpha \cdot gradient$ | DOWNHILL (minimize) | **Minimizing loss ✓** |

---

## 🧮 Why the Subtraction Works Perfectly

The magic of subtracting the gradient:

```
If gradient > 0:
   → "Increasing w makes loss go UP"
   → We want loss to go DOWN
   → So DECREASE w
   → w_new = w_old - (positive number) 
   → Weight gets SMALLER ✓

If gradient < 0:
   → "Increasing w makes loss go DOWN"
   → That's good! Let's increase w
   → So INCREASE w
   → w_new = w_old - (negative number)
   → w_new = w_old + (positive number)
   → Weight gets BIGGER ✓
```

**The subtraction automatically does the right thing!**

---

## 🎯 Real World Analogy

Think of a ball rolling downhill:
- **Gravity** always pulls the ball toward the lowest point
- The **negative gradient** is like gravity for neural networks
- It always pulls weights toward lower loss

```
      ●  ← Ball (our weight)
     ╱ ╲
    ╱   ╲
   ╱     ╲  Gravity pulls DOWN
  ╱       ╲   (negative gradient direction)
 ╱    ↓    ╲
●───────────●  ← Lowest point (minimum loss)
```


```python
# ============================================================================
# GRADIENT DESCENT: Actually Updating the Weights!
# ============================================================================
# Now let's use what we learned to update a weight and verify it helps!

print("=" * 70)
print("GRADIENT DESCENT: Updating the Weights")
print("=" * 70)
print()

# --- Recall our values from backpropagation ---

w_out_h1 = 0.7          # Current weight value
dL_dw_out_h1 = -0.4973  # Gradient we computed
learning_rate = 0.1     # How big of a step to take

print("📊 CURRENT VALUES:")
print(f"   Weight:        w_out_h1 = {w_out_h1}")
print(f"   Gradient:      ∂L/∂w = {dL_dw_out_h1:.4f}")
print(f"   Learning rate: α = {learning_rate}")
print()

# --- Apply the update rule ---

print("=" * 70)
print("APPLYING THE UPDATE RULE")
print("=" * 70)
print()
print("Formula: w_new = w_old - α × gradient")
print()

# Step-by-step calculation
step1 = learning_rate * dL_dw_out_h1
w_out_h1_new = w_out_h1 - step1

print("Step 1: Multiply learning rate by gradient")
print(f"        α × gradient = {learning_rate} × ({dL_dw_out_h1:.4f})")
print(f"                     = {step1:.4f}")
print()
print("Step 2: Subtract from old weight")
print(f"        w_new = {w_out_h1} - ({step1:.4f})")
print(f"              = {w_out_h1_new:.4f}")
print()

# --- Interpret the result ---

print("=" * 70)
print(f"✅ WEIGHT UPDATED: {w_out_h1:.4f} → {w_out_h1_new:.4f}")
print("=" * 70)
print()

# Did the weight increase or decrease?
if w_out_h1_new > w_out_h1:
    change = "INCREASED"
else:
    change = "DECREASED"

print(f"The weight {change} by {abs(w_out_h1_new - w_out_h1):.4f}")
print()
print("Why did this happen?")
print(f"   • The gradient was NEGATIVE ({dL_dw_out_h1:.4f})")
print(f"   • Negative gradient means: increasing w would DECREASE loss")
print(f"   • So the update rule increased the weight (correct!)")
print()

# --- Verify: Does the new weight actually reduce loss? ---

print("=" * 70)
print("🔍 VERIFICATION: Does the new weight reduce loss?")
print("=" * 70)
print()

# Values from our network
h1_val = 0.56
h2_val = 0.0
w_out_h2 = 0.4
b_out = 0.05
y_true = 1.0

# Calculate predictions with old and new weights
y_pred_old = w_out_h1 * h1_val + w_out_h2 * h2_val + b_out
y_pred_new = w_out_h1_new * h1_val + w_out_h2 * h2_val + b_out

# Calculate losses
loss_old = (y_true - y_pred_old) ** 2
loss_new = (y_true - y_pred_new) ** 2

print("With OLD weight:")
print(f"   Prediction = {y_pred_old:.4f}")
print(f"   Loss = ({y_true} - {y_pred_old:.4f})² = {loss_old:.4f}")
print()
print("With NEW weight:")
print(f"   Prediction = {y_pred_new:.4f}")
print(f"   Loss = ({y_true} - {y_pred_new:.4f})² = {loss_new:.4f}")
print()

# Compare
loss_reduction = loss_old - loss_new
print("=" * 70)
print(f"🎉 LOSS REDUCED BY {loss_reduction:.4f}!")
print("=" * 70)
print()
print("The gradient descent update worked perfectly!")
print("By adjusting the weight in the direction of lower loss,")
print("we made our prediction closer to the true value.")
```


**Output:**
```
======================================================================
GRADIENT DESCENT: Updating the Weights
======================================================================

📊 CURRENT VALUES:
   Weight:        w_out_h1 = 0.7
   Gradient:      ∂L/∂w = -0.4973
   Learning rate: α = 0.1

======================================================================
APPLYING THE UPDATE RULE
======================================================================

Formula: w_new = w_old - α × gradient

Step 1: Multiply learning rate by gradient
        α × gradient = 0.1 × (-0.4973)
                     = -0.0497

Step 2: Subtract from old weight
        w_new = 0.7 - (-0.0497)
              = 0.7497

======================================================================
✅ WEIGHT UPDATED: 0.7000 → 0.7497
======================================================================

The weight INCREASED by 0.0497

Why did this happen?
   • The gradient was NEGATIVE (-0.4973)
   • Negative gradient means: increasing w would DECREASE loss
   • So the update rule increased the weight (correct!)

======================================================================
🔍 VERIFICATION: Does the new weight reduce loss?
======================================================================

With OLD weight:
   Prediction = 0.4420
   Loss = (1.0 - 0.4420)² = 0.3114

With NEW weight:
   Prediction = 0.4698
   Loss = (1.0 - 0.4698)² = 0.2811

======================================================================
🎉 LOSS REDUCED BY 0.0303!
======================================================================

The gradient descent update worked perfectly!
By adjusting the weight in the direction of lower loss,
we made our prediction closer to the true value.

```


### 3.1.5 The Complete Training Loop (Putting It All Together!)

Now you understand all the pieces! Here's how they fit together in the **training loop**:

---

## 🔄 The 5 Steps of Training (Repeated Thousands of Times!)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     THE NEURAL NETWORK TRAINING LOOP                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                                                                 │    │
│  │  STEP 1: FORWARD PASS                                           │    │
│  │          Feed input through the network to get a prediction     │    │
│  │          Input → Weights → Activation → Output                  │    │
│  │                                                                 │    │
│  │  STEP 2: COMPUTE LOSS                                           │    │
│  │          Measure how wrong the prediction is                    │    │
│  │          Loss = (True - Prediction)²                            │    │
│  │                                                                 │    │
│  │  STEP 3: BACKWARD PASS (Backpropagation)                        │    │
│  │          Calculate gradients using the chain rule               │    │
│  │          "How much did each weight contribute to the error?"    │    │
│  │                                                                 │    │
│  │  STEP 4: UPDATE WEIGHTS (Gradient Descent)                      │    │
│  │          Adjust each weight to reduce the loss                  │    │
│  │          weight = weight - learning_rate × gradient             │    │
│  │                                                                 │    │
│  │  STEP 5: REPEAT!                                                │    │
│  │          Go back to Step 1 and do it again                      │    │
│  │          Each time is called an "epoch" or "step"               │    │
│  │                                                                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    ↑                                    │
│                                    │                                    │
│                              Repeat 1000s                               │
│                               of times!                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🎓 This is EXACTLY How GPT Learns!

| Component | What Gets Updated | How |
|-----------|------------------|-----|
| Embeddings | The 32 numbers for each word | Gradient descent |
| Attention weights | Which words to pay attention to | Gradient descent |
| Feed-forward weights | How to transform information | Gradient descent |
| All biases | All the starting offsets | Gradient descent |

After millions of training steps:
- Words with similar meanings get similar embeddings
- Attention learns to focus on relevant context
- The model learns to predict the next word accurately!


```python
# ============================================================================
# COMPLETE TRAINING LOOP IN PYTORCH
# ============================================================================
# Now let's see how PyTorch makes all this easy!
# PyTorch does the backpropagation automatically for us.

print("=" * 70)
print("COMPLETE TRAINING LOOP IN PYTORCH")
print("=" * 70)
print()

# --- Define our neural network ---

class SimpleNetwork(nn.Module):
    """
    A simple 2-layer neural network for learning demonstrations.
    
    Architecture:
        Input (2) → Hidden Layer (2 neurons) → ReLU → Output (1)
    """
    
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2)   # Hidden layer: 2 inputs → 2 neurons
        self.output = nn.Linear(2, 1)   # Output layer: 2 neurons → 1 output
    
    def forward(self, x):
        x = F.relu(self.hidden(x))      # Apply ReLU activation
        x = self.output(x)               # Output (no activation for regression)
        return x

# --- Create the network and optimizer ---

torch.manual_seed(42)  # For reproducibility

net = SimpleNetwork()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

print("📦 Network created!")
print(f"   Total parameters: {sum(p.numel() for p in net.parameters())}")
print()

# --- Prepare training data ---

x_train = torch.tensor([[0.8, 0.2]])  # Input: action=0.8, romance=0.2
y_train = torch.tensor([[1.0]])       # Target: we want output = 1.0

print("📊 TRAINING DATA:")
print(f"   Input:  {x_train.tolist()} (action=0.8, romance=0.2)")
print(f"   Target: {y_train.tolist()} (we want output = 1.0)")
print()

# --- The training loop ---

print("=" * 70)
print("TRAINING LOOP (10 epochs)")
print("=" * 70)
print()
print(f"{'Epoch':<8} {'Prediction':<15} {'Loss':<15} {'Status':<20}")
print("-" * 58)

for epoch in range(10):
    # STEP 1: Forward pass - get prediction
    y_pred = net(x_train)
    
    # STEP 2: Compute loss - measure error
    loss = F.mse_loss(y_pred, y_train)
    
    # Determine status
    if loss.item() > 0.1:
        status = "Learning..."
    elif loss.item() > 0.01:
        status = "Getting better!"
    else:
        status = "Almost perfect!"
    
    # Print progress
    if epoch % 2 == 0 or epoch == 9:
        print(f"{epoch:<8} {y_pred.item():<15.4f} {loss.item():<15.4f} {status:<20}")
    
    # STEP 3: Backward pass - compute gradients
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()        # Compute new gradients
    
    # STEP 4: Update weights
    optimizer.step()

# --- Final results ---

print("-" * 58)
print()
print("=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print()
print(f"Final prediction: {y_pred.item():.4f}")
print(f"Target was:       1.0")
print(f"Difference:       {abs(1.0 - y_pred.item()):.4f}")
print()
print("🎉 The network learned to produce the correct output!")
print()
print("PyTorch did all the hard work:")
print("   • Forward pass: automatic")
print("   • Loss computation: F.mse_loss()")
print("   • Backward pass: loss.backward()")
print("   • Weight updates: optimizer.step()")
```


**Output:**
```
======================================================================
COMPLETE TRAINING LOOP IN PYTORCH
======================================================================

📦 Network created!
   Total parameters: 9

📊 TRAINING DATA:
   Input:  [[0.800000011920929, 0.20000000298023224]] (action=0.8, romance=0.2)
   Target: [[1.0]] (we want output = 1.0)

======================================================================
TRAINING LOOP (10 epochs)
======================================================================

Epoch    Prediction      Loss            Status              
----------------------------------------------------------
0        0.5456          0.2065          Learning...         
2        0.7946          0.0422          Getting better!     
4        0.9066          0.0087          Almost perfect!     
6        0.9579          0.0018          Almost perfect!     
8        0.9812          0.0004          Almost perfect!     
9        0.9874          0.0002          Almost perfect!     
----------------------------------------------------------

======================================================================
✅ TRAINING COMPLETE!
======================================================================

Final prediction: 0.9874
Target was:       1.0
Difference:       0.0126

🎉 The network learned to produce the correct output!

PyTorch did all the hard work:
   • Forward pass: automatic
   • Loss computation: F.mse_loss()
   • Backward pass: loss.backward()
   • Weight updates: optimizer.step()

```


### 3.1.6 Key Takeaways: Neural Network Fundamentals

🎓 **Congratulations!** You now understand the core of how neural networks learn!

---

## 📋 Summary Table

| Concept | What It Is | Simple Explanation |
|---------|-----------|-------------------|
| **Neuron** | Basic computing unit | Takes inputs, multiplies by weights, adds bias, applies activation |
| **Weights** | Learnable numbers | How important each input is (gets adjusted during training) |
| **Bias** | Learnable offset | A starting point that shifts the output up or down |
| **Activation** | Non-linear function | Makes the network able to learn complex patterns (ReLU, sigmoid) |
| **Forward Pass** | Computing output | Input → through all layers → prediction |
| **Loss Function** | Error measurement | "How wrong is our prediction?" (MSE, Cross-Entropy) |
| **Gradient** | Slope of the loss | "Which way is uphill?" |
| **Backpropagation** | Finding gradients | Uses chain rule to compute how each weight affects loss |
| **Gradient Descent** | Updating weights | `new = old - learning_rate × gradient` |

---

## 🔗 How This Connects to GPT

Everything in GPT works the same way!

| GPT Component | What It Is | How It Learns |
|---------------|-----------|---------------|
| **Embedding Matrix** | 27 words × 32 numbers | Random at first → trained via gradient descent |
| **Attention Weights** | Q, K, V matrices | Learned to focus on relevant words |
| **Feed-Forward Layers** | Just like our SimpleNetwork! | Same neurons, weights, biases |
| **All Parameters** | ~30,000+ numbers | All updated by gradient descent |

**The only difference:** GPT is bigger and the loss function is "predict the next word" instead of movie preferences!

Now let's see how embeddings actually learn meaning! 👇


### 3.2 How Embeddings Learn Meaning (The Magic!)

Remember our embedding matrix? Those 32 random numbers for each word?

Here's the amazing part: **after training, they're not random anymore!**

---

## 🎲 Before Training vs 🎯 After Training

| Stage | What the Numbers Look Like | Meaning |
|-------|---------------------------|---------|
| **Before** | `[0.42, -0.17, 0.89, ...]` (random) | Meaningless noise |
| **After** | `[0.42, -0.17, 0.89, ...]` (trained) | Encodes word relationships! |

---

## 🧠 How Does This Happen?

1. **The model tries to predict the next word**
   - Input: "mary had a little"
   - Correct answer: "lamb"
   
2. **If wrong, backpropagation adjusts the embeddings**
   - "The embedding for 'lamb' should be more similar to 'little'"
   - Gradients flow back and tweak the 32 numbers
   
3. **Over thousands of examples, patterns emerge**
   - Words appearing in similar contexts get similar embeddings
   - Related words cluster together

---

## 🐑 In Our "Mary Had a Little Lamb" Model

After training, the model might learn:
- **"lamb" and "fleece"** have similar vectors (both describe the lamb)
- **"mary" and "went"** are connected (Mary does the going)
- **"school" and "rules"** cluster together (both about school)

---

## 🤯 The Famous Example (from larger models)

With enough training data, embeddings can do this:

$$\text{embedding("king")} - \text{embedding("man")} + \text{embedding("woman")} \approx \text{embedding("queen")}$$

The model learned that "king" is to "man" as "queen" is to "woman"!

**We never told it this.** It discovered the relationship by predicting words!


### 3.3 Understanding Attention Heads (How Words Talk to Each Other)

Now comes the **coolest part of GPT**: **attention heads**!

---

## 🤔 The Problem

We have word embeddings (32 numbers per word). But words in isolation are useless!

Consider: "The lamb was sure to go"
- What does "it" refer to in the previous sentence?
- What is the lamb going TO?

**Words need to share information with each other!**

---

## 💡 The Solution: Attention

**Attention** lets each word look at other words and gather relevant information.

Think of it like a student taking notes in class:
- You're writing about "lamb"
- You glance at "mary" to remember whose lamb it is
- You glance at "little" to remember what kind of lamb

---

## 🎭 Why MULTIPLE Heads?

One attention head might miss important relationships. So we use **multiple heads**!

With `n_heads = 2`:

| Head | What It Might Learn | Example |
|------|---------------------|---------|
| Head 1 | "Who owns this?" | lamb → looks at → mary |
| Head 2 | "What action happened?" | lamb → looks at → had |

**Each head looks at the sentence differently!** Then we combine their findings.

---

## ✂️ How Do We Split the Embedding?

Since `embedding_dim = 32` and `n_heads = 2`:

```
Original embedding: 32 dimensions
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
    ┌───────────┐       ┌───────────┐
    │  Head 1   │       │  Head 2   │
    │ 16 dims   │       │ 16 dims   │
    └─────┬─────┘       └─────┬─────┘
          │                   │
          └─────────┬─────────┘
                    ↓
          Concatenate back → 32 dimensions
```

Each head works with 16 dimensions, then we glue the results back together!


```python
# ============================================================================
# HOW THE EMBEDDING IS SPLIT ACROSS ATTENTION HEADS
# ============================================================================
# Let's see exactly how each head gets its portion of the embedding.

print("=" * 70)
print("HOW THE EMBEDDING IS SPLIT ACROSS ATTENTION HEADS")
print("=" * 70)
print()

# --- Calculate head size ---

head_size = embedding_dim // n_heads  # 32 ÷ 2 = 16 dimensions per head

print("📊 THE SPLIT:")
print(f"   Total embedding dimensions: {embedding_dim}")
print(f"   Number of attention heads:  {n_heads}")
print(f"   Dimensions per head:        {embedding_dim} ÷ {n_heads} = {head_size}")
print()

# --- Visual representation ---

print("📐 VISUAL REPRESENTATION:")
print()
print("   Full token embedding (32 dimensions):")
print("   ┌─────────────────────────────────────────────────────────────┐")
print("   │  dim0, dim1, dim2, ... dim15 │ dim16, dim17, ... dim31     │")
print("   └───────────────┬──────────────┴────────────────┬────────────┘")
print("                   │                               │")
print("                   ▼                               ▼")
print("           ┌───────────────┐               ┌───────────────┐")
print("           │    HEAD 1     │               │    HEAD 2     │")
print("           │  (dims 0-15)  │               │ (dims 16-31)  │")
print("           │   16 numbers  │               │   16 numbers  │")
print("           └───────────────┘               └───────────────┘")
print()
print(f"💡 Each head works with {head_size} dimensions independently,")
print(f"   then we combine their results back into {embedding_dim} dimensions!")
```


**Output:**
```
======================================================================
HOW THE EMBEDDING IS SPLIT ACROSS ATTENTION HEADS
======================================================================

📊 THE SPLIT:
   Total embedding dimensions: 32
   Number of attention heads:  2
   Dimensions per head:        32 ÷ 2 = 16

📐 VISUAL REPRESENTATION:

   Full token embedding (32 dimensions):
   ┌─────────────────────────────────────────────────────────────┐
   │  dim0, dim1, dim2, ... dim15 │ dim16, dim17, ... dim31     │
   └───────────────┬──────────────┴────────────────┬────────────┘
                   │                               │
                   ▼                               ▼
           ┌───────────────┐               ┌───────────────┐
           │    HEAD 1     │               │    HEAD 2     │
           │  (dims 0-15)  │               │ (dims 16-31)  │
           │   16 numbers  │               │   16 numbers  │
           └───────────────┘               └───────────────┘

💡 Each head works with 16 dimensions independently,
   then we combine their results back into 32 dimensions!

```


### 3.3.1 Concrete Example: Watch Attention Heads in Action!

Let's trace through exactly what happens with real words from our vocabulary!

We'll use: **"mary"**, **"had"**, **"a"**

You'll see:
1. Each word gets a 32-number embedding
2. The embedding is split between 2 heads (16 numbers each)
3. Each head computes attention differently
4. The results are combined back together


```python
# ============================================================================
# STEP 1: SELECT THREE WORDS FROM OUR VOCABULARY
# ============================================================================
# Let's trace through attention with real words!

print("=" * 70)
print("STEP 1: SELECT TOKENS FROM OUR VOCABULARY")
print("=" * 70)
print()

# --- Pick 3 words and get their token IDs ---

word1, word2, word3 = "mary", "had", "a"

id1 = word2idx[word1]
id2 = word2idx[word2]
id3 = word2idx[word3]

print(f"📝 We're using these 3 words:")
print(f"   '{word1}' → Token ID: {id1}")
print(f"   '{word2}' → Token ID: {id2}")
print(f"   '{word3}'   → Token ID: {id3}")
print()

# --- Get embeddings for each word (32 numbers each) ---

print("=" * 70)
print("STEP 2: CONVERT WORDS TO EMBEDDINGS (32 numbers each)")
print("=" * 70)
print()

torch.manual_seed(42)  # For reproducibility
demo_embedding = nn.Embedding(vocab_size, embedding_dim)

token_ids = torch.tensor([id1, id2, id3])
embeddings = demo_embedding(token_ids)  # Shape: [3, 32]

print(f"📊 Embedding matrix lookup:")
print(f"   Input: 3 token IDs → {token_ids.tolist()}")
print(f"   Output shape: {embeddings.shape} (3 words × 32 numbers each)")
print()

# Show a preview of each embedding
for i, word in enumerate([word1, word2, word3]):
    emb_preview = embeddings[i].data[:6].tolist()
    print(f"   '{word}' embedding (first 6 of 32):")
    print(f"      [{emb_preview[0]:.3f}, {emb_preview[1]:.3f}, {emb_preview[2]:.3f}, {emb_preview[3]:.3f}, {emb_preview[4]:.3f}, {emb_preview[5]:.3f}, ...]")
    print()
```


**Output:**
```
======================================================================
STEP 1: SELECT TOKENS FROM OUR VOCABULARY
======================================================================

📝 We're using these 3 words:
   'mary' → Token ID: 32
   'had' → Token ID: 28
   'a'   → Token ID: 13

======================================================================
STEP 2: CONVERT WORDS TO EMBEDDINGS (32 numbers each)
======================================================================

📊 Embedding matrix lookup:
   Input: 3 token IDs → [32, 28, 13]
   Output shape: torch.Size([3, 32]) (3 words × 32 numbers each)

   'mary' embedding (first 6 of 32):
      [-1.225, 0.963, -1.579, 0.672, -0.060, 0.070, ...]

   'had' embedding (first 6 of 32):
      [-0.711, -0.387, 0.958, -0.823, -2.391, 0.322, ...]

   'a' embedding (first 6 of 32):
      [0.684, -1.325, -0.516, 0.600, -0.470, -0.609, ...]


```


```python
# ============================================================================
# STEP 3: SPLIT EACH EMBEDDING BETWEEN THE TWO HEADS
# ============================================================================
# Dims 0-15 go to Head 1, dims 16-31 go to Head 2

print("=" * 70)
print("STEP 3: SPLIT EACH EMBEDDING INTO 2 HEADS")
print("=" * 70)
print()

# --- Calculate head size ---

head_size = embedding_dim // n_heads  # 32 ÷ 2 = 16 dimensions per head

print(f"📊 Splitting {embedding_dim} dimensions into {n_heads} heads:")
print(f"   Head 1 gets dimensions 0-15  ({head_size} numbers)")
print(f"   Head 2 gets dimensions 16-31 ({head_size} numbers)")
print()

# --- Show how each word's embedding is split ---

for i, word in enumerate([word1, word2, word3]):
    full_emb = embeddings[i].data
    head1_portion = full_emb[:head_size]   # Dimensions 0-15
    head2_portion = full_emb[head_size:]   # Dimensions 16-31
    
    print(f"'{word}' (Token {token_ids[i].item()}):")
    print(f"   Full embedding: [{full_emb[0]:.2f}, {full_emb[1]:.2f}, ... {full_emb[31]:.2f}]  (32 nums)")
    print(f"      ↓ split ↓")
    print(f"   Head 1 gets:    [{head1_portion[0]:.2f}, {head1_portion[1]:.2f}, ... {head1_portion[15]:.2f}]  (16 nums)")
    print(f"   Head 2 gets:    [{head2_portion[0]:.2f}, {head2_portion[1]:.2f}, ... {head2_portion[15]:.2f}]  (16 nums)")
    print()
```


**Output:**
```
======================================================================
STEP 3: SPLIT EACH EMBEDDING INTO 2 HEADS
======================================================================

📊 Splitting 32 dimensions into 2 heads:
   Head 1 gets dimensions 0-15  (16 numbers)
   Head 2 gets dimensions 16-31 (16 numbers)

'mary' (Token 32):
   Full embedding: [-1.22, 0.96, ... -0.00]  (32 nums)
      ↓ split ↓
   Head 1 gets:    [-1.22, 0.96, ... 0.27]  (16 nums)
   Head 2 gets:    [-0.35, -0.12, ... -0.00]  (16 nums)

'had' (Token 28):
   Full embedding: [-0.71, -0.39, ... -0.96]  (32 nums)
      ↓ split ↓
   Head 1 gets:    [-0.71, -0.39, ... -0.42]  (16 nums)
   Head 2 gets:    [1.86, -1.08, ... -0.96]  (16 nums)

'a' (Token 13):
   Full embedding: [0.68, -1.32, ... -0.75]  (32 nums)
      ↓ split ↓
   Head 1 gets:    [0.68, -1.32, ... 0.69]  (16 nums)
   Head 2 gets:    [-0.49, 1.14, ... -0.75]  (16 nums)


```


```python
# ============================================================================
# STEP 4: EACH HEAD COMPUTES ATTENTION INDEPENDENTLY
# ============================================================================
# Each head has its own Q, K, V matrices (learnable weights!)

print("=" * 70)
print("STEP 4: EACH HEAD COMPUTES Q, K, V INDEPENDENTLY")
print("=" * 70)
print()

print("🧠 What are Q, K, V?")
print("   Q (Query):  'What am I looking for?'")
print("   K (Key):    'What do I have to offer?'")
print("   V (Value):  'What information should I share?'")
print()

# --- Create Q, K, V matrices for each head ---

torch.manual_seed(42)

# Head 1 matrices
head1_query = nn.Linear(head_size, head_size, bias=False)
head1_key = nn.Linear(head_size, head_size, bias=False)
head1_value = nn.Linear(head_size, head_size, bias=False)

# Head 2 matrices (different learned weights!)
head2_query = nn.Linear(head_size, head_size, bias=False)
head2_key = nn.Linear(head_size, head_size, bias=False)
head2_value = nn.Linear(head_size, head_size, bias=False)

print("📊 Each head has its own learnable matrices:")
print(f"   Head 1: Q matrix {head_size}×{head_size}, K matrix {head_size}×{head_size}, V matrix {head_size}×{head_size}")
print(f"   Head 2: Q matrix {head_size}×{head_size}, K matrix {head_size}×{head_size}, V matrix {head_size}×{head_size}")
print()

# --- Split embeddings for each head ---

head1_embeddings = embeddings[:, :head_size]   # [3, 16] - dims 0-15
head2_embeddings = embeddings[:, head_size:]   # [3, 16] - dims 16-31

print("📥 Input to each head:")
print(f"   Head 1 input: {head1_embeddings.shape} (3 tokens × 16 dims)")
print(f"   Head 2 input: {head2_embeddings.shape} (3 tokens × 16 dims)")
print()

# --- Compute Q, K, V for each head ---

Q1 = head1_query(head1_embeddings)
K1 = head1_key(head1_embeddings)
V1 = head1_value(head1_embeddings)

Q2 = head2_query(head2_embeddings)
K2 = head2_key(head2_embeddings)
V2 = head2_value(head2_embeddings)

print("📤 Output Q, K, V (each is 3 tokens × 16 dims):")
print(f"   Head 1: Q1{list(Q1.shape)}, K1{list(K1.shape)}, V1{list(V1.shape)}")
print(f"   Head 2: Q2{list(Q2.shape)}, K2{list(K2.shape)}, V2{list(V2.shape)}")
```


**Output:**
```
======================================================================
STEP 4: EACH HEAD COMPUTES Q, K, V INDEPENDENTLY
======================================================================

🧠 What are Q, K, V?
   Q (Query):  'What am I looking for?'
   K (Key):    'What do I have to offer?'
   V (Value):  'What information should I share?'

📊 Each head has its own learnable matrices:
   Head 1: Q matrix 16×16, K matrix 16×16, V matrix 16×16
   Head 2: Q matrix 16×16, K matrix 16×16, V matrix 16×16

📥 Input to each head:
   Head 1 input: torch.Size([3, 16]) (3 tokens × 16 dims)
   Head 2 input: torch.Size([3, 16]) (3 tokens × 16 dims)

📤 Output Q, K, V (each is 3 tokens × 16 dims):
   Head 1: Q1[3, 16], K1[3, 16], V1[3, 16]
   Head 2: Q2[3, 16], K2[3, 16], V2[3, 16]

```


```python
# ============================================================================
# STEP 5: COMPUTE ATTENTION SCORES
# ============================================================================
# How much should each word look at others? This is the CORE of attention!

print("=" * 70)
print("STEP 5: COMPUTE ATTENTION SCORES")
print("=" * 70)
print()

print("🧮 The attention formula:")
print("   Attention Score = (Query × Key^T) / √(head_size)")
print()
print("   What this means:")
print("   • Each word's Query asks: 'What am I looking for?'")
print("   • Each word's Key answers: 'Here's what I have to offer'")
print("   • The score tells us how well they match!")
print()

# --- Compute attention scores ---

# Q (3×16) @ K.T (16×3) = (3×3) score matrix
attn_scores_head1 = (Q1 @ K1.T) / (head_size ** 0.5)
attn_scores_head2 = (Q2 @ K2.T) / (head_size ** 0.5)

# Convert scores to probabilities (0-1, sum to 1)
attn_weights_head1 = F.softmax(attn_scores_head1, dim=-1)
attn_weights_head2 = F.softmax(attn_scores_head2, dim=-1)

# --- Display attention weights ---

words = [word1, word2, word3]

print("=" * 70)
print("🔵 HEAD 1 ATTENTION WEIGHTS")
print("=" * 70)
print()
print("Read as: How much does each word (row) attend to other words (columns)?")
print()
print(f"                    Attends to:")
print(f"                    {word1:>10} {word2:>10} {word3:>10}")
for i, word in enumerate(words):
    weights = [f"{attn_weights_head1[i, j].item():.3f}" for j in range(3)]
    print(f"   '{word}'  →   {weights[0]:>10} {weights[1]:>10} {weights[2]:>10}")
print()

print("=" * 70)
print("🟢 HEAD 2 ATTENTION WEIGHTS")
print("=" * 70)
print()
print("Read as: How much does each word (row) attend to other words (columns)?")
print()
print(f"                    Attends to:")
print(f"                    {word1:>10} {word2:>10} {word3:>10}")
for i, word in enumerate(words):
    weights = [f"{attn_weights_head2[i, j].item():.3f}" for j in range(3)]
    print(f"   '{word}'  →   {weights[0]:>10} {weights[1]:>10} {weights[2]:>10}")
print()

# --- Key observation ---

print("=" * 70)
print("💡 KEY OBSERVATION:")
print("=" * 70)
print()
print("Notice how Head 1 and Head 2 have DIFFERENT attention patterns!")
print("This is because they have different learned weights.")
print("Each head learns to focus on different types of relationships.")
```


**Output:**
```
======================================================================
STEP 5: COMPUTE ATTENTION SCORES
======================================================================

🧮 The attention formula:
   Attention Score = (Query × Key^T) / √(head_size)

   What this means:
   • Each word's Query asks: 'What am I looking for?'
   • Each word's Key answers: 'Here's what I have to offer'
   • The score tells us how well they match!

======================================================================
🔵 HEAD 1 ATTENTION WEIGHTS
======================================================================

Read as: How much does each word (row) attend to other words (columns)?

                    Attends to:
                          mary        had          a
   'mary'  →        0.228      0.322      0.450
   'had'  →        0.404      0.254      0.341
   'a'  →        0.344      0.354      0.302

======================================================================
🟢 HEAD 2 ATTENTION WEIGHTS
======================================================================

Read as: How much does each word (row) attend to other words (columns)?

                    Attends to:
                          mary        had          a
   'mary'  →        0.241      0.415      0.344
   'had'  →        0.304      0.244      0.453
   'a'  →        0.526      0.241      0.233

======================================================================
💡 KEY OBSERVATION:
======================================================================

Notice how Head 1 and Head 2 have DIFFERENT attention patterns!
This is because they have different learned weights.
Each head learns to focus on different types of relationships.

```


```python
# ============================================================================
# STEP 6: APPLY ATTENTION AND CONCATENATE OUTPUTS
# ============================================================================
# Each head gathers information, then we combine the results!

print("=" * 70)
print("STEP 6: APPLY ATTENTION AND CONCATENATE OUTPUTS")
print("=" * 70)
print()

print("🧮 How attention works:")
print("   Output = Attention_Weights × Values")
print()
print("   Each word's output is a weighted average of ALL words' values,")
print("   weighted by how much attention it pays to each word!")
print()

# --- Apply attention weights to values ---

# Attention weights (3×3) @ Values (3×16) = Output (3×16)
output_head1 = attn_weights_head1 @ V1
output_head2 = attn_weights_head2 @ V2

print("📊 Each head produces a 16-dimensional output per token:")
print(f"   Head 1 output: {list(output_head1.shape)} (3 tokens × 16 dims)")
print(f"   Head 2 output: {list(output_head2.shape)} (3 tokens × 16 dims)")
print()

# --- Concatenate the two heads back together ---

print("=" * 70)
print("CONCATENATING HEADS BACK TOGETHER")
print("=" * 70)
print()

combined_output = torch.cat([output_head1, output_head2], dim=-1)  # [3, 32]

print("   Head 1 output (16 dims) + Head 2 output (16 dims)")
print("                     ↓")
print("            Combined output (32 dims)")
print()
print(f"✨ FINAL COMBINED OUTPUT: {list(combined_output.shape)}")
print()
print("We started with 32 dimensions per token, and after all that")
print("processing through 2 attention heads, we're back to 32 dimensions!")
print()
print("But now each token's embedding has been ENRICHED with information")
print("from other tokens that it paid attention to! 🎓")
```


**Output:**
```
======================================================================
STEP 6: APPLY ATTENTION AND CONCATENATE OUTPUTS
======================================================================

🧮 How attention works:
   Output = Attention_Weights × Values

   Each word's output is a weighted average of ALL words' values,
   weighted by how much attention it pays to each word!

📊 Each head produces a 16-dimensional output per token:
   Head 1 output: [3, 16] (3 tokens × 16 dims)
   Head 2 output: [3, 16] (3 tokens × 16 dims)

======================================================================
CONCATENATING HEADS BACK TOGETHER
======================================================================

   Head 1 output (16 dims) + Head 2 output (16 dims)
                     ↓
            Combined output (32 dims)

✨ FINAL COMBINED OUTPUT: [3, 32]

We started with 32 dimensions per token, and after all that
processing through 2 attention heads, we're back to 32 dimensions!

But now each token's embedding has been ENRICHED with information
from other tokens that it paid attention to! 🎓

```


### 3.3.2 Visual Summary: What Just Happened?

Let's recap what happened with our tokens "mary", "had", "a":

```
┌────────────────────────────────────────────────────────────────────┐
│                    MULTI-HEAD ATTENTION SUMMARY                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  INPUT: 3 tokens, each with 32 numbers                             │
│  ════════════════════════════════════════                          │
│  "mary" → [32 numbers]                                             │
│  "had"  → [32 numbers]                                             │
│  "a"    → [32 numbers]                                             │
│                     │                                              │
│                     ▼ SPLIT                                        │
│           ┌─────────┴─────────┐                                    │
│           ▼                   ▼                                    │
│     ┌───────────┐       ┌───────────┐                              │
│     │  HEAD 1   │       │  HEAD 2   │                              │
│     │(16 nums)  │       │(16 nums)  │                              │
│     ├───────────┤       ├───────────┤                              │
│     │ Q, K, V   │       │ Q, K, V   │                              │
│     │ Attention │       │ Attention │                              │
│     │ Scores    │       │ Scores    │                              │
│     └─────┬─────┘       └─────┬─────┘                              │
│           │                   │                                    │
│           ▼                   ▼                                    │
│     [16 numbers]        [16 numbers]                               │
│           │                   │                                    │
│           └─────────┬─────────┘                                    │
│                     ▼ CONCATENATE                                  │
│                                                                    │
│  OUTPUT: 3 tokens, each STILL with 32 numbers                      │
│  ════════════════════════════════════════                          │
│  "mary" → [32 numbers] (NOW ENRICHED!)                             │
│  "had"  → [32 numbers] (NOW ENRICHED!)                             │
│  "a"    → [32 numbers] (NOW ENRICHED!)                             │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**🎓 The Magic:**
- Each token's new 32-number embedding now contains information from ALL other tokens!
- The attention weights determined how much information to gather from each word
- Head 1 and Head 2 each looked for different patterns
- We combined their findings for a richer representation!


### 3.4 Understanding Transformer Layers (Stacking Blocks of Intelligence!)

One attention step isn't enough. We **stack multiple layers** for deeper understanding!

---

## 🏗️ What is a Transformer Layer?

A **transformer layer** (also called a "block") has TWO parts:

| Part | What It Does |
|------|-------------|
| 1. **Multi-Head Attention** | Words look at each other and share info |
| 2. **Feed-Forward Network** | Each word "thinks" independently |

---

## 📚 Why Stack Multiple Layers?

Think of it like reading comprehension:

| Layer | What It Learns | Analogy |
|-------|---------------|---------|
| Layer 1 | Basic patterns: "lamb often follows little" | Recognizing words |
| Layer 2 | Complex patterns: "mary → lamb" connection | Understanding sentences |
| Layer 3+ | Abstract patterns: grammar, context | Deep comprehension |

**More layers = deeper understanding!**

---

## 🔄 Our Model's Architecture (n_layers = 2)

```
Input Embeddings (32 numbers per token)
         │
         ▼
┌─────────────────────────────────────┐
│     TRANSFORMER BLOCK 1             │
│  ┌─────────────────────────────┐    │
│  │ Multi-Head Attention        │    │ ← Words share information
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ Feed-Forward Network        │    │ ← Each word "thinks"
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     TRANSFORMER BLOCK 2             │
│  ┌─────────────────────────────┐    │
│  │ Multi-Head Attention        │    │ ← Words share MORE info
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │ Feed-Forward Network        │    │ ← Each word "thinks" more
│  └─────────────────────────────┘    │
└─────────────────────────────────────┘
         │
         ▼
      Output → Predict next word!
```

---

## 📊 Real GPT Models

| Model | Number of Layers |
|-------|-----------------|
| Our TinyGPT | 2 layers |
| GPT-2 Small | 12 layers |
| GPT-2 XL | 48 layers |
| GPT-3 | 96 layers |

More layers = more capacity to learn = more computation needed!


```python
# ============================================================================
# COUNTING PARAMETERS: How Many Numbers Does Our Model Learn?
# ============================================================================
# Let's break down exactly where all the learnable parameters are!

print("=" * 70)
print("COUNTING PARAMETERS IN OUR MODEL")
print("=" * 70)
print()

# --- Embedding parameters ---

# Token embeddings: one row of 32 numbers for each word in vocabulary
token_emb_params = vocab_size * embedding_dim

# Position embeddings: one row of 32 numbers for each position (0-5)
pos_emb_params = block_size * embedding_dim

# Total embedding parameters
total_emb_params = token_emb_params + pos_emb_params

print("📊 EMBEDDING LAYERS:")
print(f"   Token embeddings:    {vocab_size} words × {embedding_dim} dims = {token_emb_params:,} parameters")
print(f"   Position embeddings: {block_size} positions × {embedding_dim} dims = {pos_emb_params:,} parameters")
print(f"   Total embeddings:    {total_emb_params:,} parameters")
print()

# --- Attention parameters ---

head_size = embedding_dim // n_heads  # 16

# Q, K, V matrices: 32 → 16 each
params_per_head = 3 * (embedding_dim * head_size)

# Output projection: combines heads back to 32 dims
output_proj_params = embedding_dim * embedding_dim

# Total attention per layer
attention_params_per_layer = (params_per_head * n_heads) + output_proj_params

print("🔵 ATTENTION PARAMETERS (per layer):")
print(f"   Each head has Q, K, V matrices:")
print(f"     Q: {embedding_dim} × {head_size} = {embedding_dim * head_size} params")
print(f"     K: {embedding_dim} × {head_size} = {embedding_dim * head_size} params")
print(f"     V: {embedding_dim} × {head_size} = {embedding_dim * head_size} params")
print(f"   Per head: {params_per_head} params × {n_heads} heads = {params_per_head * n_heads} params")
print(f"   Output projection: {embedding_dim} × {embedding_dim} = {output_proj_params} params")
print(f"   Total attention per layer: ~{attention_params_per_layer:,} parameters")
print()

# --- Feed-forward parameters ---

ff_hidden = 4 * embedding_dim  # 128 hidden neurons
ff_params_per_layer = (embedding_dim * ff_hidden) + (ff_hidden * embedding_dim)

print("🟢 FEED-FORWARD PARAMETERS (per layer):")
print(f"   Layer 1: {embedding_dim} → {ff_hidden} = {embedding_dim * ff_hidden} params")
print(f"   Layer 2: {ff_hidden} → {embedding_dim} = {ff_hidden * embedding_dim} params")
print(f"   Total feed-forward per layer: {ff_params_per_layer:,} parameters")
print()

# --- Total parameters ---

params_per_layer = attention_params_per_layer + ff_params_per_layer
total_transformer_params = params_per_layer * n_layers
total_params = total_emb_params + total_transformer_params

print("=" * 70)
print("📦 TOTAL PARAMETER COUNT")
print("=" * 70)
print()
print(f"   Embeddings:         {total_emb_params:,} parameters")
print(f"   Transformer blocks: {n_layers} layers × ~{params_per_layer:,} = ~{total_transformer_params:,} parameters")
print(f"   ─────────────────────────────────────")
print(f"   TOTAL:              ~{total_params:,} parameters")
print()

print("💡 KEY INSIGHT:")
print(f"   • Doubling n_layers → ~2× more transformer parameters")
print(f"   • Doubling n_heads → SAME parameters, but more diverse attention patterns")
```


**Output:**
```
======================================================================
COUNTING PARAMETERS IN OUR MODEL
======================================================================

📊 EMBEDDING LAYERS:
   Token embeddings:    35 words × 32 dims = 1,120 parameters
   Position embeddings: 6 positions × 32 dims = 192 parameters
   Total embeddings:    1,312 parameters

🔵 ATTENTION PARAMETERS (per layer):
   Each head has Q, K, V matrices:
     Q: 32 × 16 = 512 params
     K: 32 × 16 = 512 params
     V: 32 × 16 = 512 params
   Per head: 1536 params × 2 heads = 3072 params
   Output projection: 32 × 32 = 1024 params
   Total attention per layer: ~4,096 parameters

🟢 FEED-FORWARD PARAMETERS (per layer):
   Layer 1: 32 → 128 = 4096 params
   Layer 2: 128 → 32 = 4096 params
   Total feed-forward per layer: 8,192 parameters

======================================================================
📦 TOTAL PARAMETER COUNT
======================================================================

   Embeddings:         1,312 parameters
   Transformer blocks: 2 layers × ~12,288 = ~24,576 parameters
   ─────────────────────────────────────
   TOTAL:              ~25,888 parameters

💡 KEY INSIGHT:
   • Doubling n_layers → ~2× more transformer parameters
   • Doubling n_heads → SAME parameters, but more diverse attention patterns

```


### 3.5 The Complete Architecture: All the Pieces Together!

Now let's see how everything fits together into one complete model!

---

## 🏛️ TinyGPT Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                         TinyGPT ARCHITECTURE                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT: "mary had a little lamb <END>"                               │
│         │                                                            │
│         ▼                                                            │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │ 📖 TOKEN EMBEDDING TABLE                                   │      │
│  │ [vocab_size × embedding_dim] = [27 × 32]                   │      │
│  │ Each word → 32 numbers                                     │      │
│  └────────────────────────────────────────────────────────────┘      │
│         │                                                            │
│         ▼ (+ added together)                                         │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │ 📍 POSITION EMBEDDING TABLE                                │      │
│  │ [block_size × embedding_dim] = [6 × 32]                    │      │
│  │ Each position (0,1,2,3,4,5) → 32 numbers                   │      │
│  └────────────────────────────────────────────────────────────┘      │
│         │                                                            │
│         ▼                                                            │
│  ╔════════════════════════════════════════════════════════════╗      │
│  ║ 🔷 TRANSFORMER BLOCK 1                                     ║      │
│  ║  ┌──────────────────────────────────────────────────────┐  ║      │
│  ║  │ Multi-Head Attention (2 heads)                       │  ║      │
│  ║  │ Head 1 (16 dims) + Head 2 (16 dims) → 32 dims        │  ║      │
│  ║  └──────────────────────────────────────────────────────┘  ║      │
│  ║  ┌──────────────────────────────────────────────────────┐  ║      │
│  ║  │ Feed-Forward Network                                 │  ║      │
│  ║  │ 32 → 128 → 32 (expand then contract)                 │  ║      │
│  ║  └──────────────────────────────────────────────────────┘  ║      │
│  ╚════════════════════════════════════════════════════════════╝      │
│         │                                                            │
│         ▼                                                            │
│  ╔════════════════════════════════════════════════════════════╗      │
│  ║ 🔷 TRANSFORMER BLOCK 2 (same structure)                    ║      │
│  ╚════════════════════════════════════════════════════════════╝      │
│         │                                                            │
│         ▼                                                            │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │ 📊 OUTPUT HEAD                                             │      │
│  │ [embedding_dim → vocab_size] = [32 → 27]                   │      │
│  │ Converts 32 numbers → 27 probabilities (one per word)      │      │
│  └────────────────────────────────────────────────────────────┘      │
│         │                                                            │
│         ▼                                                            │
│  OUTPUT: "The next word is probably 'lamb' (85% confidence)"         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Quick Reference: Our Hyperparameters

| Parameter | Value | What It Controls |
|-----------|-------|-----------------|
| `vocab_size` | 27 | Number of unique words we can predict |
| `embedding_dim` | 32 | Size of each word's number-vector |
| `n_heads` | 2 | Parallel attention patterns |
| `n_layers` | 2 | Stacked transformer blocks |
| `block_size` | 6 | How many words of context at once |


## Part 4: Data Loading - Creating Training Batches

Now we need to prepare our data for training!

---

## 🎯 The Training Task

GPT learns by predicting the **next word**. We show it:
- **Input**: A sequence of words
- **Target**: The same sequence, but shifted by 1 (the "next words")

---

## 📝 Example

If our sentence is: `"mary had a little lamb <END>"`

| Position | Input (what model sees) | Target (what it should predict) |
|----------|------------------------|--------------------------------|
| 0 | "mary" | "had" |
| 1 | "had" | "a" |
| 2 | "a" | "little" |
| 3 | "little" | "lamb" |
| 4 | "lamb" | "<END>" |

At each position, the model tries to guess the next word!

---

## 📦 Batches

We don't train on just one sequence at a time - that's slow!

Instead, we create **batches** - multiple sequences processed together.

| Batch Item | Input (6 words) | Target (6 words) |
|------------|-----------------|------------------|
| Sequence 1 | "mary had a little lamb <END>" | "had a little lamb <END> little" |
| Sequence 2 | "lamb little lamb mary had a" | "little lamb mary had a little" |
| ... | ... | ... |


```python
# ============================================================================
# FUNCTION TO CREATE TRAINING BATCHES
# ============================================================================
# This function grabs random chunks of our text for training.

def get_batch(batch_size=16):
    """
    Create a random batch of training examples.
    
    Each example is a sequence of tokens and its targets
    (the same sequence shifted by 1 position).
    
    Args:
        batch_size: Number of sequences per batch.
    
    Returns:
        x: Input sequences [batch_size, block_size]
        y: Target sequences [batch_size, block_size]
    """
    # Pick random starting positions
    max_start = len(data) - block_size - 1
    ix = torch.randint(max_start, (batch_size,))
    
    # Create input sequences
    x = torch.stack([data[i:i+block_size] for i in ix])
    
    # Create target sequences (shifted by 1)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x, y

# --- Example batch ---

print("=" * 70)
print("EXAMPLE: A Training Batch")
print("=" * 70)
print()

# Create a tiny batch of 2 sequences
x_example, y_example = get_batch(batch_size=2)

print(f"📊 Batch shape:")
print(f"   x (input):  {list(x_example.shape)} ({x_example.shape[0]} sequences × {x_example.shape[1]} tokens each)")
print(f"   y (target): {list(y_example.shape)} ({y_example.shape[0]} sequences × {y_example.shape[1]} tokens each)")
print()

# --- Show the first sequence ---

print("=" * 70)
print("SEQUENCE 1:")
print("=" * 70)
print()
print("Input tokens (what the model sees):")
print(f"   Token IDs: {x_example[0].tolist()}")
print(f"   Words:     {' '.join([idx2word[int(i)] for i in x_example[0]])}")
print()
print("Target tokens (what the model should predict):")
print(f"   Token IDs: {y_example[0].tolist()}")
print(f"   Words:     {' '.join([idx2word[int(i)] for i in y_example[0]])}")
print()

print("💡 Notice: The target is the input shifted by 1!")
print("   At each position, y tells us what the NEXT word should be.")
```


**Output:**
```
======================================================================
EXAMPLE: A Training Batch
======================================================================

📊 Batch shape:
   x (input):  [2, 6] (2 sequences × 6 tokens each)
   y (target): [2, 6] (2 sequences × 6 tokens each)

======================================================================
SEQUENCE 1:
======================================================================

Input tokens (what the model sees):
   Token IDs: [2, 32, 22, 6, 32, 22]
   Words:     that mary went <END> mary went

Target tokens (what the model should predict):
   Token IDs: [32, 22, 6, 32, 22, 32]
   Words:     mary went <END> mary went mary

💡 Notice: The target is the input shifted by 1!
   At each position, y tells us what the NEXT word should be.

```


## Part 5: Self-Attention - The Heart of Transformers ❤️

Now we build the core component of GPT: **Self-Attention**!

---

## 🤔 What is Self-Attention?

Self-attention lets each word **look at all other words** and decide which ones are most important for predicting the next word.

---

## 🔑 The Three Magic Components: Q, K, V

Every word computes three things:

| Component | What It Means | Analogy |
|-----------|--------------|---------|
| **Q (Query)** | "What am I looking for?" | A question you ask |
| **K (Key)** | "What do I contain?" | A label describing what you have |
| **V (Value)** | "What info should I share?" | The actual information |

---

## 🎯 How Attention Works

For each word, we:
1. Compute its Query
2. Compare Query to all Keys (get attention scores)
3. Use scores to create a weighted average of all Values

```
                     "a"   is looking at other words
                      │
                      ▼
          Q("a") × K("mary") = score for "mary"
          Q("a") × K("had")  = score for "had"
          Q("a") × K("a")    = score for "a" (itself)
                      │
                      ▼
          Apply softmax → [0.3, 0.5, 0.2]  (sum to 1)
                      │
                      ▼
          Output = 0.3×V("mary") + 0.5×V("had") + 0.2×V("a")
```

The word "a" now contains information from "mary", "had", and itself!


```python
# ============================================================================
# SELF-ATTENTION HEAD: The Core of How Words Communicate!
# ============================================================================
# This is the most important class - it's how words share information.

class SelfAttentionHead(nn.Module):
    """
    A single head of self-attention.
    
    This lets each word look at other words and decide which are important.
    """
    
    def __init__(self, embedding_dim, block_size, head_size):
        super().__init__()
        
        # Q, K, V projection layers
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        
        # Causal mask: prevents looking at future words
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        
        # Compute Key and Query
        k = self.key(x)
        q = self.query(x)
        
        # Compute attention scores
        wei = q @ k.transpose(-2, -1) / (C ** 0.5)
        
        # Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Convert to probabilities
        wei = F.softmax(wei, dim=-1)
        
        # Apply attention to values
        v = self.value(x)
        out = wei @ v
        
        return out

# --- Test the attention head ---

print("=" * 70)
print("TESTING SELF-ATTENTION HEAD")
print("=" * 70)
print()

test_head = SelfAttentionHead(embedding_dim, block_size, head_size=16)
test_input = torch.randn(2, block_size, embedding_dim)
test_output = test_head(test_input)

print(f"📥 Input shape:  {list(test_input.shape)}")
print(f"   (2 sequences, {block_size} tokens, {embedding_dim} dims)")
print()
print(f"📤 Output shape: {list(test_output.shape)}")
print(f"   (2 sequences, {block_size} tokens, 16 dims)")
print()
print("✅ The attention head transformed each token while letting them")
print("   share information with each other!")
```


**Output:**
```
======================================================================
TESTING SELF-ATTENTION HEAD
======================================================================

📥 Input shape:  [2, 6, 32]
   (2 sequences, 6 tokens, 32 dims)

📤 Output shape: [2, 6, 16]
   (2 sequences, 6 tokens, 16 dims)

✅ The attention head transformed each token while letting them
   share information with each other!

```


### 🎭 The Causal Mask: No Cheating!

There's a SUPER important rule in GPT: **you can't look at the future!**

When predicting the next word after "mary had a", we can't peek at what comes later.

The **causal mask** enforces this rule!


```python
# ============================================================================
# VISUALIZING THE CAUSAL MASK
# ============================================================================
# The mask is a triangular matrix that controls what each word can see.

print("=" * 70)
print("THE CAUSAL MASK (No Peeking at the Future!)")
print("=" * 70)
print()

# --- Create the mask ---

mask = torch.tril(torch.ones(block_size, block_size))

print("Causal Mask (1 = CAN attend, 0 = CANNOT attend):")
print()
print(mask)
print()

# --- Interpret the mask ---

print("=" * 70)
print("INTERPRETATION:")
print("=" * 70)
print()
print("Each ROW represents a word position.")
print("1s show which words that position CAN look at.")
print()
print("Position 0 (word 1): [1 0 0 0 0 0] → Can only see itself")
print("Position 1 (word 2): [1 1 0 0 0 0] → Can see words 1-2")
print("Position 2 (word 3): [1 1 1 0 0 0] → Can see words 1-3")
print("Position 3 (word 4): [1 1 1 1 0 0] → Can see words 1-4")
print("Position 4 (word 5): [1 1 1 1 1 0] → Can see words 1-5")
print("Position 5 (word 6): [1 1 1 1 1 1] → Can see ALL words 1-6")
print()
print("💡 This is why GPT predicts LEFT-TO-RIGHT!")
print("   Each word only uses information from previous words.")
```


**Output:**
```
======================================================================
THE CAUSAL MASK (No Peeking at the Future!)
======================================================================

Causal Mask (1 = CAN attend, 0 = CANNOT attend):

tensor([[1., 0., 0., 0., 0., 0.],
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 1., 0., 0.],
        [1., 1., 1., 1., 1., 0.],
        [1., 1., 1., 1., 1., 1.]])

======================================================================
INTERPRETATION:
======================================================================

Each ROW represents a word position.
1s show which words that position CAN look at.

Position 0 (word 1): [1 0 0 0 0 0] → Can only see itself
Position 1 (word 2): [1 1 0 0 0 0] → Can see words 1-2
Position 2 (word 3): [1 1 1 0 0 0] → Can see words 1-3
Position 3 (word 4): [1 1 1 1 0 0] → Can see words 1-4
Position 4 (word 5): [1 1 1 1 1 0] → Can see words 1-5
Position 5 (word 6): [1 1 1 1 1 1] → Can see ALL words 1-6

💡 This is why GPT predicts LEFT-TO-RIGHT!
   Each word only uses information from previous words.

```


## Part 6: Multi-Head Attention (Multiple Perspectives!)

One attention head is good, but **multiple heads are better!**

---

## 🤔 Why Multiple Heads?

Different heads can learn to focus on different things:

| Head | What It Might Learn |
|------|---------------------|
| Head 1 | Subject-verb relationships ("mary" → "went") |
| Head 2 | Adjective-noun relationships ("little" → "lamb") |
| Head 3 | Positional patterns (nearby words) |

---

## 🔧 How It Works

1. **Split** the embedding into chunks (one per head)
2. **Run** each head independently  
3. **Concatenate** all outputs back together

```
   Input: 32 dimensions
           ↓
   ┌───────┴───────┐
   ▼               ▼
┌──────┐       ┌──────┐
│Head 1│       │Head 2│
│16 dim│       │16 dim│
└──┬───┘       └──┬───┘
   ▼               ▼
   16 dims      16 dims
   └───────┬───────┘
           ▼
   Concatenate → 32 dimensions
```


```python
# ============================================================================
# MULTI-HEAD ATTENTION: Multiple Attention Heads Working Together
# ============================================================================
# Multiple attention heads let the model look at relationships in different ways.

class MultiHeadAttention(nn.Module):
    """Multiple attention heads running in parallel."""
    
    def __init__(self, embedding_dim, block_size, num_heads):
        super().__init__()
        
        head_size = embedding_dim // num_heads
        
        # Create multiple attention heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(embedding_dim, block_size, head_size) 
            for _ in range(num_heads)
        ])
        
        # Output projection
        self.proj = nn.Linear(num_heads * head_size, embedding_dim)
    
    def forward(self, x):
        # Run all heads and concatenate
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(out)

# --- Test multi-head attention ---

print("=" * 70)
print("TESTING MULTI-HEAD ATTENTION")
print("=" * 70)
print()

test_mha = MultiHeadAttention(embedding_dim, block_size, n_heads)
test_output = test_mha(test_input)

print(f"📥 Input shape:  {list(test_input.shape)}")
print(f"📤 Output shape: {list(test_output.shape)}")
print()
print(f"💡 With {n_heads} heads, each head processes {embedding_dim // n_heads} dimensions")
print("   All heads' outputs are concatenated and projected back to 32 dims")
```


**Output:**
```
======================================================================
TESTING MULTI-HEAD ATTENTION
======================================================================

📥 Input shape:  [2, 6, 32]
📤 Output shape: [2, 6, 32]

💡 With 2 heads, each head processes 16 dimensions
   All heads' outputs are concatenated and projected back to 32 dims

```


## Part 7: Feed-Forward Network (Independent Thinking!)

After words share information through attention, each word needs to **"think"** about what it learned.

---

## 🧠 What is the Feed-Forward Network?

It's a simple neural network (just like our SimpleNetwork!) that processes each word INDEPENDENTLY.

**Structure:**
1. **Expand**: 32 dims → 128 dims (more room to think!)
2. **ReLU**: Add non-linearity
3. **Compress**: 128 dims → 32 dims (back to original)

```
Token embedding
(32 numbers)
     │
     ▼
┌────────────────┐
│   Linear       │
│  32 → 128      │ ← Expand
└────────────────┘
     │
     ▼
┌────────────────┐
│     ReLU       │ ← Add curves
└────────────────┘
     │
     ▼
┌────────────────┐
│   Linear       │
│  128 → 32      │ ← Compress back
└────────────────┘
     │
     ▼
Token embedding
(32 numbers, transformed!)
```

**Key insight**: The same feed-forward network is applied to EVERY token separately!


```python
# ============================================================================
# FEED-FORWARD NETWORK: Independent Thinking for Each Word
# ============================================================================
# A simple 2-layer network applied to each token independently.

class FeedForward(nn.Module):
    """Feed-forward network: 32 → 128 → 32 with ReLU."""
    
    def __init__(self, n_embd):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),   # Expand: 32 → 128
            nn.ReLU(),                        # Activation
            nn.Linear(4 * n_embd, n_embd)    # Compress: 128 → 32
        )
    
    def forward(self, x):
        return self.net(x)

# --- Test feed-forward network ---

print("=" * 70)
print("TESTING FEED-FORWARD NETWORK")
print("=" * 70)
print()

test_ff = FeedForward(embedding_dim)
test_output = test_ff(test_input)

print(f"📥 Input shape:  {list(test_input.shape)}")
print(f"📤 Output shape: {list(test_output.shape)}")
print()
print(f"💡 Inside the network:")
print(f"   32 dims → 128 dims (expand) → ReLU → 32 dims (compress)")
print(f"   Intermediate dimension: {4 * embedding_dim}")
```


**Output:**
```
======================================================================
TESTING FEED-FORWARD NETWORK
======================================================================

📥 Input shape:  [2, 6, 32]
📤 Output shape: [2, 6, 32]

💡 Inside the network:
   32 dims → 128 dims (expand) → ReLU → 32 dims (compress)
   Intermediate dimension: 128

```


## Part 8: Transformer Block (Putting Attention + FeedForward Together!)

Now we combine multi-head attention and feed-forward into a single **Transformer Block**.

---

## 🧱 What's in a Block?

```
          INPUT (32 dims per word)
             │
             ▼
     ┌───────────────┐
     │  Layer Norm   │  ← Normalize for stability
     └───────┬───────┘
             │
             ▼
     ┌───────────────┐
     │  Multi-Head   │  ← Words share information
     │   Attention   │
     └───────┬───────┘
             │
     ┌───────┴───────┐
     │    + INPUT    │  ← Residual connection (add original)
     └───────┬───────┘
             │
             ▼
     ┌───────────────┐
     │  Layer Norm   │  ← Normalize again
     └───────┬───────┘
             │
             ▼
     ┌───────────────┐
     │ Feed-Forward  │  ← Each word "thinks"
     │   Network     │
     └───────┬───────┘
             │
     ┌───────┴───────┐
     │    + INPUT    │  ← Residual connection again
     └───────┬───────┘
             │
             ▼
          OUTPUT (32 dims per word)
```

---

## 🔗 What's a "Residual Connection"?

It's just adding the input back to the output: `output = input + layer(input)`

**Why?** It helps gradients flow during training. Without residuals, deep networks are hard to train!


```python
# ============================================================================
# TRANSFORMER BLOCK: Combines Attention + Feed-Forward
# ============================================================================
# This is the basic building block of GPT!

class Block(nn.Module):
    """
    A single Transformer block.
    
    Components:
        1. Multi-head attention: words share information
        2. Feed-forward network: each word processes info
        3. Layer normalization: keeps numbers stable
        4. Residual connections: helps training
    """
    
    def __init__(self, embedding_dim, block_size, n_heads):
        super().__init__()
        
        # Attention and feed-forward layers
        self.sa = MultiHeadAttention(embedding_dim, block_size, n_heads)
        self.ffwd = FeedForward(embedding_dim)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        # Attention with residual connection
        x = x + self.sa(self.ln1(x))
        
        # Feed-forward with residual connection
        x = x + self.ffwd(self.ln2(x))
        
        return x

# --- Test the transformer block ---

print("=" * 60)
print("TESTING THE TRANSFORMER BLOCK")
print("=" * 60)

test_block = Block(embedding_dim, block_size, n_heads)
test_output = test_block(test_input)

print(f"\n📥 Input shape:  {test_input.shape}")
print(f"📤 Output shape: {test_output.shape}")
print("\n✅ The Transformer block preserves the shape!")
print("   - Same batch size (2)")
print("   - Same sequence length (6 words)")
print("   - Same embedding dimension (32 features per word)")
print("\nBut the OUTPUT contains richer information because:")
print("   1. Each word has 'heard' from previous words (attention)")
print("   2. Each word has 'thought' about the information (feed-forward)")
print("=" * 60)
```


**Output:**
```
============================================================
TESTING THE TRANSFORMER BLOCK
============================================================

📥 Input shape:  torch.Size([2, 6, 32])
📤 Output shape: torch.Size([2, 6, 32])

✅ The Transformer block preserves the shape!
   - Same batch size (2)
   - Same sequence length (6 words)
   - Same embedding dimension (32 features per word)

But the OUTPUT contains richer information because:
   1. Each word has 'heard' from previous words (attention)
   2. Each word has 'thought' about the information (feed-forward)
============================================================

```


## Part 9: The Complete GPT Model 🎉

Now we put EVERYTHING together into a complete GPT model!

---

## 🏗️ Full Architecture Overview

```
     Token IDs (e.g., [3, 1, 17, 23, 5, 0])
              │
              ▼
    ┌─────────────────┐
    │ Token Embedding │  ← Look up 32 numbers per word
    └────────┬────────┘
              │
              + (add together)
              │
    ┌─────────────────┐
    │Position Embedding│  ← Look up 32 numbers per position
    └────────┬────────┘
              │
              ▼
    ┌─────────────────┐
    │Transformer Block 1│  ← Attention + Feed-Forward
    └────────┬────────┘
              │
              ▼
    ┌─────────────────┐
    │Transformer Block 2│  ← Attention + Feed-Forward again!
    └────────┬────────┘
              │
              ▼
    ┌─────────────────┐
    │   Layer Norm    │  ← Final normalization
    └────────┬────────┘
              │
              ▼
    ┌─────────────────┐
    │ Output Linear   │  ← Convert 32 numbers → 27 numbers
    └────────┬────────┘   (one score per possible next word)
              │
              ▼
        LOGITS (27 scores)
        "Which word comes next?"
```

---

## 🧮 What Are Logits?

**Logits** are the raw scores the model outputs for each possible next word.

Example for a 5-word vocabulary:
```
logits = [2.3, -0.5, 1.1, 4.2, 0.8]
          │     │     │     │     │
          a     b     c     d     e

The model thinks "d" is most likely (highest score: 4.2)!
```

We convert logits to probabilities using **softmax** during training.


```python
# ============================================================================
# TINYGPT: THE COMPLETE GPT MODEL
# ============================================================================
# This is where everything comes together!
# Architecture: Token Embed → Position Embed → Transformer Blocks → Output

class TinyGPT(nn.Module):
    """
    A tiny GPT model that can learn to predict the next word.
    
    Architecture:
        1. Token Embedding: word IDs → 32-dim vectors
        2. Position Embedding: positions → 32-dim vectors  
        3. Transformer Blocks: attention + feed-forward
        4. Output Head: 32 dims → vocabulary scores
    """
    
    def __init__(self):
        super().__init__()
        
        # Token embedding: word ID → 32 numbers
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Position embedding: position → 32 numbers
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        
        # Stack of transformer blocks
        self.blocks = nn.Sequential(
            *[Block(embedding_dim, block_size, n_heads) for _ in range(n_layers)]
        )
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(embedding_dim)
        
        # Output projection: 32 → vocab_size
        self.head = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, idx, targets=None):
        """
        Forward pass: word IDs → prediction scores.
        
        Args:
            idx: Word IDs [Batch, Sequence]
            targets: Correct next words (for training)
        
        Returns:
            logits: Prediction scores [Batch, Sequence, vocab_size]
            loss: How wrong we are (only if targets provided)
        """
        B, T = idx.shape
        
        # Step 1: Get token embeddings
        tok_emb = self.token_embedding(idx)
        
        # Step 2: Get position embeddings
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        
        # Step 3: Combine embeddings
        x = tok_emb + pos_emb
        
        # Step 4: Pass through transformer blocks
        x = self.blocks(x)
        
        # Step 5: Final normalization
        x = self.ln_f(x)
        
        # Step 6: Get prediction scores
        logits = self.head(x)
        
        # Step 7: Calculate loss if training
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """
        Generate new words one at a time.
        
        Args:
            idx: Starting word IDs [Batch, Sequence]
            max_new_tokens: How many new words to generate
        
        Returns:
            Extended sequence with generated words
        """
        for _ in range(max_new_tokens):
            # Limit context to block_size
            idx_cond = idx[:, -block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Focus on last position
            logits = logits[:, -1, :]
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample next word
            next_idx = torch.multinomial(probs, num_samples=1)
            
            # Add to sequence
            idx = torch.cat((idx, next_idx), dim=1)
        
        return idx

# --- Create the model ---

print("=" * 60)
print("CREATING TINYGPT MODEL")
print("=" * 60)

model = TinyGPT()
num_params = sum(p.numel() for p in model.parameters())

print(f"\n✅ TinyGPT Model Created!")
print(f"📊 Total parameters: {num_params:,}")
print(f"\n🏗️ Model Architecture:")
print("-" * 40)
print(model)
print("-" * 40)
print("\nThis tiny model has the same architecture as ChatGPT,")
print("just with much smaller dimensions!")
print("=" * 60)
```


**Output:**
```
============================================================
CREATING TINYGPT MODEL
============================================================

✅ TinyGPT Model Created!
📊 Total parameters: 27,747

🏗️ Model Architecture:
----------------------------------------
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
----------------------------------------

This tiny model has the same architecture as ChatGPT,
just with much smaller dimensions!
============================================================

```


## Part 10: Training the Model 🏋️

Now we teach our GPT to predict words! Training is a loop:

---

## 🔄 The Training Loop (What Happens Each Step)

```
      ┌─────────────────────────────────────────────────┐
      │                 TRAINING LOOP                    │
      └─────────────────────────────────────────────────┘
                           │
                           ▼
      ┌──────────────────────────────────────────────────┐
      │ 1. GET A BATCH                                    │
      │    Pick random chunks from our training data      │
      │    Input:  "mary had a little lamb whose"        │
      │    Target: "had a little lamb whose fleece"      │
      └──────────────────────────────────────────────────┘
                           │
                           ▼
      ┌──────────────────────────────────────────────────┐
      │ 2. FORWARD PASS                                   │
      │    Run input through the model                    │
      │    Get predictions + calculate loss              │
      │    "How wrong are we?"                           │
      └──────────────────────────────────────────────────┘
                           │
                           ▼
      ┌──────────────────────────────────────────────────┐
      │ 3. BACKWARD PASS                                  │
      │    Calculate gradients using backpropagation      │
      │    "Which way should each weight move?"          │
      └──────────────────────────────────────────────────┘
                           │
                           ▼
      ┌──────────────────────────────────────────────────┐
      │ 4. UPDATE WEIGHTS                                 │
      │    Adjust weights in the direction that           │
      │    reduces the loss (gradient descent)            │
      └──────────────────────────────────────────────────┘
                           │
                           ▼
                   Repeat 1500 times!
```

---

## ⚙️ AdamW Optimizer

We use **AdamW**, the standard optimizer for transformers. It's smarter than basic gradient descent:
- **Momentum**: Keeps moving in the same direction (like a ball rolling downhill)
- **Adaptive learning rates**: Different learning rates for different parameters
- **Weight decay**: Prevents weights from getting too large (regularization)


```python
# ============================================================================
# TRAINING LOOP: TEACH THE MODEL!
# ============================================================================
# This is where learning happens! We repeat these steps 1500 times:
#   1. Get a batch of training examples
#   2. Make predictions (forward pass)
#   3. Calculate how wrong we are (loss)
#   4. Calculate gradients (backward pass)
#   5. Update weights (optimizer step)

# --- Create the optimizer ---

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

print("=" * 60)
print("🏋️ STARTING TRAINING")
print("=" * 60)
print(f"📊 Epochs (training steps): {epochs}")
print(f"📈 Learning rate: {lr}")
print(f"📦 Batch size: 4 sequences per step")
print(f"🔢 Sequence length: {block_size} words")
print("-" * 60)
print("Step | Loss (lower is better)")
print("-" * 60)

# --- The training loop ---

for step in range(epochs):
    # STEP 1: Get a batch of training data
    xb, yb = get_batch()
    
    # STEP 2: Forward pass - make predictions
    logits, loss = model(xb, yb)
    
    # STEP 3: Zero out old gradients
    optimizer.zero_grad()
    
    # STEP 4: Backward pass - calculate gradients
    loss.backward()
    
    # STEP 5: Update weights
    optimizer.step()
    
    # Print progress every 300 steps
    if step % 300 == 0:
        print(f"{step:4d}  | {loss.item():.4f}")

# --- Training complete ---

print("-" * 60)
print(f"✅ Training complete!")
print(f"📉 Final loss: {loss.item():.4f}")
print()
print("The model has now learned patterns from Mary Had a Little Lamb!")
print("Let's see what it can generate...")
print("=" * 60)
```


**Output:**
```
============================================================
🏋️ STARTING TRAINING
============================================================
📊 Epochs (training steps): 1500
📈 Learning rate: 0.001
📦 Batch size: 4 sequences per step
🔢 Sequence length: 6 words
------------------------------------------------------------
Step | Loss (lower is better)
------------------------------------------------------------
   0  | 3.7665
 300  | 0.4061
 600  | 0.3361
 900  | 0.2343
1200  | 0.2704
------------------------------------------------------------
✅ Training complete!
📉 Final loss: 0.2198

The model has now learned patterns from Mary Had a Little Lamb!
Let's see what it can generate...
============================================================

```


## Part 11: Generating Text! 🎨

Now the fun part - let's make our model generate nursery rhyme text!

---

## 🔮 How Text Generation Works

```
    Start: "mary"
       │
       ▼
    ┌─────────────────┐
    │     GPT Model   │ → Predicts "had" is most likely next
    └────────┬────────┘
             │
             ▼
    Sequence is now: "mary had"
       │
       ▼
    ┌─────────────────┐
    │     GPT Model   │ → Predicts "a" is most likely next
    └────────┬────────┘
             │
             ▼
    Sequence is now: "mary had a"
       │
       ▼
    ... and so on!
```

This is called **autoregressive generation** - each new word becomes part of the context for the next prediction!


```python
# ============================================================================
# TEXT GENERATION: LET'S SEE WHAT OUR MODEL LEARNED!
# ============================================================================
# We give the model a starting word and let it generate the rest!

# --- Set model to evaluation mode ---

model.eval()

# --- Choose a starting word ---

start_word = "mary"
print("=" * 60)
print("🎨 TEXT GENERATION")
print("=" * 60)
print(f"\n🚀 Starting word: '{start_word}'")
print("⏳ Generating text...\n")

# --- Prepare the input ---

context = torch.tensor([[word2idx[start_word]]], dtype=torch.long)

# --- Generate new tokens ---

with torch.no_grad():
    generated = model.generate(context, max_new_tokens=15)

# --- Convert back to words ---

generated_text = " ".join([idx2word[int(i)] for i in generated[0]])

# --- Print the result ---

print("=" * 60)
print("📝 GENERATED TEXT:")
print("=" * 60)
print()
print(f"   {generated_text}")
print()
print("=" * 60)
print("\n💡 Note: The model learned from 'Mary Had a Little Lamb'")
print("   so it should generate text that sounds like the rhyme!")
print("=" * 60)
```


**Output:**
```
============================================================
🎨 TEXT GENERATION
============================================================

🚀 Starting word: 'mary'
⏳ Generating text...

============================================================
📝 GENERATED TEXT:
============================================================

   mary went <END> the lamb was sure to go <END> it followed her to school one

============================================================

💡 Note: The model learned from 'Mary Had a Little Lamb'
   so it should generate text that sounds like the rhyme!
============================================================

```


```python
# ============================================================================
# BONUS: GENERATE FROM DIFFERENT STARTING WORDS
# ============================================================================
# Let's try starting with different words to see how the model generates
# different text based on its starting context!

def generate_from_word(start_word, max_tokens=12):
    """
    Generate text starting from a given word.
    
    Args:
        start_word: The word to start from (must be in vocabulary)
        max_tokens: How many new words to generate
    
    Returns:
        The generated text, or None if word not in vocabulary
    """
    # Check if the word exists
    if start_word not in word2idx:
        print(f"❌ '{start_word}' not in vocabulary!")
        print(f"📖 Available words: {list(word2idx.keys())}")
        return None
    
    # Convert to tensor
    context = torch.tensor([[word2idx[start_word]]], dtype=torch.long)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_tokens)
    
    # Convert to text
    text = " ".join([idx2word[int(i)] for i in generated[0]])
    return text

# --- Try different starting words ---

start_words = ["mary", "the", "it", "little", "lamb"]

print("=" * 60)
print("🎨 GENERATING FROM DIFFERENT STARTING WORDS")
print("=" * 60)
print()

for word in start_words:
    result = generate_from_word(word)
    if result:
        print(f"🚀 Starting with '{word}':")
        print(f"   → {result}")
        print()

print("=" * 60)
print("💡 Notice how the model learned patterns from the rhyme!")
print("   It predicts words that commonly follow each other.")
print("=" * 60)
```


**Output:**
```
============================================================
🎨 GENERATING FROM DIFFERENT STARTING WORDS
============================================================

🚀 Starting with 'mary':
   → mary went <END> mary went mary went <END> everywhere that mary went <END>

🚀 Starting with 'the':
   → the rules <END> it made the children laugh and play <END> laugh and

🚀 Starting with 'it':
   → it made the children laugh and play <END> to see a lamb at

🚀 Starting with 'little':
   → little lamb <END> mary had a little lamb <END> its fleece was white

🚀 Starting with 'lamb':
   → lamb <END> its fleece was white as snow <END> and everywhere that mary

============================================================
💡 Notice how the model learned patterns from the rhyme!
   It predicts words that commonly follow each other.
============================================================

```


## Part 12: Understanding What We Built 📚

### 🎉 Congratulations! You've Built a GPT from Scratch!

---

## 📋 Summary of Components

Here's everything we built and what each part does:

| Component | What It Does | Analogy |
|-----------|--------------|---------|
| **Token Embedding** | Converts word IDs to 32-number vectors | Dictionary: word → meaning |
| **Position Embedding** | Tells model where each word is located | Numbered seats in a theater |
| **Self-Attention** | Words look at previous words | Students asking classmates questions |
| **Multi-Head Attention** | Multiple perspectives on relationships | Multiple discussions at once |
| **Feed-Forward Network** | Each word "thinks" independently | Personal reflection time |
| **Residual Connections** | Adds input back to output | Remembering what you started with |
| **Layer Normalization** | Keeps numbers in reasonable range | Volume control |
| **Language Model Head** | Converts to word probabilities | Making a final prediction |

---

## 📊 Our Model vs. ChatGPT

| Feature | Our TinyGPT | GPT-3 | GPT-4 |
|---------|-------------|-------|-------|
| **Layers** | 2 | 96 | ~120 |
| **Embedding Size** | 32 | 12,288 | ~12,000 |
| **Attention Heads** | 2 | 96 | ~96 |
| **Vocabulary** | 27 words | 50,000 tokens | ~100,000 tokens |
| **Parameters** | ~20,000 | 175 Billion | 1.7 Trillion |
| **Training Data** | 1 nursery rhyme | 300 billion words | Trillions of words |

**Key Insight**: The ARCHITECTURE is the same! ChatGPT is just MUCH bigger.

---

## 🔑 Key Concepts You Learned

1. **Tokenization**: Converting text to numbers the model can process
2. **Embeddings**: Dense vector representations that capture meaning
3. **Self-Attention**: The mechanism that makes Transformers powerful
4. **Backpropagation**: How neural networks learn from mistakes
5. **Gradient Descent**: Moving weights in the direction that reduces loss
6. **Autoregressive Generation**: Predicting one word at a time

---

## 🚀 Next Steps to Improve This Model

Want to make it better? Try:
1. **More training data**: Use longer texts (books, articles)
2. **Bigger model**: More layers, larger embeddings
3. **Better tokenization**: Use BPE (byte-pair encoding) instead of words
4. **GPU training**: Much faster than CPU
5. **Learning rate scheduling**: Start fast, slow down later
6. **Dropout**: Randomly turn off neurons to prevent overfitting


```python
# ============================================================================
# FINAL MODEL STATISTICS
# ============================================================================
# Let's print a summary of everything we built!

print("=" * 60)
print("📊 FINAL MODEL STATISTICS")
print("=" * 60)
print()
print(f"📖 Vocabulary size:     {vocab_size} words")
print(f"📏 Context window:      {block_size} tokens (max sequence length)")
print(f"🔢 Embedding dimension: {embedding_dim} numbers per word")
print(f"👀 Attention heads:     {n_heads} (parallel attention patterns)")
print(f"📚 Transformer layers:  {n_layers} (stacked blocks)")
print(f"⚙️  Total parameters:   {sum(p.numel() for p in model.parameters()):,}")
print()
print("=" * 60)
print()
print("🎉 CONGRATULATIONS! 🎉")
print()
print("You've successfully built a GPT (Generative Pre-trained Transformer)")
print("completely from scratch! This is the same architecture that powers:")
print("  • ChatGPT")
print("  • GPT-4")
print("  • GitHub Copilot")
print("  • Many other AI assistants")
print()
print("The only difference is SIZE. Our model has ~20,000 parameters,")
print("while GPT-4 has over 1.7 TRILLION parameters!")
print()
print("But you now understand how it all works! 🧠")
print("=" * 60)
```


**Output:**
```
============================================================
📊 FINAL MODEL STATISTICS
============================================================

📖 Vocabulary size:     35 words
📏 Context window:      6 tokens (max sequence length)
🔢 Embedding dimension: 32 numbers per word
👀 Attention heads:     2 (parallel attention patterns)
📚 Transformer layers:  2 (stacked blocks)
⚙️  Total parameters:   27,747

============================================================

🎉 CONGRATULATIONS! 🎉

You've successfully built a GPT (Generative Pre-trained Transformer)
completely from scratch! This is the same architecture that powers:
  • ChatGPT
  • GPT-4
  • GitHub Copilot
  • Many other AI assistants

The only difference is SIZE. Our model has ~20,000 parameters,
while GPT-4 has over 1.7 TRILLION parameters!

But you now understand how it all works! 🧠
============================================================

```


## MIT License

