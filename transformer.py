import torch
import torch.nn as nn
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Available device : " , device)

class Embedding(nn.Module):
  def __init__(self, vocab_size, embed_dim):
    """
    Args:
        vocab_size (int): Size of the vocabulary (number of unique tokens)
        embed_dim (int): Dimension of the embedding vectors
    """
    super().__init__()

    # nn.Embedding maps token indices to dense embedding vectors
    # Shape: [vocab_size, embed_dim]
    self.embed = nn.Embedding(vocab_size, embed_dim)

  def forward(self, x):
    """
    Args:
        x (Tensor): Input tensor of token indices, shape [batch_size, seq_len]
    Returns:
        Tensor: Embedded vectors of shape [batch_size, seq_len, embed_dim]
    """
    # Convert token indices to embeddings
    out = self.embed(x)

    return out

class PositionalEmbedding(nn.Module):
  def __init__(self, max_seq_len, embed_dim):
    """
    Args:
        max_seq_len (int): The maximum length of input sequences
        embed_dim (int): The embedding dimension (model dimension)
    """
    super().__init__()
    self.embed_dim = embed_dim

    # Create a zero tensor for positional encodings
    # Shape: [max_seq_len, embed_dim]
    pe = torch.zeros(max_seq_len, embed_dim)

    # Generate sinusoidal positional encodings
    for pos in range(max_seq_len):
      for i in range(0, embed_dim, 2):  # even index (i), i+1 will be odd
        pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_dim)))
        if i + 1 < embed_dim:
          pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / embed_dim)))

    # Add batch dimension: [1, max_seq_len, embed_dim]
    pe = pe.unsqueeze(0)

    # Register as buffer (non-trainable, but moves with the model and is saved)
    self.register_buffer("pe", pe)

  def forward(self, x):
    """
    Args:
        x (Tensor): Token embeddings of shape [batch_size, seq_len, embed_dim]
    Returns:
        x (Tensor): Embeddings with positional encodings added
    """
    # Scale input embeddings by sqrt(embed_dim), as in the Transformer paper
    x = x * math.sqrt(self.embed_dim)

    # Get the sequence length from input
    seq_len = x.size(1)

    # Add positional encoding (broadcasted across batch)
    # Make sure pe is on the same device as x
    x = x + self.pe[:, :seq_len].to(x.device)

    return x


class MultiHeadAttention(nn.Module):
  def __init__(self, embed_dim=512, n_heads=8):
      """
      Multi-Head Attention initialization.

      Args:
          embed_dim (int): Total embedding dimension (d_model), e.g., 512.
          n_heads (int): Number of attention heads, e.g., 8.

      The embedding dimension per head will be:
          head_dim = embed_dim // n_heads
      """
      super().__init__()

      self.embed_dim = embed_dim              # Total embedding dimension (e.g., 512)
      self.n_heads = n_heads                  # Number of attention heads (e.g., 8)
      self.head_dim = embed_dim // n_heads   # Dimension per head (512/8 = 64)

      assert embed_dim % n_heads == 0, "Embedding dimension must be divisible by number of heads otherwise it generate an error AssertionError"

      # Linear layers to project input embeddings into queries, keys, and values
      # Each projects from embed_dim (512) to embed_dim (512) so we can split into heads later
      self.query_matrix = nn.Linear(embed_dim, embed_dim, bias=False)
      self.key_matrix = nn.Linear(embed_dim, embed_dim, bias=False)
      self.value_matrix = nn.Linear(embed_dim, embed_dim, bias=False)

      # Final linear layer to combine concatenated multi-head outputs back to embed_dim
      self.out = nn.Linear(embed_dim, embed_dim)

  def forward(self , query , key , value , mask):
    """
    Forward pass of Multi-Head Attention.

    Args:
        query: Tensor of shape (batch_size, query_len, embed_dim)
        key: Tensor of shape (batch_size, key_len, embed_dim)
        value: Tensor of shape (batch_size, value_len, embed_dim)
        mask: Optional attention mask (e.g., padding or look-ahead)

    Returns:
        output: Tensor of shape (batch_size, query_len, embed_dim)
    """

    batch_size = query.size(0)

    # 1. Apply linear layers to obtain Q, K, V
    Q = self.query_matrix(query)  # (batch_size, query_len, embed_dim)
    K = self.key_matrix(key)      # (batch_size, key_len, embed_dim)
    V = self.value_matrix(value)  # (batch_size, value_len, embed_dim)

    # 2. Reshape Q, K, V for multi-head attention
    #    New shape: (batch_size, n_heads, seq_len, head_dim)
    Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
    K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
    V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

    # 3. Calculate scaled dot-product attention scores
    #    Q ⋅ K^T → (batch_size, n_heads, query_len, key_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

    # 4. Apply mask if provided (e.g., to block padding or future tokens)
    if mask is not None:
      # mask shape should be broadcastable to scores shape
      scores = scores.masked_fill(mask == 0, float('-1e20'))

    # 5. Softmax over the last dimension (key length) to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # 6. Multiply attention weights with values → weighted sum
    #    (batch_size, n_heads, query_len, head_dim)
    attn_output = torch.matmul(attn_weights, V)

    # 7. Concatenate all heads
    #    First: (batch_size, query_len, n_heads, head_dim)
    #    Then: (batch_size, query_len, embed_dim)
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)

    # 8. Final linear projection to mix head information
    output = self.out(attn_output)  # (batch_size, query_len, embed_dim)

    return output


class TransformerBlock(nn.Module):
  def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
    """
    Transformer Encoder Block consisting of:
    - Multi-head self-attention
    - Layer normalization
    - Feed-forward network
    - Dropout and residual connections

    Args:
        embed_dim (int): Dimension of input embeddings (e.g., 512).
        expansion_factor (int): Controls the size of the hidden layer in FFN.
                                Output size = expansion_factor * embed_dim.
        n_heads (int): Number of attention heads in multi-head attention.
    """
    super().__init__()

    # Multi-Head Self-Attention Layer
    self.attention = MultiHeadAttention(embed_dim, n_heads)

    # Layer Normalization (applied after attention and FFN)
    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)

    # Position-wise Feed Forward Network
    self.feed_forward = nn.Sequential(
        nn.Linear(embed_dim, expansion_factor * embed_dim),  # Expand dimensionality
        nn.ReLU(),                                            # Non-linearity
        nn.Linear(expansion_factor * embed_dim, embed_dim)    # Reduce back to embed_dim
    )

    # Dropout for regularization (after norm1 and norm2)
    self.dropout1 = nn.Dropout(0.1)
    self.dropout2 = nn.Dropout(0.1)

  def forward(self, key, query, value):
      """
      Performs the forward pass for a single Transformer Encoder Block.

      Args:
          key: The key tensor. Shape: (batch_size, sequence_length, features)
          query: The query tensor. Shape: (batch_size, sequence_length, features)
          value: The value tensor. Shape: (batch_size, sequence_length, features)

      Returns:
          norm2_out: The output of the transformer encoder block. Shape: (batch_size, sequence_length, features)
      """

      # Multi-Head Self-Attention Layer
      # Computes attention scores and combines value vectors.
      # The expected output shape is (batch_size, sequence_length, features), e.g., 32x10x512.
      attention_out = self.attention(key, query, value)

      # Add & Norm - First Residual Connection
      # Adds the original input 'value' to the attention output (residual connection)
      # to help with gradient flow and prevent vanishing gradients.
      attention_residual_out = attention_out + value

      # Applies layer normalization and dropout to the residual connection.
      # Layer normalization normalizes across the feature dimension for each sample.
      # Dropout randomly sets a fraction of input units to 0 at each update during training.
      # The expected output shape remains (batch_size, sequence_length, features), e.g., 32x10x512.
      norm1_out = self.dropout1(self.norm1(attention_residual_out))

      # Feed-Forward Network (FFN)
      # Consists of two linear transformations with a ReLU activation in between.
      # It processes each position independently and identically.
      # The shape transformation typically looks like:
      # (batch_size, sequence_length, features) -> (batch_size, sequence_length, hidden_dim) -> (batch_size, sequence_length, features)
      # E.g., 32x10x512 -> 32x10x2048 (intermediate expansion) -> 32x10x512 (projection back).
      feed_fwd_out = self.feed_forward(norm1_out)

      # Add & Norm - Second Residual Connection
      # Adds the output of the first normalization (norm1_out) to the feed-forward output
      # as another residual connection.
      feed_fwd_residual_out = feed_fwd_out + norm1_out

      # Applies layer normalization and dropout to the second residual connection.
      # This is the final output of the transformer encoder block.
      # The expected output shape remains (batch_size, sequence_length, features), e.g., 32x10x512.
      norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))

      return norm2_out
  
  
class TransformerEncoder(nn.Module):
  """
  Implements the Encoder component of the Transformer architecture.
  It processes an input sequence through a stack of identical encoder layers.
  """
  def __init__(self, seq_len: int, vocab_size: int, embed_dim: int, num_layers: int = 6, expansion_factor: int = 4, n_heads: int = 8):
    """
    Initializes the TransformerEncoder.

    Args:
        seq_len (int): The maximum length of the input sequence. Used for positional encoding.
        vocab_size (int): The size of the vocabulary, i.e., the number of unique tokens.
        embed_dim (int): The dimension of the token embeddings and the model's internal representations.
        num_layers (int, optional): The number of identical TransformerBlock layers to stack.
                                    Defaults to 6, as specified in the original "Attention Is All You Need" paper.
        expansion_factor (int, optional): Factor by which the hidden dimension of the feed-forward network
                                          within each TransformerBlock expands. Defaults to 4.
        n_heads (int, optional): The number of attention heads in the MultiHeadAttention mechanism.
                                  Defaults to 8.
    """
    super().__init__()

    # Layer for converting input token IDs into dense vector embeddings.
    self.embedding_layer = Embedding(vocab_size, embed_dim)

    # Layer for adding positional information to the embeddings,
    # as Transformers are permutation-invariant without it.
    self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

    # Stack of TransformerBlock layers. Each block processes the sequence
    # and enhances its contextual representation.
    # num_layers defaults to 6, mirroring the original Transformer paper.
    self.layers = nn.ModuleList([ TransformerBlock(embed_dim, expansion_factor, n_heads) for _ in range(num_layers) ])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Performs the forward pass through the Transformer Encoder.

    Args:
        x (torch.Tensor): The input tensor, typically containing token IDs.
                          Expected shape: (batch_size, seq_len)

    Returns:
        torch.Tensor: The output tensor from the final encoder layer,
                      representing the contextually enriched sequence.
                      Expected shape: (batch_size, seq_len, embed_dim)
                      Example: 32x10x512
    """
    # 1. Convert input token IDs to embeddings.
    # Shape: (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
    embed_out = self.embedding_layer(x)

    # 2. Add positional encodings to the embeddings.
    # This provides information about the order of tokens in the sequence.
    # Shape: (batch_size, seq_len, embed_dim)
    out = self.positional_encoder(embed_out)

    # 3. Pass the combined embeddings and positional encodings through
    # each TransformerBlock in the stack.
    # In an encoder's self-attention, the query, key, and value are all derived
    # from the same input sequence representation ('out' in this case).
    for layer in self.layers:
      out = layer(out, out, out) # Q=out, K=out, V=out for self-attention

    # The final 'out' tensor is the encoded representation of the input sequence.
    return out

class DecoderBlock(nn.Module):
  def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
    super().__init__()

    """
    Decoder block consisting of:
    - Masked self-attention
    - Add & Norm
    - Encoder-Decoder attention + feedforward via TransformerBlock
    """

    # Masked Multi-Head Self Attention (x attends to past and current positions only)
    self.attention = MultiHeadAttention(embed_dim, n_heads)

    # Layer Normalization for stable training
    self.norm = nn.LayerNorm(embed_dim)

    # Dropout for regularization
    self.dropout = nn.Dropout(0.2)

    # Encoder-Decoder attention and feedforward layers
    self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)

  def forward(self, key, query, x, mask):
    """
    Args:
        key: encoder output used as key for cross-attention
        query: decoder hidden states used as query for cross-attention
        x: decoder input (target tokens)
        mask: look-ahead mask for masked self-attention

    Returns:
        out: processed output after masked attention and cross attention
    """

    # Step 1: Masked Self-Attention (decoder attends to its past tokens only)
    attention = self.attention(x, x, x, mask=mask)  # Shape: [batch_size, seq_len, embed_dim]

    # Step 2: Add & Norm with residual connection and dropout
    value = self.dropout(self.norm(attention + x))  # Residual connection

    # Step 3: Cross Attention + Feedforward (using encoder output and decoder value)
    out = self.transformer_block(key, query, value)

    return out


class TransformerDecoder(nn.Module):
  def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=6, expansion_factor=4, n_heads=8):
    super().__init__()

    """
    Transformer Decoder composed of:
    - Token embeddings
    - Positional embeddings
    - Stack of DecoderBlocks
    - Output projection to vocabulary size
    """

    # Embedding for target vocabulary tokens
    self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)

    # Positional encoding to retain sequence order
    self.position_embedding = PositionalEmbedding(seq_len, embed_dim)

    # Stack of DecoderBlocks (6 layers as requested)
    self.layers = nn.ModuleList([ DecoderBlock(embed_dim, expansion_factor, n_heads) for _ in range(num_layers) ])

    # Final linear layer projecting decoder output to vocabulary logits
    self.fc_out = nn.Linear(embed_dim, target_vocab_size)

    # Dropout for regularization
    self.dropout = nn.Dropout(0.2)

  def forward(self, x, enc_out, mask):
    """
    Args:
        x: target sequence input tensor [batch_size, tgt_seq_len]
        enc_out: encoder output to be used for cross-attention
        mask: look-ahead mask to prevent peeking future tokens in decoder

    Returns:
        out: probability distribution over target vocabulary
    """

    # Step 1: Token + Positional Embedding
    x = self.word_embedding(x)               # Shape: [batch_size, tgt_seq_len, embed_dim]
    x = self.position_embedding(x)           # Adds positional information
    x = self.dropout(x)

    # Step 2: Pass through all Decoder Blocks
    for layer in self.layers:
      x = layer(enc_out, x, enc_out, mask)  # key=query=enc_out, x=self decoder input

    # Step 3: Project output to vocabulary logits
    out = F.softmax(self.fc_out(x), dim=-1)   # Shape: [batch_size, tgt_seq_len, vocab_size]

    return out

class Transformer(nn.Module):
  def __init__(self, embed_dim, src_vocab_size, target_vocab_size, en_seq_length, de_seq_length, num_layers=6, expansion_factor=4, n_heads=8):
    super().__init__()

    """
    Args:
        embed_dim: Dimension of embeddings for both encoder and decoder.
        src_vocab_size: Size of the source vocabulary (input language).
        target_vocab_size: Size of the target vocabulary (output language).
        seq_length: Maximum length of input/output sequences.
        num_layers: Number of encoder and decoder layers (stacked blocks).
        expansion_factor: Factor to expand the hidden size in feed-forward networks.
        n_heads: Number of attention heads in multi-head self-attention.
    """

    # Save target vocab size (can be useful during decoding)
    self.target_vocab_size = target_vocab_size

    # Initialize the Transformer encoder
    self.encoder = TransformerEncoder(
      seq_len=en_seq_length,
      vocab_size=src_vocab_size,
      embed_dim=embed_dim,
      num_layers=num_layers,
      expansion_factor=expansion_factor,
      n_heads=n_heads
    )

    # Initialize the Transformer decoder
    self.decoder = TransformerDecoder(
      target_vocab_size=target_vocab_size,
      embed_dim=embed_dim,
      seq_len=de_seq_length,
      num_layers=num_layers,
      expansion_factor=expansion_factor,
      n_heads=n_heads
    )

  def make_trg_mask(self, trg):
    """
    Creates a look-ahead mask for the target sequence to prevent the decoder
    from attending to future tokens (ensures autoregressive property).

    Args:
        trg: Target tensor of shape [batch_size, target_seq_len]

    Returns:
        trg_mask: Causal mask of shape [batch_size, 1, target_seq_len, target_seq_len]
    """
    batch_size, trg_len = trg.shape

    # Create a lower triangular matrix with ones (causal mask)
    # Shape: [trg_len, trg_len] then expand for batch
    trg_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).expand(
      batch_size, 1, trg_len, trg_len
    ).bool()

    return trg_mask

  def forward(self, src, trg):
    """
    Forward pass through the Transformer model.

    Args:
        src: Source input tensor of shape [batch_size, src_seq_len]
        trg: Target input tensor of shape [batch_size, trg_seq_len]

    Returns:
        outputs: Predicted probability distribution over the target vocabulary for each position in the target sequence.
    """
    # Create target mask to prevent attention to future tokens
    trg_mask = self.make_trg_mask(trg)

    # Pass source input through the encoder to get encoder outputs
    enc_out = self.encoder(src)  # Shape: [batch_size, src_seq_len, embed_dim]

    # Pass target sequence and encoder output into the decoder
    outputs = self.decoder(trg, enc_out, trg_mask)  # Shape: [batch_size, trg_seq_len, target_vocab_size]

    return outputs


if __name__ == "__main__":
    model = Transformer(
        embed_dim=512,
        src_vocab_size=37000,
        target_vocab_size=37000,
        en_seq_length=50,        # or more if needed
        de_seq_length=50,        # or more if needed
        num_layers=6,
        expansion_factor=4,   # because 512 * 4 = 2048
        n_heads=8
    )

    print(model)