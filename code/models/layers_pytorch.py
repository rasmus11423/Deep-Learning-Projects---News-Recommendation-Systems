import torch
import torch.nn as nn
import torch.nn.functional as F


class AttLayer2(nn.Module):
    """
    Soft alignment attention implementation in PyTorch.

    Attributes:
        dim (int): Attention hidden dimension.
    """

    def __init__(self, dim=200, seed=0):
        """
        Initialize the AttLayer2.

        Args:
            dim (int): Attention hidden dimension.
            seed (int): Seed for initializing weights.
        """
        super(AttLayer2, self).__init__()
        self.dim = dim
        torch.manual_seed(seed)

    def build(self, input_dim):
        """
        Initialize the weights of the attention layer.

        Args:
            input_dim (int): Dimension of the input features.
        """
        self.W = nn.Parameter(torch.empty(input_dim, self.dim, dtype=torch.float32))
        nn.init.xavier_uniform_(self.W)  # Equivalent to Glorot Uniform in TF

        self.b = nn.Parameter(torch.zeros(self.dim, dtype=torch.float32))

        self.q = nn.Parameter(torch.empty(self.dim, 1, dtype=torch.float32))
        nn.init.xavier_uniform_(self.q)

    def forward(self, inputs, mask=None):
        """
        Core implementation of the soft attention.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
            mask (torch.Tensor): Optional mask tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Weighted sum of the input tensor (batch_size, input_dim).
        """
        # Dynamically ensure weights are initialized
        if not hasattr(self, "W"):
            self.build(inputs.size(-1))

        # Move weights to the same device as inputs
        device = inputs.device
        self.W = self.W.to(device)
        self.b = self.b.to(device)
        self.q = self.q.to(device)

        # Compute attention scores
        attention = torch.tanh(torch.matmul(inputs, self.W) + self.b)  # (batch_size, seq_len, dim)
        attention = torch.matmul(attention, self.q).squeeze(-1)  # (batch_size, seq_len)

        # Apply mask if provided
        if mask is not None:
            attention = attention.masked_fill(~mask, float('-inf'))  # Apply mask
        attention = F.softmax(attention, dim=-1)  # Normalize attention scores

        # Compute weighted input
        attention = attention.unsqueeze(-1)  # (batch_size, seq_len, 1)
        weighted_input = inputs * attention  # (batch_size, seq_len, input_dim)

        return weighted_input.sum(dim=1)  # (batch_size, input_dim)

class SelfAttention(nn.Module):
    def __init__(self, multiheads, head_dim, seed=0, mask_right=False):
        super(SelfAttention, self).__init__()
        self.multiheads = multiheads
        self.head_dim = head_dim
        self.output_dim = multiheads * head_dim
        self.mask_right = mask_right
        torch.manual_seed(seed)

        # Placeholder weights; dimensions will be defined during build
        self.WQ = None
        self.WK = None
        self.WV = None

    def build(self, input_dim, device):
        """
        Initialize the weights for query, key, and value transformations.

        Args:
            input_dim (int): Dimension of the input features.
            device (torch.device): Device to initialize the weights on.
        """
        self.WQ = nn.Parameter(torch.empty(input_dim, self.output_dim, device=device))
        self.WK = nn.Parameter(torch.empty(input_dim, self.output_dim, device=device))
        self.WV = nn.Parameter(torch.empty(input_dim, self.output_dim, device=device))

        # Initialize weights using Xavier (Glorot) uniform initialization
        nn.init.xavier_uniform_(self.WQ)
        nn.init.xavier_uniform_(self.WK)
        nn.init.xavier_uniform_(self.WV)

    def mask(self, scores, seq_len, mode="add"):
        """
        Mask operation used in multi-head self-attention.

        Args:
            scores (torch.Tensor): Attention scores (logits).
            seq_len (torch.Tensor): Sequence lengths.
            mode (str): Masking mode ("add" or "mul").

        Returns:
            torch.Tensor: Masked scores.
        """
        if seq_len is None:
            return scores

        # Create a mask of shape (batch_size, seq_len)
        batch_size, seq_len_max = scores.size(0), scores.size(2)
        mask = torch.arange(seq_len_max, device=scores.device).expand(
            batch_size, seq_len_max
        ) < seq_len.unsqueeze(1)

        # Expand mask to match score dimensions
        mask = mask.unsqueeze(1).expand_as(scores)

        if mode == "mul":
            return scores * mask
        elif mode == "add":
            return scores.masked_fill(~mask, -1e12)

    def forward(self, Q, K, V, Q_len=None, V_len=None):
        """
        Core logic of multi-head self-attention.

        Args:
            Q (torch.Tensor): Query tensor (batch_size, seq_len, input_dim).
            K (torch.Tensor): Key tensor (batch_size, seq_len, input_dim).
            V (torch.Tensor): Value tensor (batch_size, seq_len, input_dim).
            Q_len (torch.Tensor, optional): Query sequence lengths.
            V_len (torch.Tensor, optional): Value sequence lengths.

        Returns:
            torch.Tensor: Attention output (batch_size, seq_len, output_dim).
        """
        device = Q.device  # Get device from the input tensor
        input_dim = Q.size(-1)

        # Build weights if they haven't been initialized yet
        if self.WQ is None:
            self.build(input_dim, device)

        # Move Q, K, V to the same device as the weights
        Q, K, V = Q.to(device), K.to(device), V.to(device)

        # Linear transformations for Q, K, V
        Q = torch.matmul(Q, self.WQ).view(Q.size(0), Q.size(1), self.multiheads, self.head_dim).permute(0, 2, 1, 3)
        K = torch.matmul(K, self.WK).view(K.size(0), K.size(1), self.multiheads, self.head_dim).permute(0, 2, 1, 3)
        V = torch.matmul(V, self.WV).view(V.size(0), V.size(1), self.multiheads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=device))

        # Masking
        if self.mask_right:
            seq_len = scores.size(-1)
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1) * -1e12
            scores = scores + mask
        scores = self.mask(scores, V_len, mode="add")

        # Apply softmax to attention scores
        attention_weights = F.softmax(scores, dim=-1)

        # Weighted sum of value vectors
        output = torch.matmul(attention_weights, V)  # (batch_size, multiheads, seq_len, head_dim)

        # Concatenate multiple heads
        output = output.permute(0, 2, 1, 3).contiguous().view(output.size(0), -1, self.output_dim)

        # Mask output for query length
        output = self.mask(output, Q_len, mode="mul")
        return output
