import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearMultiheadAttention(nn.Module):
    def __init__(self, d_model, feature_dim, num_heads):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.epsilon = 1e-9

        self.proj_q = nn.Linear(d_model, feature_dim * num_heads, bias=False)
        self.proj_k = nn.Linear(d_model, feature_dim * num_heads, bias=False)
        self.proj_v = nn.Linear(d_model, self.head_dim * num_heads, bias=False)
        self.proj_o = nn.Linear(self.head_dim * num_heads, d_model, bias=False)

    def forward(self, hidden_states, attention_mask):
        batch_size, seq_len, _ = hidden_states.size()

        q = self.proj_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.feature_dim).transpose(1, 2)
        k = self.proj_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.feature_dim).transpose(1, 2)
        v = self.proj_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = torch.cat([torch.ones_like(q[..., :1]), q, 0.5 * q ** 2], dim=-1)
        k = torch.cat([torch.ones_like(k[..., :1]), k, 0.5 * k ** 2], dim=-1)

        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        qkv = torch.einsum('bhnd,bhde->bhne', q, kv)
        z = torch.einsum('bhnd,bhne->bhne', q, k.sum(dim=2, keepdim=True))
        output = qkv / (z + self.epsilon)
        output = output.masked_fill(attention_mask.unsqueeze(1).unsqueeze(-1) == 0, 0)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.proj_o(output)
        return output


class SlidingWindowMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size

        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        self.proj_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden_states, attention_mask):
        batch_size, seq_len, _ = hidden_states.size()

        q = self.proj_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.proj_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.proj_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.einsum('bhnd,bhmd->bhnm', q, k) / (self.head_dim ** 0.5)
        mask = torch.ones_like(attn_weights).tril(diagonal=self.window_size - 1)
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.einsum('bhnm,bhmd->bhnd', attn_weights, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.proj_o(output)
        return output


class Based(nn.Module):
    def __init__(self, d_model, num_heads, feature_dim, window_size):
        super().__init__()
        self.linear_attention = LinearMultiheadAttention(d_model, feature_dim, num_heads)
        self.window_attention = SlidingWindowMultiheadAttention(d_model, num_heads, window_size)

    def forward(self, hidden_states, attention_mask):
        linear_output = self.linear_attention(hidden_states, attention_mask)
        window_output = self.window_attention(hidden_states, attention_mask)
        return linear_output + window_output
