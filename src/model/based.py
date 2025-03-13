import torch
import torch.nn as nn
from torch.nn.functional import softmax
from math import sqrt


class LinearMultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, feature_dim):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.epsilon = 1e-9

        self.proj_q = nn.Linear(d_model, num_heads * feature_dim, bias=False)
        self.proj_k = nn.Linear(d_model, num_heads * feature_dim, bias=False)
        self.proj_v = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.proj_out = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

    def feature_map(self, x):  # x (..., feature_dim)
        x2 = x.unsqueeze(dim=-1) * x.unsqueeze(dim=-2)  # (..., feature_dim, feature_dim)
        x2 = x2.flatten(start_dim=-2) / sqrt(2)  # (..., feature_dim ** 2)
        x2_normalize_const = sqrt(self.feature_dim)
        x_normalize_const = sqrt(x2_normalize_const)
        return torch.cat(
            [x[..., :1] ** 0, x / x_normalize_const, x2 / x2_normalize_const], dim=-1
        )

    def forward(self, hidden_states, inference_params):
        batch_size, seq_len, _ = hidden_states.size()

        q = self.proj_q(hidden_states)
        q = q.view(batch_size, seq_len, self.num_heads, self.feature_dim)
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, feature_dim)

        k = self.proj_k(hidden_states)
        k = k.view(batch_size, seq_len, self.num_heads, self.feature_dim)
        k = k.transpose(1, 2)  # (batch_size, num_heads, seq_len, feature_dim)

        v = self.proj_v(hidden_states)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        q = self.feature_map(q)  # (batch_size, num_heads, seq_len, 1 + feature_dim + feature_dim^2)
        k = self.feature_map(k)  # (batch_size, num_heads, seq_len, 1 + feature_dim + feature_dim^2)

        if inference_params is None:
            output = self.quadratic_forward(q, k, v)
        else:
            if inference_params.get('kv_state', None) is None:
                inference_params['kv_state'] = torch.zeros(
                    batch_size, self.num_heads, 1, self.head_dim, 1 + self.feature_dim + self.feature_dim ** 2, self.head_dim).to(q.device)
                inference_params['k_state'] = torch.zeros(
                    batch_size, self.num_heads, 1, 1,  1 + self.feature_dim + self.feature_dim ** 2).to(q.device)
            output = self.recurrent_forward(q, k, v)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, num_heads * head_dim)
        output = self.proj_out(output)  # (batch_size, seq_len, d_model)
        return output

    def quadratic_forward(self, q, k, v):
        qk = q @ k.transpose(2, 3)  # (batch_size, num_heads, seq_len, seq_len)
        qk = torch.tril(qk)
        qkv_state = qk @ v  # (batch_size, num_heads, seq_len, head_dim)
        qk_state = (q * k.cumsum(dim=2)).sum(dim=-1)  # (batch_size, num_heads, seq_len)
        output = qkv_state / (qk_state.unsqueeze(dim=-1) + self.epsilon)  # (batch_size, num_heads, seq_len, head_dim)
        return output

    def recurrent_forward(self, q, k, v, kv_state, k_state):
        assert q.size(2) == 1
        q = q.unsqueeze(dim=-2)  # (batch_size, num_heads, 1,       1,        1 + feature_dim + feature_dim^2)
        k = k.unsqueeze(dim=-2)  # (batch_size, num_heads, seq_len, 1,        1 + feature_dim + feature_dim^2)
        v = v.unsqueeze(dim=-1)  # (batch_size, num_heads, seq_len, head_sim, 1)

        kv_state += k[:, :, -1:] * v[:, :, -1:]  # (batch_size, num_heads, 1, head_sim, 1 + feature_dim + feature_dim^2)
        k_state += k[:, :, -1:]  # (batch_size, num_heads, 1, 1, 1 + feature_dim + feature_dim^2)
        qkv_state = (q * kv_state).sum(dim=-1)  # (batch_size, num_heads, 1, head_dim)
        qk_state = (q * k_state).sum(dim=-1)  # (batch_size, num_heads, 1, 1)
        output = qkv_state / (qk_state + self.epsilon)  # (batch_size, num_heads, 1, head_dim)
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
        self.proj_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden_states, attention_mask):
        batch_size, seq_len, _ = hidden_states.size()

        q = self.proj_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.proj_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.proj_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = q @ k.transpose(2, 3) / (self.head_dim ** 0.5)
        mask = torch.ones_like(attn_weights, device=attn_weights.device).tril(diagonal=self.window_size - 1)
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = attn_weights.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        attn_weights = softmax(attn_weights, dim=-1)
        output = attn_weights @ v

        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.proj_out(output)
        return output


class Based(nn.Module):
    def __init__(self, d_model, num_heads, feature_dim, window_size):
        super().__init__()
        self.linear_attention = LinearMultiheadAttention(d_model, num_heads, feature_dim)
        self.window_attention = SlidingWindowMultiheadAttention(d_model, num_heads, window_size)

    def forward(self, hidden_states, attention_mask):
        linear_output = self.linear_attention(hidden_states, None)
        window_output = self.window_attention(hidden_states, attention_mask)
        return linear_output + window_output
