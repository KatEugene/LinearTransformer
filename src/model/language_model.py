from torch import nn
import torch
from math import log
from src.model.based import Based


class PositionalEncoder(nn.Module):
    def __init__(self, max_seq_len, d_model, dropout):
        super(PositionalEncoder, self).__init__()

        position = torch.arange(0, max_seq_len).reshape(-1, 1)
        dimension = torch.exp(-torch.arange(0, d_model, 2) * log(1e4) / d_model)

        pos_embedding = torch.zeros(size=(max_seq_len, d_model))
        pos_embedding[:, 0::2] = torch.sin(position * dimension)
        pos_embedding[:, 1::2] = torch.cos(position * dimension)
        self.register_buffer('pos_embedding', pos_embedding)

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        return self.dropout(batch + self.pos_embedding[:batch.size(1)])


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, feature_dim, window_size, num_layers, max_seq_len, dropout, tokenizer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(max_seq_len, d_model, dropout)
        self.layers = nn.ModuleList([Based(d_model, num_heads, feature_dim, window_size) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask):
        embeddings = self.embedding(input_ids)
        hidden_states = self.positional_encoder(embeddings)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states)
        logits = self.fc_out(hidden_states)
        return {"logits": logits}

    @torch.inference_mode()
    def inference(self, input_ids, attention_mask, max_length=50):
        for _ in range(max_length):
            logits = self(input_ids, attention_mask)
            next_token = logits[:, -1, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token.unsqueeze(-1))], dim=-1)
        return {"generated_texts": input_ids}
