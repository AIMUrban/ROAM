import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import MultiheadAttention



class TransEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model, nhead,
            dropout=0, batch_first=True)
        self.ffn = FFN(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, need_weights=False):
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
        )
        x = attn_output
        x = self.dropout(x)

        x = self.norm1(src + x)
        x = self.norm2(x + self.ffn(x))

        if need_weights:
            return x, attn_weights
        else:
            return x


class TransEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([
            TransEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = norm

    def forward(self, x, mask=None, src_key_padding_mask=None, need_weights=True):
        all_attn = []
        for layer in self.layers:
            if need_weights:
                x, attn = layer(x, mask, src_key_padding_mask, need_weights=True)
                all_attn.append(attn)
            else:
                x = layer(x, mask, src_key_padding_mask, need_weights=False)

        if self.norm is not None:
            x = self.norm(x)

        if need_weights:
            all_attn = torch.stack(all_attn, dim=0)
            return x, all_attn
        else:
            return x



class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout_opt = nn.Dropout(dropout)

    def forward(self, query, key, value, need_weights=False):
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        seq_len_v = value.shape[1]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_v, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        attn_weights = F.softmax(attn_scores, dim=-1)

        if self.dropout > 0.0:
            attn_weights = self.dropout_opt(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).reshape(batch_size, seq_len_q, self.embed_dim)
        output = self.out_proj(output)

        if need_weights:
            avg_attn_weights = attn_weights.mean(dim=1)
            return output, avg_attn_weights
        else:
            return output, None

class FFN(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1, activation='relu', bias=True):
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_hidden, bias=bias)
        self.down_proj = nn.Linear(d_hidden, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'swish':
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        x = self.dropout(x)

        return x


def expert_balance_loss(hard_index, soft_score, num_proto):
    N = hard_index.size(0)

    count = torch.bincount(hard_index, minlength=num_proto).float()
    freq = count / N

    prob_sum = soft_score.sum(dim=0)
    prob_mean = prob_sum / N

    balance_loss = (freq * prob_mean).sum() * num_proto

    return balance_loss


class AttentionRouterGate(nn.Module):
    def __init__(self, dim_in_seq, dim_in_gnn, dim_hidden, num_proto, dim_out):
        super().__init__()
        self.num_proto = num_proto
        self.seq_align = nn.Linear(dim_in_seq, dim_hidden)
        self.gnn_c_align = nn.Linear(dim_in_gnn, dim_hidden)
        self.gnn_i_align = nn.Linear(dim_in_gnn, dim_hidden)
        self.q_proj = nn.Linear(dim_hidden, dim_hidden)
        self.v_proj = nn.Linear(dim_hidden, dim_hidden)
        self.k_proj = nn.Linear(dim_hidden, dim_hidden)

        self.gate = nn.Linear(dim_hidden, dim_out)

    def forward(self, branch_vecs, gate_input=None):
        seq_context = branch_vecs['seq']
        gnn_c_context = branch_vecs['gnn_c']
        gnn_i_context = branch_vecs['gnn_i']
        seq_ctx = self.seq_align(seq_context)
        gnn_c_ctx = self.gnn_i_align(gnn_i_context)
        gnn_i_ctx = self.gnn_c_align(gnn_c_context)

        v_stack = torch.stack([seq_ctx, gnn_c_ctx, gnn_i_ctx], dim=1)
        mean_context = v_stack.mean(dim=1)
        q = self.q_proj(mean_context).unsqueeze(1)
        k = self.k_proj(v_stack)
        v = self.v_proj(v_stack)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        weights = torch.softmax(attn_scores, dim=-1)
        fused = torch.matmul(weights, v).squeeze(1)

        out = self.gate(fused)
        out = F.softmax(out, dim=-1)

        return out


class SharedRouterGate(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_proto=None, rank=None):
        super().__init__()
        self.num_proto = num_proto
        self.rank = rank
        self.in_proj = nn.Linear(dim_in, dim_hidden)
        self.drop = nn.Dropout(0.1)
        self.relu = nn.ReLU()

        if num_proto > 1:
            self.gate_weight = nn.Parameter(torch.randn(num_proto, dim_hidden, dim_out))
            self.gate_bias = nn.Parameter(torch.zeros(num_proto, dim_out))
        else:
            self.out_proj = nn.Linear(dim_hidden, dim_out)

    def forward(self, gate_input, proto_ids=None, proto_soft=None):
        x = self.drop(self.in_proj(gate_input))
        B, H = x.shape

        if self.num_proto > 1:
            W = self.gate_weight[proto_ids]
            b = self.gate_bias[proto_ids]
            out = torch.einsum('bh,bhd->bd', x, W) + b

            if proto_soft is not None:
                w_weight = proto_soft[torch.arange(B), proto_ids]
                out = out * w_weight.unsqueeze(1)
        else:
            out = self.out_proj(x)

        gate = F.softmax(out, dim=-1)
        return gate




