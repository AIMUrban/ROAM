import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_modules import expert_balance_loss, TransEncoder
from gnn_mod import GNNEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim
        pos_encoding = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * -(math.log(10000.0) / emb_dim))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)
        self.dropout = nn.Dropout(0.1)

    def forward(self, out):
        out = out + self.pos_encoding[:, :out.size(1)].detach()
        out = self.dropout(out)
        return out


class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Classifier, self).__init__()
        self.up_proj = nn.Linear(in_dim,in_dim * 2)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(in_dim * 2, out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x1 = self.up_proj(x)
        x = self.relu(x1)
        # x = self.dropout(x)
        x = self.classifier(x)
        return x


class Predictor(nn.Module):
    def __init__(self, num_loc, num_user, args):
        super(Predictor, self).__init__()
        time_dim = args.emb_dim
        user_dim = args.emb_dim
        loc_dim = args.emb_dim
        num_time = args.timeslot
        self.test = args.test
        self.gate = args.gate
        self.draw = args.draw
        self.only_gnnc = args.only_gnnc
        self.no_hier = args.no_hier
        self.balance_weight = args.balance_weight
        self.ortho_weight = args.ortho_weight

        self.run_type = args.run_type
        self.num_proto = args.num_proto
        self.num_loc = num_loc
        self.num_user = num_user

        tau_start = float(getattr(args, "proto_tau_start", 1.0))
        tau_min = float(getattr(args, "proto_tau_min", 0.5))
        tau_decay = float(getattr(args, "proto_tau_decay", 0.95))

        self.register_buffer("proto_tau", torch.tensor(tau_start, dtype=torch.float32))
        self.proto_tau_min = tau_min
        self.proto_tau_decay = tau_decay

        self.loc_emb_layer = nn.Embedding(num_loc + 1, loc_dim, padding_idx=0)
        self.time_embed_layer = nn.Embedding(num_time + 1, time_dim, padding_idx=0)
        self.user_embed_layer = nn.Embedding(num_user, user_dim)

        input_dim = loc_dim
        input_dim += time_dim
        encoder_norm = nn.LayerNorm(input_dim)
        self.pe = PositionalEncoding(input_dim)
        self.seq_encoder = TransEncoder(
            d_model=input_dim,
            nhead=2,
            dim_feedforward=input_dim,
            num_layers=2,
            dropout=0.1,
            norm=encoder_norm,
        )
        self.dropout = nn.Dropout(0.1)

        self.user_noroutine_embed_layer = nn.Embedding(num_user, user_dim)
        self.ul_gnn = GNNEncoder(
            in_dim=loc_dim,
            hidden_dims=[loc_dim, loc_dim],
            edge_feat_dim=0,
            dropout=0.1
        )
        self.gnn_context_proj_i = nn.Linear(loc_dim * 2, loc_dim, bias=False)
        self.proto_emb_layer = nn.Embedding(self.num_proto, loc_dim)
        self.gnn_context_proj_c = nn.Linear(loc_dim * 2, loc_dim, bias=False)
        self.pl_gnn = GNNEncoder(
            in_dim=loc_dim,
            hidden_dims=[loc_dim],
            edge_feat_dim=0,
            dropout=0.1
        )
        self.user_q = nn.Linear(loc_dim, loc_dim, bias=False)
        self.proto_k = nn.Linear(loc_dim, loc_dim, bias=False)
        self.proto_q_sem = nn.Linear(loc_dim, loc_dim, bias=False)

        gate_input_dim = 4
        gate_hidden_dim = gate_input_dim
        self.top_gate = nn.Linear(gate_hidden_dim, 2)
        p_top = torch.tensor([0.7, 0.3])
        self.sub_gate = nn.Linear(gate_hidden_dim+1, 2)
        p_sub = torch.tensor([0.7, 0.3])
        with torch.no_grad():
            self.top_gate.bias.copy_(p_top.log())
            self.sub_gate.bias.copy_(p_sub.log())
            nn.init.xavier_uniform_(self.top_gate.weight)
            nn.init.xavier_uniform_(self.sub_gate.weight)

        self.loc_classifier = Classifier(input_dim + user_dim, num_loc + 1)

    def forward(self, batch, extra_info):
        device = self.loc_emb_layer.weight.device
        uid = batch['uid'].to(device)
        uid = uid-1
        loc_seqs = batch['loc_seq'].to(device)
        time_seqs = batch['time_seq'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        valid_len = batch['valid_len'].to(device)
        lcst = batch['lcst_score'].to(device)
        B, L = loc_seqs.size()
        valid_idx = valid_len - 1
        ll_edge_index = extra_info['ll_graph']['edge_index']
        ll_edge_weight = extra_info['ll_graph']['edge_weight']

        user_emb = self.user_embed_layer(uid)
        time_emb = self.time_embed_layer(time_seqs)

        # user_emb_q = self.user_noroutine_embed_layer.weight
        # user_emb_q = self.user_q(self.user_embed_layer.weight)
        user_emb_q = self.user_q(self.user_noroutine_embed_layer.weight)
        proto_emb_k = self.proto_k(self.proto_emb_layer.weight)
        # proto_emb_k = self.proto_emb_layer.weight
        proto_attn_weight = user_emb_q @ proto_emb_k.T
        logits = proto_attn_weight / (proto_emb_k.size(1) ** 0.5)
        proto_attn_soft = F.softmax(logits, dim=-1)
        hard_index = proto_attn_soft.argmax(dim=-1)

        ul_edge_index = extra_info['ul_graph']['edge_index']
        ul_edge_weight = extra_info['ul_graph']['edge_weight']
        user_node_ids = torch.arange(self.num_user, device=device)+self.num_loc
        proto_node_ids = torch.arange(self.num_proto, device=device) + self.num_user + self.num_loc

        bi_node_emb_all = torch.cat([
            self.loc_emb_layer.weight[1:],
            self.user_noroutine_embed_layer.weight,
            # self.user_embed_layer.weight,
        ], dim=0)  # [L2+P, d]
        bi_edge_index = torch.cat([
            ul_edge_index,
            ll_edge_index,
        ], dim=1)
        bi_edge_weight = torch.cat([
            ul_edge_weight,
            ll_edge_weight,
        ], dim=0)

        pu_edge_src = proto_node_ids[hard_index]
        pu_edge_dst = user_node_ids
        pu_edge_index = torch.stack([pu_edge_src, pu_edge_dst], dim=0)
        pu_edge_weight = proto_attn_soft[torch.arange(self.num_user, device=device), hard_index]
        bi_node_emb_all = torch.cat([
            bi_node_emb_all,
            self.proto_emb_layer.weight,
        ], dim=0)

        bi_edge_index = torch.cat([
            bi_edge_index,
            pu_edge_index
        ], dim=1)
        bi_edge_weight = torch.cat([
            bi_edge_weight,
            pu_edge_weight,
        ], dim=0)

        bi_node_emb_all = self.dropout(bi_node_emb_all)
        # bi_edge_index, bi_edge_weight = add_self_loops(
        #     bi_edge_index, bi_edge_weight, fill_value=0.1, num_nodes=bi_node_emb_all.size(0)
        # )
        bignn_output = self.ul_gnn(
            bi_node_emb_all,
            edge_index=bi_edge_index,
            edge_weight=bi_edge_weight,
        )
        bignn_output_proto = bignn_output[-self.num_proto:]
        bignn_output_user = bignn_output[self.num_loc:-self.num_proto]
        bignn_output_loc = bignn_output[:self.num_loc]

        bignn_output_loc = torch.cat([
            bignn_output_loc.new_zeros(1, bignn_output_loc.shape[1]),
            bignn_output_loc
        ], dim=0)

        proto_q = self.proto_q_sem(bignn_output_proto)
        loc_k = bignn_output_loc[1:]
        proto_q = self.dropout(proto_q)
        proto_q_norm = F.normalize(proto_q, p=2, dim=-1)
        loc_k_norm = F.normalize(loc_k, p=2, dim=-1)
        proto2loc_score_semantic = proto_q_norm @ loc_k_norm.T
        score_semantic = proto2loc_score_semantic.clone()

        # U, L, P = self.num_user, self.num_loc, 10
        # pu_index = torch.stack([hard_index, torch.arange(U, device=device)], dim=0)
        # pu_value = proto_attn_soft[torch.arange(U), hard_index]
        # A_pu = torch.sparse_coo_tensor(pu_index, pu_value, size=(P, U))  # [P, U]
        #
        # ul_edge_index = extra_info['ul_graph']['edge_index']
        # ul_edge_weight = extra_info['ul_graph']['edge_weight']
        # ul_src = ul_edge_index[0] - self.num_loc
        # ul_dst = ul_edge_index[1]
        # valid = (ul_src >= 0) & (ul_src < U) & (ul_dst >= 0) & (ul_dst < L)
        # lu_index = torch.stack([ul_dst[valid], ul_src[valid]], dim=0)
        # lu_value = ul_edge_weight[valid]
        # A_lu = torch.sparse_coo_tensor(lu_index, lu_value, size=(L, U))
        # score_struct = torch.spmm(A_pu, A_lu.T).to_dense()
        # score_semantic_pos = F.relu(score_semantic)
        # score_semantic = F.normalize((score_semantic_pos+score_struct), p=1, dim=1)
        # proto2loc_score[proto2loc_score < 0] = float('-inf')
        # proto2loc_score = proto2loc_score + score_struct
        # proto2loc_score_soft = F.softmax(proto2loc_score, dim=1)

        proto_node_ids = torch.arange(self.num_proto, device=device) + self.num_loc
        loc_node_ids = torch.arange(self.num_loc, device=device)
        proto2loc_src = proto_node_ids.repeat_interleave(self.num_loc)
        proto2loc_dst = loc_node_ids.repeat(self.num_proto)
        pl_edge_index = torch.stack([proto2loc_src, proto2loc_dst], dim=0)
        pl_edge_weight = score_semantic.flatten()

        mask = pl_edge_weight > 0
        pl_edge_index = pl_edge_index[:, mask]
        pl_edge_weight = pl_edge_weight[mask]

        pl_node_emb_all = torch.cat([
            self.loc_emb_layer.weight[1:],
            self.proto_emb_layer.weight,
        ])
        pl_edge_index = torch.cat([
            pl_edge_index,
            ll_edge_index,
        ], dim=1)
        pl_edge_weight = torch.cat([
            pl_edge_weight,
            ll_edge_weight,
        ], dim=0)

        pl_node_emb_all = self.dropout(pl_node_emb_all)
        # pl_edge_index, pl_edge_weight = add_self_loops(
        #     pl_edge_index, pl_edge_weight, fill_value=0.1, num_nodes=pl_node_emb_all.size(0)
        # )
        plgnn_output = self.pl_gnn(
            pl_node_emb_all,
            edge_index=pl_edge_index,
            edge_weight=pl_edge_weight,
        )
        plgnn_output_proto = plgnn_output[-self.num_proto:]
        plgnn_output_loc = plgnn_output[:self.num_loc]
        plgnn_output_loc = torch.cat([
            plgnn_output_loc.new_zeros(1, plgnn_output_loc.shape[1]),
            plgnn_output_loc
        ], dim=0)

        # === Transformer ===
        loc_emb = self.loc_emb_layer(loc_seqs)
        lt_emb = torch.cat([loc_emb, time_emb], dim=-1)
        src_key_padding_mask = ~attention_mask
        seq_out = self.seq_encoder(
            self.pe(lt_emb * math.sqrt(lt_emb.shape[-1])),
            mask=None,
            src_key_padding_mask=src_key_padding_mask,
        )

        seq_h, seq_attn_scores = seq_out
        layer_entropies = []
        attn = seq_attn_scores[-1]
        idx_1 = valid_len - 1
        idx_2 = valid_len - 2
        idx_3 = valid_len - 2
        attn_1 = attn[torch.arange(B), idx_1]
        # exit()
        attn_2 = attn[torch.arange(B), idx_2]
        attn_3 = attn[torch.arange(B), idx_3]

        ent_1 = -((attn_1+ 1e-10) * (attn_1 + 1e-10).log()).sum(-1)
        ent_2 = -((attn_2+ 1e-10) * (attn_2 + 1e-10).log()).sum(-1)
        ent_3 = -((attn_3+ 1e-10) * (attn_3 + 1e-10).log()).sum(-1)

        max_H = valid_len.float().log().clamp(min=1.0)
        norm_ent_1 = ent_1 / max_H
        norm_ent_2 = ent_2 / max_H
        norm_ent_3 = ent_3 / max_H
        # print(norm_ent_1.max(), norm_ent_2.max(), norm_ent_3.max())
        layer_entropies.append(norm_ent_3.unsqueeze(1))
        layer_entropies.append(norm_ent_2.unsqueeze(1))
        layer_entropies.append(norm_ent_1.unsqueeze(1))

        entropy_features = torch.cat(layer_entropies, dim=1)

        cur_loc_id = loc_seqs[torch.arange(B), valid_idx]
        cur_time_id = time_seqs[torch.arange(B), valid_idx]
        seq_emb = seq_h[torch.arange(B), valid_idx]
        seq_context_h = torch.cat([user_emb, seq_emb], dim=-1)
        seq_logits = self.loc_classifier(seq_context_h)

        cur_time_emb = self.time_embed_layer(cur_time_id)
        cur_loc_emb = self.loc_emb_layer(cur_loc_id)
        # bignn_context_i = torch.cat([bignn_output_user[uid], bignn_output_loc[cur_loc_id]], dim=-1)
        bignn_context_i = torch.cat([self.user_noroutine_embed_layer(uid), bignn_output_loc[cur_loc_id]], dim=-1)
        # bignn_context_i = torch.cat([self.user_noroutine_embed_layer(uid), cur_loc_emb], dim=-1)
        bignn_context_i = self.gnn_context_proj_i(bignn_context_i)
        gnn_logits_i = bignn_context_i @ bignn_output_loc.T

        bignn_context_c = torch.cat([plgnn_output_proto[hard_index][uid], plgnn_output_loc[cur_loc_id]], dim=-1)
        # bignn_context_c = torch.cat([self.proto_emb_layer(hard_index[uid]), plgnn_output_loc[cur_loc_id]], dim=-1)
        # bignn_context_c = torch.cat([self.proto_emb_layer(hard_index[uid]), cur_loc_emb], dim=-1)
        bignn_context_c = self.gnn_context_proj_c(bignn_context_c)
        gnn_logits_c = bignn_context_c @ plgnn_output_loc.T
        # gnn_logits_c = bignn_context_c @ self.gnn_context_proj_c(plgnn_output_loc).T

        gnn_col = F.softmax(gnn_logits_c, dim=-1)
        topk_vals_gnnc, _ = gnn_col.topk(k=2, dim=-1)
        p1_gnnc = topk_vals_gnnc[:, 0]  # [B]
        p2_gnnc = topk_vals_gnnc[:, 1]  # [B]
        margin_gnnc = p1_gnnc - p2_gnnc  # [B]  ∈(0,1)
        gnn_ind = F.softmax(gnn_logits_i, dim=-1)
        topk_vals_gnni, _ = gnn_ind.topk(k=2, dim=-1)
        p1_gnni = topk_vals_gnni[:, 0]  # [B]
        p2_gnni = topk_vals_gnni[:, 1]  # [B]
        margin_gnni = p1_gnni - p2_gnni  # [B]  ∈(0,1)

        feat_sub = torch.cat([
            lcst.unsqueeze(-1),
            margin_gnnc.unsqueeze(-1),
            p1_gnnc.unsqueeze(-1),
            margin_gnni.unsqueeze(-1),
            p1_gnni.unsqueeze(-1),
        ], dim=-1)  # [B,4]
        g_sub = F.softmax(self.sub_gate(feat_sub), dim=-1)  # [B,2]
        gamma_i = g_sub[:, 0].unsqueeze(-1)
        gamma_c = g_sub[:, 1].unsqueeze(-1)

        gnn_mix = gamma_i * gnn_logits_i + gamma_c * gnn_logits_c

        seq_ent = entropy_features[:, -1:]
        seq_probs = F.softmax(seq_logits, dim=-1)
        topk_vals_seq, _ = seq_probs.topk(k=2, dim=-1)
        p1_seq = topk_vals_seq[:, 0]  # [B]
        p2_seq = topk_vals_seq[:, 1]  # [B]
        margin_seq = p1_seq - p2_seq  # [B]  ∈(0,1)
        feat_top = torch.cat([
            lcst.unsqueeze(-1),
            seq_ent,
            margin_seq.unsqueeze(-1),
            p1_seq.unsqueeze(-1),
        ], dim=-1)
        gate_w = F.softmax(self.top_gate(feat_top), dim=-1)
        w_routine = gate_w[:, 0].unsqueeze(1)
        w_nonroutine = gate_w[:, 1].unsqueeze(1)
        final_logits = w_routine * seq_logits + w_nonroutine * gnn_mix


        aux_loss = None
        if self.balance_weight>0 or self.ortho_weight>0:
            balance_loss = expert_balance_loss(hard_index, proto_attn_soft, self.num_proto)
            proto_vecs = score_semantic
            proto_vecs_norm = F.normalize(proto_vecs, dim=-1)  # [P, d]
            sim_matrix = proto_vecs_norm @ proto_vecs_norm.T  # [P, P] ∈ [-1, 1]
            mask = ~torch.eye(self.num_proto, dtype=torch.bool, device=proto_vecs.device)
            sim_matrix_offdiag = sim_matrix[mask]  # [P * (P-1)]
            ortho_loss = sim_matrix_offdiag.pow(2).mean()  # 越小越正交
            aux_loss = {
                'balance_loss': balance_loss,
                'ortho_loss': ortho_loss
            }
        if self.draw:
            exit()
        return F.log_softmax(final_logits, dim=-1), aux_loss

