import os
from collections import defaultdict

import numpy as np
import torch
import tqdm
from sklearn.decomposition import PCA
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, args, cache_path):
        self.samples = []
        self.num_timeslot = args.timeslot

        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        return {
            'uid': item['uid'],
            'loc_seq': item['loc_seq'],
            'time_seq': item[f'time_seq_{self.num_timeslot}'],
            'target_loc': item['target_loc'],
            'target_time': item[f'target_time_{self.num_timeslot}'],
            'js_score': item['js_score'],
            'lcst_score': item['lcst_score'],
            'valid_len': item['valid_len'],
        }


def build_user_location_bipartite_graph(df, pos2id, user2id, save_path):
    save_path = os.path.join(save_path, "user_location_graph.pt")
    if os.path.exists(save_path):
        print(f"Loading bipartite graph from: {save_path}")
        return torch.load(save_path)

    num_loc  = len(pos2id)
    num_user = len(user2id)
    num_nodes = num_loc + num_user

    edge_src, edge_dst = [], []
    edge_w_cnt = defaultdict(int)

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Building bipartite graph"):
        l_idx = pos2id[(row["x"], row["y"])] - 1          # 0 … L2-1
        u_idx = user2id[row["uid"]] - 1 + num_loc         # L2 … L2+U-1
        edge_src.append(u_idx)                            # user → loc
        edge_dst.append(l_idx)
        edge_w_cnt[(u_idx, l_idx)] += 1

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_weight = torch.tensor(
        [edge_w_cnt[(u, v)] for u, v in zip(edge_src, edge_dst)],
        dtype=torch.float
    )

    src_nodes = edge_index[0]
    row_sum = torch.zeros(num_nodes, dtype=torch.float)
    row_sum.index_add_(0, src_nodes, edge_weight)
    edge_weight = edge_weight / row_sum[src_nodes].clamp_min(1e-6)

    data = dict(edge_index=edge_index,
                edge_weight=edge_weight)

    torch.save(data, save_path)
    print(f"Saved bipartite graph to: {save_path}")
    return data

def build_location_graph(df, pos2id, save_path):
    save_path = os.path.join(save_path, "location_graph.pt")
    if os.path.exists(save_path):
        print(f"Loading location graph from: {save_path}")
        return torch.load(save_path)

    edge_cnt = defaultdict(int)
    for _, user_df in tqdm.tqdm(df.groupby("uid"), desc="Building location graph"):
        user_df = user_df.sort_values(["d", "t"])
        loc_ids = [pos2id[(r.x, r.y)] - 1 for _, r in user_df.iterrows()]
        for i in range(len(loc_ids) - 1):
            u, v = loc_ids[i], loc_ids[i + 1]
            edge_cnt[(u, v)] += 1

    edge_src, edge_dst, edge_w = [], [], []
    for (u, v), freq in edge_cnt.items():
        edge_src.append(u)
        edge_dst.append(v)
        edge_w.append(float(freq))

    edge_index  = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_weight = torch.tensor(edge_w,                dtype=torch.float)

    src = edge_index[0]
    row_sum = torch.zeros(src.max().item() + 1, dtype=torch.float)
    row_sum.index_add_(0, src, edge_weight)
    edge_weight = edge_weight / row_sum[src].clamp_min(1e-6)

    data = dict(edge_index=edge_index,
                edge_weight=edge_weight)

    torch.save(data, save_path)
    print(f"Saved location graph to: {save_path}")
    return data