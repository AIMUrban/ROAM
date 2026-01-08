import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, CosineAnnealingLR
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from transformers import get_linear_schedule_with_warmup


def load_data(path):
    train_df = pd.read_csv(f'{path}/train.csv')
    test_df = pd.read_csv(f'{path}/test.csv')
    pos2id = json.load(open(f'{path}/location_to_id.json'))
    user2id = json.load(open(f'{path}/user_to_id.json'))
    pos2id = {eval(k): v for k, v in pos2id.items()}
    user2id = {eval(k): v for k, v in user2id.items()}
    print('user_num:', len(user2id))
    print('poi_num:', len(pos2id))
    return train_df, test_df, pos2id, user2id


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collate_batch(batch):
    uid_list = [torch.tensor(sample['uid']) for sample in batch]
    loc_seqs = [torch.tensor(sample['loc_seq']) for sample in batch]
    time_seqs = [torch.tensor(sample['time_seq']) for sample in batch]
    valid_lens = [torch.tensor(sample['valid_len']) for sample in batch]

    # Padding
    loc_seqs_padded = pad_sequence(loc_seqs, batch_first=True, padding_value=0)     # [B, L2]
    time_seqs_padded = pad_sequence(time_seqs, batch_first=True, padding_value=0)   # [B, L2]
    valid_lens_tensor = torch.stack(valid_lens)  # [B]
    uid_tensor = torch.stack(uid_list)

    attention_mask = torch.arange(loc_seqs_padded.size(1))[None, :] < valid_lens_tensor[:, None]  # [B, L2]

    batch_dict = {'uid': uid_tensor, 'loc_seq': loc_seqs_padded, 'time_seq': time_seqs_padded,
                  'valid_len': valid_lens_tensor, 'attention_mask': attention_mask,
                  'target_loc': torch.tensor([sample['target_loc'] for sample in batch]),
                  'target_time': torch.tensor([sample['target_time'] for sample in batch]),
                  'js_score': torch.tensor([sample['js_score'] for sample in batch], dtype=torch.float),
                  'lcst_score': torch.tensor([sample['lcst_score'] for sample in batch], dtype=torch.float)}

    return batch_dict

def save_checkpoint(args, model, optimizer, epoch):
    save_dir = args.dataset
    save_path = os.path.join(save_dir, f'{args.run_type} {args.file_desc}')
    os.makedirs(save_path, exist_ok=True)
    checkpoint_file = os.path.join(save_path, f"checkpoint_epoch{epoch}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_file)
    print(f"***** Checkpoint saved to {checkpoint_file}")


def prepare_graph(graph_fn, device, norm, **kwargs):
    graph = graph_fn(**kwargs)
    edge_index = graph['edge_index']
    edge_weight = graph['edge_weight']
    if norm:
        edge_index_new, edge_weight_norm = gcn_norm(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=graph['num_nodes'],
            # improved=True,
            add_self_loops=True
        )
        graph['edge_index'] = edge_index_new
        graph['edge_weight'] = edge_weight_norm
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in graph.items()}

# ======== Scheduler Factory ========
def build_scheduler(args, optimizer, len_train_loader, ms):
    if args.lr_scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode='max',
                                 factor=0.3, patience=1,
                                 threshold=0.03, threshold_mode='rel')
    elif args.lr_scheduler == "multistep":
        print(f"MultiStepLR milestones: {ms}")
        return MultiStepLR(optimizer, milestones=ms, gamma=0.3)
    elif args.lr_scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epoch,
                                 eta_min=args.min_lr)
    elif args.lr_scheduler == "linear":
        warmup_ratio = 0.1
        num_steps = len_train_loader * args.epoch
        num_warmup = int(num_steps * warmup_ratio)
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup, num_steps)
    else:
        raise ValueError(f"Unknown scheduler {args.lr_scheduler}")
