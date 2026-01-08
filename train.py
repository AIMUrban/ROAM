import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default=None,
                    help="CUDA visible devices")
parser.add_argument("--run_type", type=str,
                    choices=["seq", "gnn"], default="gnn", help="Run mode")
parser.add_argument("--file_desc", type=str, default='0731_ablation_GIL')
parser.add_argument("--early_stop_k", type=int, default=5,
                    help="Early stop if no improvement in k epochs")
parser.add_argument("--early_stop_rel", type=float, default=0.005,
                    help="Early stop if improvement is less than this relative change")
parser.add_argument("--min_lr", type=float, default=1e-6,
                    help="Minimum learning rate threshold for early stopping")
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--test_epoch", type=int, default=1)
parser.add_argument("--min_test_epoch", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--dataset", type=str, default='cityb')
parser.add_argument("--timeslot", type=int, default=168)
parser.add_argument("--emb_dim", type=int, default=32)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--weight_decay", type=float, default=5e-6)
parser.add_argument("--ortho_weight", type=float, default=0)
parser.add_argument("--balance_weight", type=float, default=0)
parser.add_argument("--lr_scheduler", type=str,
                    choices=["plateau", "multistep", "cosine", "linear"],
                    default="linear",
                    help="Learning rate scheduler type")
args = parser.parse_args()

def auto_select_gpu(threshold_mb=20):
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8')
        for i, mem in enumerate(int(x) for x in result.strip().split('\n')):
            if mem < threshold_mb:
                print(f"Using GPU {i}, current memory usage: {mem}MB")
                return str(i)
        print("No available GPU with sufficient memory, using default")
    except Exception as e:
        print(f"Failed to get GPU info: {e}")
    return "0"

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu or auto_select_gpu(20)

import torch
import pandas as pd
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader
from dataset import TrajectoryDataset, build_location_graph, \
    build_user_location_bipartite_graph
from evaluate import evaluate
from tools import load_data, setup_seed, collate_batch, save_checkpoint, prepare_graph, build_scheduler
from datetime import datetime
from model import Predictor

batch_size = args.batch_size
args.dataset = os.path.join('dataset', args.dataset)
data_path = args.dataset
num_timeslot = args.timeslot
num_epoch = args.epoch
test_epoch = args.test_epoch if args.test_epoch else args.epoch
device = torch.device("cuda")
setup_seed(args.seed)

# === Load Data ===
train_df, test_df, pos2id, user2id = load_data(data_path)
num_loc, num_user = len(pos2id), len(user2id)

ul_graph = prepare_graph(lambda **kw: {
    **build_user_location_bipartite_graph(train_df, pos2id, user2id, data_path),
    'num_nodes': num_user + num_loc}, device, norm=False)
ll_graph = prepare_graph(lambda **kw: {
    **build_location_graph(train_df, pos2id, data_path),
    'num_nodes': num_loc}, device, norm=False)

extra_info = {'ul_graph': ul_graph, 'll_graph': ll_graph}

# === Init Model ===
model = Predictor(num_loc, num_user, args).to(device)
print(model)

# === Datasets and Loaders ===
train_dataset = TrajectoryDataset(args, cache_path=os.path.join(data_path, 'train.npy'))
valid_dataset = TrajectoryDataset(args, cache_path=os.path.join(data_path, 'valid.npy'))
test_dataset = TrajectoryDataset(args, cache_path=os.path.join(data_path, 'test.npy'))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

print('Train set size:', len(train_dataset))
print('Test set size:', len(test_dataset))

# === Optimizer and LR scheduler ===
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
loss_fn = nn.NLLLoss(reduction='mean', ignore_index=0)
ms = [8]
if args.lr_scheduler == "multistep":
    args.min_test_epoch = ms[-1]
scheduler = build_scheduler(args, optimizer, len(train_loader), ms)

# ======== Training ========
best_metric, best_epoch, no_improve_count = 0.0, -1, 0
print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  Start training with {args.lr_scheduler} scheduler")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params:,}")

# === Early stop tracker ===
best_user_acc = None
best_sample_records = None
best_overall_acc = None

if args.test:
    fold_name = f"{args.run_type} {args.file_desc}"
    epoch = args.epoch
    saved_model_path = os.path.join(args.dataset, fold_name, f'checkpoint_epoch{epoch}.pth')
    print(f'Loading model from {saved_model_path}')
    if args.gate:
        print("Top gate bias (before load):", model.top_gate.bias[:5])
    model.load_state_dict(torch.load(saved_model_path, map_location=device)['model_state_dict'], strict=True)
    if args.gate:
        print("Top gate bias (after load):", model.top_gate.bias[:5])
    overall_acc, sample_records = evaluate(model, test_loader, extra_info, device, sample_level=True)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] === Overall Location Prediction Accuracy ({args.run_type}) ===")
    for k, acc in overall_acc.items():
        if k.startswith("Acc@"):
            print(f"{k}: {acc:.2%}")
    print(f"MRR: {overall_acc['MRR']:.2%}")
    sample_acc_df = pd.DataFrame(sample_records)
    out_file_sample = "sample_level_accuracy.csv"
    save_path = os.path.join(data_path, fold_name, out_file_sample)
    sample_acc_df.to_csv(save_path, index=False)
    print(f"Sample-level accuracy saved to {out_file_sample}")
    exit()

# === Training Loop ===
for epoch in range(num_epoch):
    model.train()
    total_loss_val = 0
    current_lr = optimizer.param_groups[0]['lr']
    print(f"\nEpoch {epoch+1}/{args.epoch}  LR={current_lr:.6f}")

    for batch in tqdm.tqdm(train_loader, disable=True if epoch > 0 else False):
        logp, aux_loss = model(batch, extra_info)
        ce = loss_fn(logp, batch["target_loc"].to(device))
        loss = ce
        if args.ortho_weight > 0:
            loss += args.ortho_weight * aux_loss['ortho_loss']
        if args.balance_weight > 0:
            loss += args.balance_weight * aux_loss['balance_loss']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.lr_scheduler == "linear":
            scheduler.step()

        total_loss_val += loss.item()

    print(f"Train loss: {total_loss_val / len(train_loader):.4f}")

    if not (epoch + 1) >= args.min_test_epoch:
        continue

    if (epoch+1) % test_epoch == 0:
        overall_acc, sample_records = evaluate(model, test_loader, extra_info, device, sample_level=False)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now_str}] === Overall Location Prediction Accuracy ({args.run_type}) ===")
        for k, acc in overall_acc.items():
            if k.startswith("Acc@"):
                print(f"{k}: {acc:.2%}")
        print(f"MRR: {overall_acc['MRR']:.2%}")

    # === Early Stopping ===
    if (epoch + 1) % test_epoch == 0:
        current_metric = overall_acc["MRR"]
        relative_improvement = (current_metric - best_metric) / (best_metric + 1e-8)

    if args.lr_scheduler != "linear":
        if args.lr_scheduler == "plateau":
            scheduler.step(current_metric)
        else:
            scheduler.step()

    new_lr = optimizer.param_groups[0]['lr']

    if (epoch + 1) % test_epoch == 0:
        if current_metric > best_metric:
            best_metric = current_metric
            best_sample_records = sample_records
            best_overall_acc = overall_acc
            best_epoch = epoch + 1
            print(f"New best MRR: {best_metric:.4f} at epoch {best_epoch}")
            save_checkpoint(args, model, optimizer, epoch)

    if args.lr_scheduler == "plateau":
        if relative_improvement > args.early_stop_rel:
            no_improve_count = 0
            if epoch > 0:
                print(f"MRR improved by {relative_improvement:.2%}, counter reset.")
        else:
            no_improve_count += 1
            print(f"No significant improvement ({relative_improvement:.2%}), count = {no_improve_count}")

        if new_lr < current_lr:
            print(f"LR reduced: {current_lr:.6f} → {new_lr:.6f}.")

        if no_improve_count >= args.early_stop_k:
            print(f"Early stopping triggered after {args.early_stop_k} epochs.")
            break

        if new_lr < args.min_lr:
            print(f"Learning rate too small ({new_lr:.6f} < {args.min_lr}), early stopping.")
            break

if best_sample_records is not None:
    save_checkpoint(args, model, optimizer, best_epoch)
    sample_acc_df = pd.DataFrame(best_sample_records)
    out_file_sample = "sample_level_accuracy.csv"
    save_path = os.path.join(data_path, f"{args.run_type} {args.file_desc}", out_file_sample)
    sample_acc_df.to_csv(save_path, index=False)
    print(f"Sample-level accuracy saved to {out_file_sample}")
