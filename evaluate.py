import torch
from tqdm import tqdm

@torch.no_grad()
def evaluate(model, dataloader, extra_info, device, topk=(1, 3, 5, 10), sample_level=False):
    model.eval()

    sample_records = []
    overall_correct = {k: 0 for k in topk}
    total_mrr = 0.0
    total_samples = 0
    sample_idx = 0

    for batch in tqdm(dataloader, desc="Evaluating", disable=False):
        target = batch["target_loc"].to(device)       # [B]
        uid = batch["uid"]                             # [B]
        lcst_score = batch["lcst_score"].cpu().numpy() # [B]
        B = target.size(0)

        # Forward
        loc_out, _ = model(batch, extra_info)          # [B, num_poi]
        _, pred_topk = loc_out.topk(max(topk), dim=1)  # [B, maxk]

        if sample_level:
            uid = uid.numpy()
            target_np = target.cpu().numpy()
            pred_topk = pred_topk.cpu().numpy()

            for i in range(B):
                u = uid[i]
                t = target_np[i]
                preds = pred_topk[i]
                score = lcst_score[i]

                sample_info = {
                    "sample_idx": sample_idx,
                    "uid": u,
                    "target": t,
                    "lcst_score": score,
                    "pred_topk": preds.tolist(),
                }

                for k in topk:
                    sample_info[f"Acc@{k}"] = int(t in preds[:k])

                logits = loc_out[i]  # [num_poi]
                sorted_idx = torch.argsort(logits, descending=True)
                rank_tensor = (sorted_idx == t).nonzero(as_tuple=True)[0]
                rank = rank_tensor.item() + 1
                sample_info["MRR"] = 1.0 / rank

                sample_records.append(sample_info)
                sample_idx += 1

        else:
            # --- Batched Top-K Accuracy ---
            for k in topk:
                correct = (pred_topk[:, :k] == target.view(-1, 1)).any(dim=1).sum().item()
                overall_correct[k] += correct

            # --- Batched MRR ---
            sorted_indices = torch.argsort(loc_out, dim=1, descending=True)
            target_expanded = target.view(-1, 1).expand_as(sorted_indices)
            ranks = (sorted_indices == target_expanded).nonzero(as_tuple=False)[:, 1] + 1  # [B]
            total_mrr += (1.0 / ranks.float()).sum().item()

        total_samples += B

    if sample_level:
        overall_acc = {
            f"Acc@{k}": sum(r[f"Acc@{k}"] for r in sample_records) / len(sample_records)
            for k in topk
        }
        overall_acc["MRR"] = sum(r["MRR"] for r in sample_records) / len(sample_records)
    else:
        overall_acc = {
            f"Acc@{k}": overall_correct[k] / total_samples for k in topk
        }
        overall_acc["MRR"] = total_mrr / total_samples

    return overall_acc, sample_records
