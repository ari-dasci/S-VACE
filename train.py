import copy
import math
from tqdm import tqdm
import torch
import torch.nn.functional as F


def train_model(model, train_loader, train_patches, device, num_iter=200,
                pretext_step=64, lr=1e-4, file_name=None, no_wandb=False,
                pretext_fraction=0.1):
    """Train the encoder with velocity pretext followed by BN calibration.

    Phase 1 (first pretext_fraction * num_iter iters): velocity smoothness loss
    with linear decay. Shapes the embedding manifold so that temporal velocity
    is directionally consistent.

    Phase 2 (remaining iters): no loss, BN statistics calibration only.
    """
    try:
        import wandb
        _wandb_ok = not no_wandb
    except ImportError:
        _wandb_ok = False

    if file_name:
        parts = file_name.replace('.csv', '').split('_')
        category = parts[1] if len(parts) > 1 else "unknown"
        ds_id = f"id_{parts[3]}" if len(parts) > 3 else parts[0]
        wandb_prefix = f"{category}/{ds_id}"
    else:
        wandb_prefix = "unknown"

    initial_lr = lr
    final_lr = lr / 10

    def cosine_annealed_lr(iteration):
        t = min(iteration, num_iter)
        return final_lr + (initial_lr - final_lr) * 0.5 * (1 + math.cos(math.pi * t / num_iter))

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)

    total_len = len(train_patches)
    pretext_window = num_iter * pretext_fraction

    iteration_count = 0
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"    [Training]  num_iter={num_iter}, pretext_window={int(pretext_window)}")
    pbar = tqdm(total=num_iter, desc="    >> Training", ncols=80)
    model.train()

    while iteration_count < num_iter:
        for batch_data, batch_indexes in train_loader:
            if iteration_count >= num_iter:
                break

            iteration_count += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = cosine_annealed_lr(iteration_count)

            batch_data = batch_data.to(device, non_blocking=True)
            batch_indexes = batch_indexes.squeeze()
            M = batch_data.shape[0]

            current_lambda = (1.0 - iteration_count / pretext_window
                              if iteration_count < pretext_window else 0.0)

            if current_lambda > 0.0:
                _prev_idx = batch_indexes - pretext_step
                _next_idx = batch_indexes + pretext_step
                _prev_mask = (_prev_idx >= 0) & (_prev_idx < total_len)
                _next_mask = (_next_idx >= 0) & (_next_idx < total_len)
                _prev_clamped = _prev_idx.clamp(0, total_len - 1)
                _next_clamped = _next_idx.clamp(0, total_len - 1)

                prev_patches, next_patches = [], []
                for i in range(M):
                    prev_patches.append(
                        train_patches[_prev_clamped[i].item()].unsqueeze(0) if _prev_mask[i]
                        else torch.zeros_like(train_patches[0].unsqueeze(0))
                    )
                    next_patches.append(
                        train_patches[_next_clamped[i].item()].unsqueeze(0) if _next_mask[i]
                        else torch.zeros_like(train_patches[0].unsqueeze(0))
                    )
                prev_patches = torch.cat(prev_patches, dim=0).to(device, non_blocking=True)
                next_patches = torch.cat(next_patches, dim=0).to(device, non_blocking=True)

                all_embeddings = model.embedding(
                    torch.cat([batch_data, prev_patches, next_patches], dim=0)
                )
                h_anchors = all_embeddings[:M]
                h_prev    = all_embeddings[M:2*M]
                h_next    = all_embeddings[2*M:3*M]

                both_valid = _prev_mask.to(device) & _next_mask.to(device)
                if both_valid.any():
                    v_back = F.normalize(h_anchors[both_valid] - h_prev[both_valid], dim=1, eps=1e-12)
                    v_fwd  = F.normalize(h_next[both_valid]   - h_anchors[both_valid], dim=1, eps=1e-12)
                    smooth_loss = (1.0 - (v_back * v_fwd).sum(dim=1)).mean()
                else:
                    smooth_loss = torch.tensor(0.0, device=device)

                final_loss = current_lambda * smooth_loss
            else:
                with torch.no_grad():
                    model.embedding(batch_data)  # BN calibration forward pass
                smooth_loss = torch.tensor(0.0, device=device)
                final_loss  = torch.tensor(0.0, device=device, requires_grad=True)

            if _wandb_ok:
                import wandb
                wandb.log({
                    f"{wandb_prefix}/smooth_loss": smooth_loss.item(),
                    f"{wandb_prefix}/lambda": current_lambda,
                    f"{wandb_prefix}/lr": cosine_annealed_lr(iteration_count),
                    f"{wandb_prefix}/iteration": iteration_count,
                })

            optimizer.zero_grad(set_to_none=True)
            final_loss.backward()
            optimizer.step()
            pbar.update(1)

            if final_loss.item() < best_loss:
                best_loss = final_loss.item()
                best_model_wts = copy.deepcopy(model.state_dict())

    pbar.close()
    model.load_state_dict(best_model_wts)
