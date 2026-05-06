import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from sklearn.cluster import MiniBatchKMeans

# Distance-based anomaly scoring 
@torch.inference_mode()
def calculate_anomaly_scores(model, data_loader, memory_bank, device, top_k=3):
    
    model.eval()
    all_scores = []
    memory_bank = F.normalize(memory_bank.to(device, dtype=torch.float32), dim=1, eps=1e-12)


    for data, _ in data_loader:
        data = data.to(device, non_blocking=True, dtype=torch.float32)
        feats = model.embedding(data)  # (B, D)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        feats = F.normalize(feats, dim=1, eps=1e-12)
        feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Cosine similarity & distance
        sims = feats @ memory_bank.T                    # (B, M)
        sims = torch.nan_to_num(sims, nan=-1.0, posinf=1.0, neginf=-1.0)
        topk_sim, _ = torch.topk(sims, k=top_k, dim=1, largest=True)
        dists = 1.0 - topk_sim
        scores = dists.mean(dim=1)

        scores = torch.nan_to_num(scores, nan=1.0, posinf=1.0, neginf=0.0)
        all_scores.extend(scores.cpu().tolist())

    return all_scores


# Patch-to-point score distribution 
def distribute_patch_scores_to_points(patch_scores, patch_size, num_points): 

    patch_scores = np.nan_to_num(np.asarray(patch_scores, dtype=np.float32),
                                 nan=0.0, posinf=0.0, neginf=0.0)

    kernel = np.ones(patch_size, dtype=np.float32)
    sums   = np.convolve(patch_scores, kernel, mode='full')[:num_points]
    counts = np.convolve(np.ones_like(patch_scores), kernel, mode='full')[:num_points]

    point_scores = np.divide(
        sums, counts,
        out=np.zeros(num_points, dtype=np.float32),
        where=counts != 0
    )
    return np.nan_to_num(point_scores, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Mahalanobis scorer — drop-in replacement for calculate_anomaly_scores
# ---------------------------------------------------------------------------

def fit_mahalanobis(model, data_loader, device, reg=1e-5, mode="full"):
    model.eval()
    all_embs = []

    with torch.inference_mode():
        for data, _ in data_loader:
            data = data.to(device, non_blocking=True, dtype=torch.float32)
            emb = model.embedding(data)                     # (B, D)
            emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
            all_embs.append(emb.cpu().numpy())

    embs = np.concatenate(all_embs, axis=0).astype(np.float64)  # (N, D)
    N, D = embs.shape

    mu = embs.mean(axis=0)                                  # (D,)
    centered = embs - mu                                    # (N, D)

    if mode == "full":
        Sigma = (centered.T @ centered) / N                 # (D, D)
        Sigma += reg * np.eye(D)
        
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)   # ascending
        inv_lam = 1.0 / eigenvalues
        precision = (eigenvectors * inv_lam) @ eigenvectors.T  # (D, D)

        print(f"Cond Number: {eigenvalues[-1] / eigenvalues[0]:.2e}")
        print(f"Max Eigenval: {eigenvalues[-1]:.4f}")
        print(f"Min Eigenval: {eigenvalues[0]:.4e}")
        print(f"Trace (Sum Var): {np.sum(eigenvalues):.4f}")

    elif mode == "diag":
        variances = (centered ** 2).mean(axis=0) + reg      # (D,)
        precision = np.diag(1.0 / variances)

    elif mode == "iso":
        var = (centered ** 2).mean() + reg
        precision = (1.0 / var) * np.eye(D)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {"mu": mu, "precision": precision, "mode": mode}

@torch.inference_mode()
def calculate_anomaly_scores_mahalanobis(model, data_loader, mahal_params, device):
    model.eval()
    mu        = mahal_params["mu"]                          # (D,)
    precision = mahal_params["precision"]                   # (D, D)

    mu_t   = torch.tensor(mu,        dtype=torch.float32, device=device)
    prec_t = torch.tensor(precision, dtype=torch.float32, device=device)

    all_scores = []

    for data, _ in data_loader:
        data = data.to(device, non_blocking=True, dtype=torch.float32)
        emb = model.embedding(data)                         # (B, D)
        emb = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)

        centered = emb - mu_t                               # (B, D)
        mahal_sq = (centered @ prec_t * centered).sum(dim=1)  # (B,)
        scores   = torch.sqrt(torch.clamp(mahal_sq, min=0.0)) # (B,)
        scores   = torch.nan_to_num(scores, nan=1.0, posinf=1.0, neginf=0.0)

        all_scores.extend(scores.cpu().tolist())

    return all_scores


# ---------------------------------------------------------------------------
# Velocity memory bank — captures normal dynamics / tendencies
# ---------------------------------------------------------------------------

@torch.inference_mode()
def collect_ordered_embeddings(model, data_loader, device):
    """Collect all embeddings sorted by patch index (handles shuffled loaders)."""
    model.eval()
    all_embs = []
    all_idx  = []

    for data, idx in data_loader:
        data = data.to(device, non_blocking=True, dtype=torch.float32)
        emb  = model.embedding(data)
        emb  = torch.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        all_embs.append(emb.cpu())
        all_idx.append(idx.squeeze(1).cpu())

    all_embs = torch.cat(all_embs, dim=0)   # (T, D)
    all_idx  = torch.cat(all_idx,  dim=0)   # (T,)
    order    = torch.argsort(all_idx)
    return all_embs[order]                   # temporally ordered


def build_velocity_bank(ordered_embs, delta=48, num_cores=0.1):
    """Build a memory bank of normalised velocity vectors.

    For each consecutive pair of training embeddings separated by `delta`
    steps, computes v_t = normalize(z_{t+delta} - z_t).  The resulting
    velocity vectors are compressed via MiniBatchKMeans to a compact bank
    of typical training tendencies.

    Args:
        ordered_embs: (T, D) tensor of temporally ordered training embeddings.
        delta:        Temporal stride for velocity computation.
        num_cores:    Fraction of velocities to keep as cluster centres (same
                      convention as create_memory_bank).
    Returns:
        (M, D) tensor of L2-normalised velocity cluster centres.
    """
    T, D = ordered_embs.shape
    if T <= delta:
        raise ValueError(f"Too few training patches ({T}) for delta={delta}")

    raw_vel    = ordered_embs[delta:] - ordered_embs[:-delta]  # (T-delta, D)
    velocities = F.normalize(raw_vel, dim=1, eps=1e-12)        # (T-delta, D)
    velocities = torch.nan_to_num(velocities, nan=0.0)

    N = velocities.shape[0]
    k = int(round(num_cores * N)) if isinstance(num_cores, float) else int(num_cores)
    k = min(k, 500, N - 1)   # hard cap: velocity bank never exceeds 500 centres

    if k >= N:
        return velocities

    mbk = MiniBatchKMeans(
        n_clusters=k, init='k-means++', random_state=42,
        batch_size=max(8192, k), max_iter=50, n_init=1,
        reassignment_ratio=0.01,
    )
    mbk.fit(velocities.numpy())

    centers = torch.tensor(mbk.cluster_centers_, dtype=torch.float32)
    centers = F.normalize(centers, dim=1, eps=1e-12)   # re-normalise after averaging
    return centers                                      # (M, D


@torch.inference_mode()
def calculate_anomaly_scores_velocity(model, test_loader, vel_bank, device,
                                      delta=48, top_k=3):
    """Score test patches by how unusual their embedding velocity is.

    For each test patch at position t, computes
        v_t = normalize(z_{t+delta} - z_t)
    and measures the average cosine distance to the top-k nearest velocity
    cluster centres.  The last `delta` positions (no forward neighbour) are
    assigned the mean score of the sequence.

    Args:
        vel_bank: (M, D) tensor returned by build_velocity_bank.
    """
    ordered_embs = collect_ordered_embeddings(model, test_loader, device)  # (T, D)
    T = ordered_embs.shape[0]

    vel_bank = F.normalize(vel_bank.to(device, dtype=torch.float32), dim=1, eps=1e-12)

    raw_vel    = ordered_embs[delta:] - ordered_embs[:-delta]   # stays on CPU
    velocities = F.normalize(raw_vel, dim=1, eps=1e-12)         # (T-delta, D)
    velocities = torch.nan_to_num(velocities, nan=0.0)

    # Chunked similarity to avoid OOM on long sequences
    chunk_size = 4096
    k_top      = min(top_k, vel_bank.shape[0])
    vel_scores_list = []
    for start in range(0, velocities.shape[0], chunk_size):
        chunk = velocities[start:start + chunk_size].to(device)  # (C, D)
        sims  = chunk @ vel_bank.T                               # (C, M)
        sims  = torch.nan_to_num(sims, nan=-1.0)
        topk_sim, _ = torch.topk(sims, k=k_top, dim=1, largest=True)
        vel_scores_list.append((1.0 - topk_sim.mean(dim=1)).clamp(0.0, 2.0).cpu())
    vel_scores = torch.cat(vel_scores_list, dim=0)               # (T-delta,)

    # Pad last `delta` positions with the sequence mean
    pad    = vel_scores.mean().expand(delta)
    scores = torch.cat([vel_scores, pad], dim=0)            # (T,)

    return scores.cpu().tolist()