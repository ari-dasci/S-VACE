import torch


def compute_embedding_geometry(model, train_loader, device):
    """Compute spectral geometry diagnostics of the training embedding distribution.

    Returns a dict with:
        eff_rank_pr      -- participation-ratio effective rank / d
        eff_rank_entropy -- entropy effective rank / d
        d                -- embedding dimension
        n_active_dims    -- number of eigenvalues above 1e-10
        var_top1         -- fraction of variance in the leading component
    """
    model.eval()
    all_emb = []

    with torch.no_grad():
        for data, _ in train_loader:
            data = data.to(device, non_blocking=True)
            all_emb.append(model.embedding(data))

    embeddings = torch.cat(all_emb, dim=0).double()   # (N, d)
    N, d = embeddings.shape

    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(centered, full_matrices=False)
    eigvals = (S ** 2) / (N - 1)

    sum_eig = eigvals.sum() + 1e-12

    pr           = (sum_eig ** 2) / ((eigvals ** 2).sum() + 1e-12)
    p            = eigvals / sum_eig
    p            = p[p > 0]
    entropy_rank = torch.exp(-torch.sum(p * torch.log(p)))
    n_active     = int(torch.sum(eigvals > 1e-10).item())
    var_top1     = float((eigvals[0] / sum_eig).item())

    return {
        'eff_rank_pr':      float((pr / d).item()),
        'eff_rank_entropy': float((entropy_rank / d).item()),
        'd':                d,
        'n_active_dims':    n_active,
        'var_top1':         var_top1,
    }
