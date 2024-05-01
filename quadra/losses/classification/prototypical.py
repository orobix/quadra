from __future__ import annotations

import torch
from torch.nn import functional as F


def euclidean_dist(
    query: torch.Tensor,
    prototypes: torch.Tensor,
    sen: bool = True,
    eps_pos: float = 1.0,
    eps_neg: float = -1e-7,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute euclidean distance between two tensors.
        SEN dissimilarity from https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680120.pdf
    Args:
        query: feature of the network
        prototypes: prototypes of the center
        sen: Sen dissimilarity flag
        eps_pos: similarity arg
        eps_neg: similarity arg
        eps: similarity arg.

    Returns:
        Euclidian loss

    """
    # query: (n_classes * n_query) x d
    # prototypes: n_classes x d
    n = query.size(0)
    m = prototypes.size(0)
    d = query.size(1)
    if d != prototypes.size(1):
        raise ValueError("query and prototypes size[1] should be equal")

    if sen:
        # SEN dissimilarity from https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680120.pdf
        norm_query = torch.linalg.norm(query, ord=2, dim=1)  # (n_classes * n_query) X 1
        norm_prototypes = torch.linalg.norm(prototypes, ord=2, dim=1)  # n_classes X 1

        # We have to compute (||z|| - ||c||)^2 between all query points w.r.t.
        # all support points

        # Replicate each single query norm value m times
        norm_query = norm_query.view(-1, 1).unsqueeze(1).expand(n, m, 1)
        # Replicate all prototypes norm values n times
        norm_prototypes = norm_prototypes.view(-1, 1).unsqueeze(0).expand(n, m, 1)
        norm_diff = torch.pow(norm_query - norm_prototypes, 2).squeeze(2)
        epsilon = torch.full((n, m), eps_neg).type_as(query)
        if eps_pos != eps_neg:
            # n_query = n // m
            # for i in range(m):
            #     epsilon[i * n_query : (i + 1) * n_query, i] = 1.0

            # Since query points with class i need to have a positive epsilon
            # whenever they refer to support point with class i and since
            # query and support points are ordered, we need to set:
            # the 1st column of the 1st n_query rows to eps_pos
            # the 2nd column of the 2nd n_query rows to eps_pos
            # and so on
            idxs = torch.eye(m, dtype=torch.bool).unsqueeze(1).expand(m, n // m, m).reshape(-1, m)
            epsilon[idxs] = eps_pos
        norm_diff = norm_diff * epsilon

    # Replicate each single query point value m times
    query = query.unsqueeze(1).expand(n, m, d)
    # Replicate all prototype points values n times
    prototypes = prototypes.unsqueeze(0).expand(n, m, d)

    norm = torch.pow(query - prototypes, 2).sum(2)
    if sen:
        return torch.sqrt(norm + norm_diff + eps)

    return norm


def prototypical_loss(
    coords: torch.Tensor,
    target: torch.Tensor,
    n_support: int,
    prototypes: torch.Tensor | None = None,
    sen: bool = True,
    eps_pos: float = 1.0,
    eps_neg: float = -1e-7,
):
    """Prototypical loss implementation.

    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
        Compute the barycentres by averaging the features of n_support
        samples for each class in target, computes then the distances from each
        samples' features to each one of the barycentres, computes the
        log_probability for each n_query samples for each one of the current
        classes, of appartaining to a class c, loss and accuracy are then computed and returned.

    Args:
        coords: The model output for a batch of samples
        target: Ground truth for the above batch of samples
        n_support: Number of samples to keep in account when computing
            barycentres, for each one of the current classes
        prototypes: if not None, is used for classification
        sen: Sen dissimilarity flag
        eps_pos: Sen positive similarity arg
        eps_neg: Sen negative similarity arg
    """
    classes = torch.unique(target, sorted=True)
    n_classes = len(classes)
    n_query = len(torch.where(target == classes[0])[0]) - n_support

    # Check equality between classes and target with broadcasting:
    # class_idxs[i, j] = True iff classes[i] == target[j]
    class_idxs = classes.unsqueeze(1) == target
    if prototypes is None:
        # Get the prototypes as the mean of the support points,
        # ordered by class
        prototypes = torch.stack([coords[idx_list][:n_support] for idx_list in class_idxs]).mean(1)  # n_classes X d
    # Get query samples as the points NOT in the support set,
    # where, after .view(-1, d), one has that
    # the 1st n_query points refer to class 1
    # the 2nd n_query points refer to class 2
    # and so on
    query_samples = torch.stack([coords[idx_list][n_support:] for idx_list in class_idxs]).view(
        -1, prototypes.shape[-1]
    )  # (n_classes * n_query) X d
    # Get distances, where dists[i, j] is the distance between
    # query point i to support point j
    dists = euclidean_dist(
        query_samples, prototypes, sen=sen, eps_pos=eps_pos, eps_neg=eps_neg
    )  # (n_classes * n_query) X n_classes
    log_p_y = F.log_softmax(-dists, dim=1)
    log_p_y = log_p_y.view(n_classes, n_query, -1)  # n_classes X n_query X n_classes

    target_inds = torch.arange(0, n_classes).view(n_classes, 1, 1)
    # One solution is to use type_as(coords[0])
    target_inds = target_inds.type_as(coords)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    # Since we need to backpropagate the log softmax of query points
    # of class i that refers to support of the same class for every i,
    # and since query and support are ordered we select:
    # from the 1st n_query X n_classes the 1st column
    # from the 2nd n_query X n_classes the 2st column
    # and so on
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

    return loss_val, acc_val, prototypes
