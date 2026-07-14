"""Pure PyTorch replacements for mmcv CUDA ops to avoid ScatterGatherKernel bug."""
import torch
import torch.nn.functional as F

__all__ = ['apply_patches']


def _furthest_point_sample_pure(xyz, npoint):
    B, N, _ = xyz.shape
    device = xyz.device
    fps_idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        fps_idx[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.argmax(distance, -1)

    return fps_idx.to(torch.int32)


def _ball_query_pure(min_radius, max_radius, sample_num, xyz, center_xyz,
                     xyz_batch_cnt=None, center_xyz_batch_cnt=None):
    B, N, _ = xyz.shape
    _, M, _ = center_xyz.shape

    dist = torch.cdist(center_xyz, xyz)
    mask = ((dist >= min_radius) & (dist < max_radius)) | (dist == 0)
    masked_dist = dist.clone()
    masked_dist[~mask] = 1e10

    _, idx = torch.topk(masked_dist, sample_num, dim=-1, largest=False)
    return idx.to(torch.int32)


class _GroupingOperationPure(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, indices, features_batch_cnt=None, indices_batch_cnt=None):
        ctx.save_for_backward(indices)
        ctx.features_shape = features.shape

        B, C, N = features.shape
        _, M, S = indices.shape

        idx_flat = indices.long().reshape(B, 1, -1).expand(-1, C, -1)
        grouped = torch.gather(features, 2, idx_flat)
        return grouped.reshape(B, C, M, S)

    @staticmethod
    def backward(ctx, grad_out):
        indices, = ctx.saved_tensors
        features_shape = ctx.features_shape
        B, C, N = features_shape
        _, M, S = indices.shape

        grad_out_flat = grad_out.reshape(B, C, -1)
        idx_flat = indices.long().reshape(B, 1, -1).expand(-1, C, -1)

        grad_features = torch.zeros(B, C, N + 1, device=grad_out.device, dtype=grad_out.dtype)
        grad_features.scatter_add_(2, idx_flat, grad_out_flat)
        return grad_features[:, :, :N], None, None, None


_grouping_operation_pure = _GroupingOperationPure.apply


def _query_and_group_pure_forward(self, points_xyz, center_xyz, features=None):
    if self.max_radius is None:
        from mmcv.ops.knn import knn
        idx = knn(self.sample_num, points_xyz, center_xyz, False)
        idx = idx.transpose(1, 2).contiguous()
    else:
        idx = _ball_query_pure(self.min_radius, self.max_radius, self.sample_num,
                               points_xyz, center_xyz)

    if features is not None:
        new_features = _grouping_operation_pure(features, idx)
    else:
        new_features = None

    shifted_xyz = points_xyz.unsqueeze(1).expand(-1, center_xyz.shape[1], -1, -1)
    shifted_xyz = torch.gather(shifted_xyz, 2,
                               idx.long().unsqueeze(-1).expand(-1, -1, -1, 3))

    if self.normalize_xyz:
        shifted_xyz = shifted_xyz - center_xyz.unsqueeze(2)

    if self.return_grouped_xyz or self.return_grouped_idx:
        grouped_xyz = shifted_xyz.permute(0, 3, 1, 2)

    results = []
    if self.use_xyz:
        results.append(grouped_xyz if self.return_grouped_xyz else
                       shifted_xyz.permute(0, 3, 1, 2))
    if features is not None:
        results.append(new_features)

    ret = torch.cat(results, dim=1)

    if hasattr(self, 'return_unique_cnt') and self.return_unique_cnt:
        unique_cnt = torch.zeros(idx.shape[0], idx.shape[1], dtype=torch.int32,
                                 device=idx.device)
        for i_batch in range(idx.shape[0]):
            for i_region in range(idx.shape[1]):
                unique_indices = torch.unique(idx[i_batch, i_region])
                num_unique = unique_indices.shape[0]
                unique_cnt[i_batch, i_region] = num_unique
        ret = (ret, unique_cnt)

    if self.return_grouped_idx:
        ret = (ret, idx) if isinstance(ret, tuple) else (ret, idx)

    return ret


def _group_all_pure_forward(self, features, xyz):
    B, C, N = features.shape
    _, C_xyz, _ = xyz.shape
    grouped_xyz = xyz.reshape(B, C_xyz, 1, N)
    grouped_features = features.reshape(B, C, 1, N)
    return torch.cat([grouped_xyz, grouped_features], dim=1)


def _three_nn_pure(target, source):
    """Pure PyTorch 3-nearest neighbor search.

    Args:
        target: (B, N, 3) - query points
        source: (B, M, 3) - reference points

    Returns:
        dist: (B, N, 3) - L2 distances to 3 nearest neighbors
        idx: (B, N, 3) - indices of 3 nearest neighbors (int32)
    """
    dist = torch.cdist(target, source)  # (B, N, M)
    dist_top3, idx_top3 = torch.topk(dist, 3, dim=-1, largest=False)
    return torch.sqrt(dist_top3), idx_top3.to(torch.int32)


class _ThreeInterpolatePure(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, indices, weight):
        B, C, M = features.shape
        B, N, K = indices.shape
        ctx.save_for_backward(features, indices, weight)

        # Gather features at indices: (B, N, K) -> (B, C, N, K)
        # Use advanced indexing: features[b, :, indices[b, n, k]]
        batch_idx = torch.arange(B, device=features.device).view(B, 1, 1)
        gathered = features[batch_idx, :, indices.long()]  # (B, N, K, C)
        gathered = gathered.permute(0, 3, 1, 2)  # (B, C, N, K)

        # Weighted sum: (B, C, N, K) * (B, 1, N, K) -> sum over K
        interpolated = (gathered * weight.unsqueeze(1)).sum(-1)  # (B, C, N)
        return interpolated

    @staticmethod
    def backward(ctx, grad_out):
        features, indices, weight = ctx.saved_tensors
        B, C, M = features.shape
        B, N, K = indices.shape

        # Backward: grad_out (B, C, N) -> grad_features (B, C, M)
        # dL/dF[m] = sum over (n,k) where idx[n,k]=m of grad_out[:,:,n] * w[n,k]
        grad_features = features.new_zeros(B, C, M)
        grad_out_weighted = grad_out.unsqueeze(-1) * weight.unsqueeze(1)  # (B, C, N, K)

        idx_flat = indices.long().view(B, 1, -1)  # (B, 1, N*K)
        grad_weighted_flat = grad_out_weighted.reshape(B, C, -1)  # (B, C, N*K)
        grad_features.scatter_add_(2, idx_flat.expand(-1, C, -1), grad_weighted_flat)

        return grad_features, None, None


_three_interpolate_pure = _ThreeInterpolatePure.apply


def apply_patches():
    import mmcv.ops
    import mmcv.ops.group_points as gp_mod

    mmcv.ops.furthest_point_sample = _furthest_point_sample_pure
    mmcv.ops.ball_query = _ball_query_pure
    mmcv.ops.grouping_operation = _grouping_operation_pure
    mmcv.ops.three_nn = _three_nn_pure
    mmcv.ops.three_interpolate = _three_interpolate_pure
    gp_mod.grouping_operation = _grouping_operation_pure

    gp_mod.QueryAndGroup.forward = _query_and_group_pure_forward
    gp_mod.GroupAll.forward = _group_all_pure_forward

    print('[mmcv_pure_patch] All CUDA ops replaced with pure PyTorch.')
