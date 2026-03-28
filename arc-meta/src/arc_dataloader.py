import torch
from torch.utils.data import DataLoader, Sampler
import random

class ARCAugmenter:
    @staticmethod
    def permute_colors(tensors):
        perm = torch.randperm(9) + 1
        mapping = torch.cat([torch.tensor([0]), perm])
        return [mapping[t.long()] for t in tensors]

    @staticmethod
    def apply_d4_symmetry(tensors, transform_idx):
        out = []
        for t in tensors:
            if transform_idx == 0: res = t
            elif transform_idx == 1: res = torch.rot90(t, k=1, dims=(-2, -1))
            elif transform_idx == 2: res = torch.rot90(t, k=2, dims=(-2, -1))
            elif transform_idx == 3: res = torch.rot90(t, k=3, dims=(-2, -1))
            elif transform_idx == 4: res = torch.flip(t, dims=[-1])
            elif transform_idx == 5: res = torch.flip(t, dims=[-2])
            elif transform_idx == 6: res = torch.transpose(t, -2, -1)
            elif transform_idx == 7: res = torch.transpose(torch.rot90(t, k=2, dims=(-2, -1)), -2, -1)
            out.append(res)
        return out

class ExactBucketSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.buckets = {}
        for i, item in enumerate(dataset):
            key = (item["support_x"].shape[-2], item["support_x"].shape[-1]) 
            self.buckets.setdefault(key, []).append(i)

        self.batches = []
        for bucket in self.buckets.values():
            random.shuffle(bucket)
            for i in range(0, len(bucket), batch_size):
                self.batches.append(bucket[i:i + batch_size])
        random.shuffle(self.batches)

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)

def pad_to_max(tensors, max_h, max_w, pad_value=10):
    out = []
    for t in tensors:
        _, h, w = t.shape
        t = torch.nn.functional.pad(t, (0, max_w - w, 0, max_h - h), value=pad_value)
        out.append(t)
    return torch.stack(out)

def augmented_collate_fn(batch):
    B = len(batch)
    max_s = max(b["support_x"].shape[0] for b in batch)
    max_q = max(b["query_x"].shape[0] for b in batch)

    base_h, base_w = batch[0]["support_x"].shape[-2:]
    transform_idx = random.randint(0, 7)
    apply_color_perm = random.random() < 0.5
    
    swap_dims = transform_idx in [1, 3, 6, 7]
    out_h, out_w = (base_w, base_h) if swap_dims else (base_h, base_w)

    support_x = torch.full((B, max_s, 1, out_h, out_w), 10.0)
    support_y = torch.full((B, max_s, 1, out_h, out_w), 10.0)
    query_x = torch.full((B, max_q, 1, out_h, out_w), 10.0)
    query_y = torch.full((B, max_q, 1, out_h, out_w), 10.0)

    support_mask, query_mask = torch.zeros(B, max_s), torch.zeros(B, max_q)

    for i, b in enumerate(batch):
        s, q = b["support_x"].shape[0], b["query_x"].shape[0]
        sx, sy = b["support_x"], b["support_y"]
        qx, qy = b["query_x"], b["query_y"]

        sx, sy, qx, qy = ARCAugmenter.apply_d4_symmetry([sx, sy, qx, qy], transform_idx)
        if apply_color_perm:
            sx, sy, qx, qy = ARCAugmenter.permute_colors([sx, sy, qx, qy])

        support_x[i, :s], support_y[i, :s] = sx, sy
        query_x[i, :q], query_y[i, :q] = qx, qy
        support_mask[i, :s], query_mask[i, :q] = 1, 1

    return {
        "support_x": support_x, "support_y": support_y,
        "query_x": query_x, "query_y": query_y,
        "support_mask": support_mask, "query_mask": query_mask,
        "actual_batch_size": B
    }

def get_arc_loader(dataset, batch_size=16):
    sampler = ExactBucketSampler(dataset, batch_size)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=augmented_collate_fn, num_workers=4)