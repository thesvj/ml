import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Positional Encoding (2D)
# =========================
class CoordinatePositionalEncoding(nn.Module):
    def __init__(self, max_size=30, d_model=256):
        super().__init__()
        assert d_model % 2 == 0
        self.max_size = max_size

        self.row_embed = nn.Embedding(max_size, d_model // 2)
        self.col_embed = nn.Embedding(max_size, d_model // 2)

    def forward(self, batch_size):
        device = self.row_embed.weight.device

        rows = torch.arange(self.max_size, device=device)
        cols = torch.arange(self.max_size, device=device)

        row_idx = rows.view(-1, 1).expand(-1, self.max_size).flatten()
        col_idx = cols.view(1, -1).expand(self.max_size, -1).flatten()

        pos = torch.cat([
            self.row_embed(row_idx),
            self.col_embed(col_idx)
        ], dim=-1)

        return pos.unsqueeze(0).expand(batch_size, -1, -1)


# =========================
# Timestep embedding
# =========================
def timestep_embedding(logsnr, dim):
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=logsnr.device) * (torch.log(torch.tensor(10000.0)) / half)
    )
    args = logsnr[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


# =========================
# Transformer Denoiser
# =========================
class TransformerDenoiser(nn.Module):
    def __init__(
        self,
        vocab_size=11,
        d_model=384,
        nhead=6,
        num_layers=8,
        max_size=30,
        max_tasks=1000,
    ):
        super().__init__()

        self.max_size = max_size
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=10)
        self.pos_encoding = CoordinatePositionalEncoding(max_size, d_model)

        self.task_embedding = nn.Embedding(max_tasks, d_model)

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )

        self.sc_proj = nn.Sequential(
            nn.Linear(10, d_model),
            nn.LayerNorm(d_model)
        )

        self.sc_gate = nn.Parameter(torch.tensor(0.3))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_head = nn.Linear(d_model, 10)

    def forward(
        self,
        xt,
        input_grid,
        task_ids,
        logsnr,
        d4_idx=None,
        color_shift=None,
        masks=None,
        sc_p0=None,
        sc_gain=1.0
    ):
        B = xt.shape[0]

        xt_flat = xt.view(B, -1)
        input_flat = input_grid.view(B, -1)

        xt_emb = self.token_embedding(xt_flat)
        input_emb = self.token_embedding(input_flat)

        pos = self.pos_encoding(B)

        xt_emb = xt_emb + pos
        input_emb = input_emb + pos

        if sc_p0 is not None:
            sc = self.sc_proj(sc_p0.view(B, -1, 10))
            xt_emb = xt_emb + sc_gain * self.sc_gate * sc

        if masks is not None:
            m = masks.view(B, -1, 1).float()
            xt_emb = xt_emb * m

        task_token = self.task_embedding(task_ids).unsqueeze(1)

        time_emb = timestep_embedding(logsnr, self.d_model)
        time_token = self.time_mlp(time_emb).unsqueeze(1)

        seq = torch.cat([
            task_token,
            time_token,
            input_emb,
            xt_emb
        ], dim=1)

        out = self.transformer(seq)

        start = 2 + self.max_size * self.max_size
        out_xt = out[:, start:, :]

        logits = self.output_head(out_xt)
        logits = logits.view(B, self.max_size, self.max_size, 10)

        return logits


# =========================
# Full Diffusion Model
# =========================
class ARCDiffusionModel(nn.Module):
    def __init__(
        self,
        vocab_size=11,
        d_model=384,
        nhead=6,
        num_layers=8,
        max_size=30,
        max_tasks=1000,
        include_size_head=True
    ):
        super().__init__()

        self.max_size = max_size
        self.include_size_head = include_size_head

        self.denoiser = TransformerDenoiser(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_size=max_size,
            max_tasks=max_tasks
        )

        if include_size_head:
            hidden = int(d_model * 0.67)

            self.size_head = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU()
            )

            self.height_head = nn.Linear(hidden, max_size)
            self.width_head = nn.Linear(hidden, max_size)

    def forward(self, *args, **kwargs):
        return self.denoiser(*args, **kwargs)

    def compute_loss(
        self,
        x0,
        input_grid,
        task_ids,
        xt,
        logsnr,
        d4_idx=None,
        color_shift=None,
        heights=None,
        widths=None,
        sc_p0=None,
        sc_gain=1.0
    ):
        logits = self.forward(
            xt=xt,
            input_grid=input_grid,
            task_ids=task_ids,
            logsnr=logsnr,
            d4_idx=d4_idx,
            color_shift=color_shift,
            sc_p0=sc_p0,
            sc_gain=sc_gain
        )

        loss = F.cross_entropy(
            logits.view(-1, 10),
            x0.view(-1),
            ignore_index=10
        )

        pred = logits.argmax(dim=-1)
        acc = (pred == x0).float().mean()

        total_loss = loss
        metrics = {
            "total_loss": total_loss,
            "grid_loss": loss,
            "accuracy": acc
        }

        if self.include_size_head and heights is not None:
            h_logits, w_logits = self.predict_size(input_grid, task_ids)

            h_loss = F.cross_entropy(h_logits, heights - 1)
            w_loss = F.cross_entropy(w_logits, widths - 1)

            size_loss = h_loss + w_loss
            total_loss = total_loss + 0.1 * size_loss

            metrics["size_loss"] = size_loss

        metrics["total_loss"] = total_loss
        return metrics

    def predict_size(self, input_grid, task_ids, d4_idx=None, color_shift=None):
        B = input_grid.shape[0]

        x = input_grid.view(B, -1)
        x = self.denoiser.token_embedding(x)

        pos = self.denoiser.pos_encoding(B)
        x = x + pos

        task = self.denoiser.task_embedding(task_ids).unsqueeze(1)

        seq = torch.cat([task, x], dim=1)
        out = self.denoiser.transformer(seq)

        feats = out[:, 1:, :].mean(dim=1)

        feats = self.size_head(feats)

        h = self.height_head(feats)
        w = self.width_head(feats)

        return h, w

    def predict_sizes(self, input_grid, task_ids, d4_idx=None, color_shift=None):
        h, w = self.predict_size(input_grid, task_ids, d4_idx=d4_idx, color_shift=color_shift)

        h = h.argmax(dim=-1) + 1
        w = w.argmax(dim=-1) + 1

        return h, w