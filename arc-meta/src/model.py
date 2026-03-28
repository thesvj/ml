import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Spatial Embedding
# ==========================================
class GridEmbedder(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        # 11 tokens: 0-9 colors + 10 for padding
        self.embed = nn.Embedding(11, dim)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        """
        Input: (B, 1, H, W) -> Output: (B, H*W, dim)
        """
        B, C, H, W = x.shape
        # Flatten H and W into a sequence
        x = x.view(B, H * W).long()
        x = self.embed(x) 
        return self.proj(x)

# ==========================================
# 2. Hierarchical Reasoning Cell
# ==========================================
class HRMCell(nn.Module):
    def __init__(self, dim=128, T_steps=4):
        super().__init__()
        self.dim = dim
        self.T_steps = T_steps
        
        # Low-Level: Updates local pixel hidden states
        self.L_step = nn.GRUCell(input_size=dim * 2, hidden_size=dim)
        # High-Level: Updates the global plan
        self.H_step = nn.GRUCell(input_size=dim, hidden_size=dim)
        
        self.to_logits = nn.Linear(dim, 10)

    def forward(self, x_tilde, z_L, z_H):
        """
        x_tilde: (B, L, dim) - Embedded query grid
        z_L: (B, L, dim) - Local latent states
        z_H: (B, dim) - Global latent state (the "rule")
        """
        B, L, D = x_tilde.shape
        
        # --- Internal Simulation (Local Refinement) ---
        for _ in range(self.T_steps):
            # Broadcast Global State to every pixel in the sequence
            # (B, D) -> (B, L, D) -> (B*L, D)
            z_H_expanded = z_H.unsqueeze(1).expand(-1, L, -1).contiguous().view(B * L, D)
            
            x_flat = x_tilde.view(B * L, D)
            z_L_flat = z_L.view(B * L, D)
            
            # Concatenate local features with global context
            l_input = torch.cat([x_flat, z_H_expanded], dim=-1)
            
            # Update local state
            z_L_flat = self.L_step(l_input, z_L_flat)
            z_L = z_L_flat.view(B, L, D)
        
        # --- Global Planning Update ---
        # Average local insights to update the global rule/strategy
        z_L_pooled = z_L.mean(dim=1) 
        z_H_new = self.H_step(z_L_pooled, z_H)
        
        # Output color predictions for each pixel
        logits = self.to_logits(z_L) 
        
        return z_L, z_H_new, logits

# ==========================================
# 3. ARC-HRM Integration
# ==========================================
class ARCFewShotHRM(nn.Module):
    def __init__(self, dim=128, T_steps=4, max_segments=8):
        super().__init__()
        self.dim = dim
        self.max_segments = max_segments
        
        self.embedder = GridEmbedder(dim)
        self.hrm_cell = HRMCell(dim, T_steps)
        
        self.rule_compressor = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def extract_rule(self, support_x, support_y):
        """
        Compresses support pairs into a single task-level vector.
        Handles variable number of support examples by mean-pooling.
        """
        # Embed and pool across spatial and example dimensions
        sx_emb = self.embedder(support_x).mean(dim=[0, 1]) 
        sy_emb = self.embedder(support_y).mean(dim=[0, 1]) 
        
        rule_context = torch.cat([sx_emb, sy_emb], dim=-1)
        # Returns (1, dim)
        return self.rule_compressor(rule_context).unsqueeze(0)

    def forward(self, support_x, support_y, query_x, return_all_segments=False):
        B_q, _, H_q, W_q = query_x.shape
        
        # 1. Extract the rule (latent representation of the transformation)
        z_H_rule = self.extract_rule(support_x, support_y) # (1, dim)
        
        # 2. Embed the Query Grid
        q_tilde = self.embedder(query_x) # (B_q, L, dim)
        seq_len = H_q * W_q
        
        # 3. Initialize Latents
        z_L = torch.zeros(B_q, seq_len, self.dim, device=query_x.device)
        # Broadcast the single rule vector to match the query batch size
        z_H = z_H_rule.expand(B_q, -1) 
        
        segment_logits = []
        
        # 4. Multi-step reasoning "Segments"
        for m in range(self.max_segments):
            z_L, z_H, logits = self.hrm_cell(q_tilde, z_L, z_H)
            
            # Reshape logits to (Batch, Colors, Height, Width)
            pred_grid = logits.transpose(1, 2).view(B_q, 10, H_q, W_q)
            segment_logits.append(pred_grid)
            
        return segment_logits if return_all_segments else segment_logits[-1]