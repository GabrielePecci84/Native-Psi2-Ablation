# ==============================================================================
# REPRODUCIBILITY SCRIPT: Native \Psi_2 Architectural Ablation
# Target Hardware: NVIDIA Ampere (A100) or Hopper (H100)
# Note: Designed for Double-Blind Peer Review. 
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

N_SAMPLES, D_MODEL, D_FUNC, SEQ_LEN, BATCH_SIZE, EPOCHS = 10000, 8192, 4096, 64, 16, 15

torch.manual_seed(42)
X = torch.randn(N_SAMPLES, SEQ_LEN, D_MODEL, dtype=torch.bfloat16, device=device)
true_features = X[:, :, :D_FUNC].mean(dim=1).to(torch.float32)
W = torch.randn(D_FUNC, 1, dtype=torch.float32, device=device)
y = torch.matmul(true_features, W).to(torch.bfloat16)
dataloader = DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)

class Psi1a_Baseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=32, batch_first=True, norm_first=True), num_layers=4)
        self.functional_head = nn.Linear(D_MODEL, 1)
        self.moral_head = nn.Linear(D_MODEL, D_MODEL - D_FUNC)
    def forward(self, x):
        pooled = self.encoder(x).mean(dim=1)
        return self.functional_head(pooled), self.moral_head(pooled)

class Psi2_Native(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=D_FUNC, nhead=32, batch_first=True, norm_first=True), num_layers=4)
        self.functional_head = nn.Linear(D_FUNC, 1)
    def forward(self, x):
        # The Epistemic Closure Operator: \Omega(z) Orthogonal Truncation
        return self.functional_head(self.encoder(x[:, :, :D_FUNC]).mean(dim=1)) 
