"""
=============================================================================
THE \Psi_2 PARADIGM: INDUSTRIAL-GRADE ARCHITECTURAL ABLATION (NVIDIA A100)
=============================================================================
Double-Blind Peer Review Reproducibility Suite

This standalone script contains the full ablation study presented in the 
manuscript. It dynamically generates the training loop, applies the 
Epistemic Closure Operator (\Omega(z)) to isolate the functional subspace, 
and profiles hardware utilization (VRAM and Latency) using asynchronous 
CUDA events.

REQUIREMENTS:
pip install torch numpy matplotlib seaborn

EXECUTION:
python A100_Reproducibility_Suite.py

OUTPUT:
Generates 'A100_Ablation_Results.pdf' and 'A100_Metrics_Summary.txt' in 
the current working directory, replicating the paper's benchmarks.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. HARDWARE & DATA INITIALIZATION
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Scientific Profiling Initialization on: {device.type.upper()}")

if device.type == 'cuda':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"Hardware Detected: {torch.cuda.get_device_name(0)}")

# Paper Parameters
N_SAMPLES, D_MODEL, D_FUNC, SEQ_LEN, BATCH_SIZE, EPOCHS = 10000, 8192, 4096, 64, 16, 15
LAMBDA_SEMANTIC = 0.5   

torch.manual_seed(42) 
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print("\nGenerating PCOF tensors (BFloat16)...")
X = torch.randn(N_SAMPLES, SEQ_LEN, D_MODEL, dtype=torch.bfloat16, device=device)
true_func_features = X[:, :, :D_FUNC].mean(dim=1).to(torch.float32)
W_true = torch.randn(D_FUNC, 1, dtype=torch.float32, device=device)
y = torch.matmul(true_func_features, W_true).to(torch.bfloat16)

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 2. ARCHITECTURAL DEFINITIONS
# ==========================================
class Psi1a_Aligned_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=32, batch_first=True, dim_feedforward=D_MODEL*4, norm_first=True), num_layers=4)
        self.functional_head = nn.Linear(D_MODEL, 1)
        self.moral_guardrail_head = nn.Linear(D_MODEL, D_MODEL - D_FUNC)
    def forward(self, x):
        pooled = self.encoder(x).mean(dim=1)
        return self.functional_head(pooled), self.moral_guardrail_head(pooled)

class Psi2_Native_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=D_FUNC, nhead=32, batch_first=True, dim_feedforward=D_FUNC*4, norm_first=True), num_layers=4)
        self.functional_head = nn.Linear(D_FUNC, 1)
    def forward(self, x):
        # Epistemic Closure Operator: \Omega(z) Orthogonal Truncation
        pooled = self.encoder(x[:, :, :D_FUNC]).mean(dim=1)
        return self.functional_head(pooled)

# ==========================================
# 3. ABLATION TRAINING LOOP & PROFILING
# ==========================================
def run_referee_proof_ablation(model, name, is_psi1a=False):
    print(f"--------------------------------------------------\nStarting Training: {name}")
    model = model.to(device=device, dtype=torch.bfloat16)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4) 
    criterion = nn.MSELoss()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    loss_history = []
    
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
    else:
        import time
        start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad(set_to_none=True)
            if is_psi1a:
                logical_pred, moral_pred = model(batch_X)
                loss_logic = criterion(logical_pred.float(), batch_y.float())
                loss_semantic = criterion(moral_pred.float(), torch.zeros_like(moral_pred).float())
                loss = loss_logic + (LAMBDA_SEMANTIC * loss_semantic)
            else:
                loss = criterion(model(batch_X).float(), batch_y.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(dataloader))
    
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize() 
        latency_sec = start_event.elapsed_time(end_event) / 1000.0
        max_vram_gb = torch.cuda.max_memory_allocated() / (1024**3) 
    else:
        latency_sec = time.time() - start_time
        max_vram_gb = 0.0

    print(f" > Net Latency: {latency_sec:.2f}s | Peak VRAM: {max_vram_gb:.2f} GB | PCOF Error (MSE): {loss_history[-1]:.4f}")
    return loss_history, latency_sec, max_vram_gb

# ==========================================
# 4. EXECUTION
# ==========================================
model_1a = Psi1a_Aligned_Model()
history_1a, time_1a, vram_1a = run_referee_proof_ablation(model_1a, r"Baseline \Psi_{1a} (RLHF Dual-Objective)", is_psi1a=True)
del model_1a
if torch.cuda.is_available():
    torch.cuda.empty_cache()

model_2 = Psi2_Native_Model()
history_2, time_2, vram_2 = run_referee_proof_ablation(model_2, r"Native \Psi_2 (\Omega(z) Single-Objective)", is_psi1a=False)

# ==========================================
# 5. LOCAL RESULTS GENERATION
# ==========================================
print("\nGenerating charts and textual report...")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300) 
fig.suptitle(r'Architectural Ablation on NVIDIA: Thermodynamic Cost of Semantic Interference ($\Psi_{1a}$ vs $\Psi_2$)', fontsize=16, fontweight='bold', y=1.05)

axes[0].plot(history_1a, label=r'Baseline $\Psi_{1a}$', color='#d9534f', linewidth=3, marker='o')
axes[0].plot(history_2, label=r'Native $\Psi_2$', color='#5cb85c', linewidth=3, marker='s')
axes[0].set_title('PCOF Gradient Convergence', fontweight='bold')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Mean Squared Error (MSE)')
axes[0].legend()

axes[1].bar([r'Baseline $\Psi_{1a}$', r'Native $\Psi_2$'], [vram_1a, vram_2], color=['#d9534f', '#5cb85c'], edgecolor='black')
axes[1].set_title('Peak Hardware Allocation (VRAM)', fontweight='bold')
axes[1].set_ylabel('Gigabytes (GB)')
for i, v in enumerate([vram_1a, vram_2]):
    axes[1].text(i, v + (v*0.02), f"{v:.2f} GB", ha='center', fontweight='bold', fontsize=12)

axes[2].bar([r'Baseline $\Psi_{1a}$', r'Native $\Psi_2$'], [time_1a, time_2], color=['#d9534f', '#5cb85c'], edgecolor='black')
axes[2].set_title('Thermodynamic Latency', fontweight='bold')
axes[2].set_ylabel('Seconds (s)')
for i, v in enumerate([time_1a, time_2]):
    axes[2].text(i, v + (v*0.02), f"{v:.2f} s", ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig("A100_Ablation_Results.pdf", format="pdf", bbox_inches="tight") 
plt.close() 

mse_gain = ((history_1a[-1] - history_2[-1]) / history_1a[-1]) * 100 if history_1a[-1] > 0 else 0
time_gain = ((time_1a - time_2) / time_1a) * 100 if time_1a > 0 else 0
vram_gain = ((vram_1a - vram_2) / vram_1a) * 100 if vram_1a > 0 else 0

summary_text = f"""====================================================================
             INDUSTRIAL STRATEGIC YIELD METRICS              
====================================================================
Double-Blind Peer Review Data (Anonymous Submission)

--- BASELINE PSI_1a ---
MSE:     {history_1a[-1]:.4f}
Latency: {time_1a:.2f} s
VRAM:    {vram_1a:.2f} GB

--- NATIVE PSI_2 (\Omega(z)) ---
MSE:     {history_2[-1]:.4f}
Latency: {time_2:.2f} s
VRAM:    {vram_2:.2f} GB

--- STRATEGIC YIELD ---
PCOF Precision Target:   +{mse_gain:.2f}%
Computational Speed:     +{time_gain:.2f}% 
VRAM Memory Freed:       +{vram_gain:.2f}%
===================================================================="""

with open("A100_Metrics_Summary.txt", "w") as f:
    f.write(summary_text)

print("\nExecution Completed. Artifacts saved: 'A100_Ablation_Results.pdf' and 'A100_Metrics_Summary.txt'.")
