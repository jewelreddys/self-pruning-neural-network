"""
Self-Pruning Neural Network for CIFAR-10 Classification
Tredence Analytics – AI Engineering Intern Case Study
Author: Nithin Joel J
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(42)

# ─────────────────────────────────────────────
# PART 1: PrunableLinear Layer
# ─────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A custom linear layer augmented with learnable gate_scores.
    Each weight has a corresponding gate (via sigmoid) that can shrink to 0,
    effectively pruning that connection from the network.

    Forward pass:
        gates       = sigmoid(gate_scores)          ∈ (0, 1)
        pruned_w    = weight ⊙ gates                (element-wise)
        output      = input @ pruned_w.T + bias
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight and bias – identical to nn.Linear
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores – same shape as weight
        # Initialized near 0.5 after sigmoid so gates start ~open
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming init for weight (good for ReLU activations)
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert raw scores → gates ∈ (0, 1) via sigmoid
        gates = torch.sigmoid(self.gate_scores)

        # Gated (pruned) weights
        pruned_weights = self.weight * gates          # element-wise multiply

        # Standard affine transform – gradients flow through both weight and gate_scores
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Returns the current gate values (detached from graph)."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction of gates below threshold (considered pruned)."""
        gates = self.get_gates()
        return (gates < threshold).float().mean().item()


# ─────────────────────────────────────────────
# Network Definition
# ─────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    Feed-forward network for CIFAR-10 using PrunableLinear layers.
    Architecture:  3072 → 512 → 256 → 128 → 10
    """

    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 128)
        self.fc4 = PrunableLinear(128, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)           # flatten CIFAR image
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def prunable_layers(self):
        return [self.fc1, self.fc2, self.fc3, self.fc4]

    def all_gates(self) -> torch.Tensor:
        """Concatenate all gate tensors into a single 1D tensor."""
        return torch.cat([torch.sigmoid(l.gate_scores).detach().view(-1)
                          for l in self.prunable_layers()])

    def network_sparsity(self, threshold: float = 1e-2) -> float:
        """Overall fraction of pruned weights across the entire network."""
        gates = self.all_gates()
        return (gates < threshold).float().mean().item()


# ─────────────────────────────────────────────
# PART 2: Sparsity Regularisation Loss
# ─────────────────────────────────────────────

def sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """
    L1 norm of all gate values across every PrunableLinear layer.
    Minimising this term drives gates toward 0 (pruning).
    Since gates = sigmoid(.) > 0, |gate| = gate, so L1 = sum(gates).
    """
    total = torch.tensor(0.0, requires_grad=True)
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)   # keep in graph for grad
        total = total + gates.sum()
    return total


# ─────────────────────────────────────────────
# PART 3: Data Loading
# ─────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, lambda_sparse, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        # Total Loss = Cross-Entropy + λ * L1(gates)
        cls_loss   = F.cross_entropy(outputs, labels)
        spar_loss  = sparsity_loss(model)
        total_loss = cls_loss + lambda_sparse * spar_loss

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
    return running_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
    return 100.0 * correct / total


# ─────────────────────────────────────────────
# Main Experiment
# ─────────────────────────────────────────────

def run_experiment(lambda_val: float, epochs: int, train_loader, test_loader, device):
    print(f"\n{'='*55}")
    print(f"  Running experiment: λ = {lambda_val}")
    print(f"{'='*55}")

    model     = SelfPruningNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, lambda_val, device)
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            acc      = evaluate(model, test_loader, device)
            sparsity = model.network_sparsity()
            print(f"  Epoch {epoch:3d}/{epochs}  |  Loss: {loss:.4f}"
                  f"  |  Test Acc: {acc:.2f}%  |  Sparsity: {sparsity*100:.1f}%")

    final_acc      = evaluate(model, test_loader, device)
    final_sparsity = model.network_sparsity()
    gate_values    = model.all_gates().numpy()

    print(f"\n  ── Final Results ──")
    print(f"  Test Accuracy : {final_acc:.2f}%")
    print(f"  Sparsity Level: {final_sparsity*100:.2f}%")

    return final_acc, final_sparsity, gate_values, model


def plot_gate_distribution(gate_values: np.ndarray, lambda_val: float, ax):
    """Histogram of gate values for one experiment."""
    ax.hist(gate_values, bins=80, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.set_title(f'Gate Distribution  (λ={lambda_val})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Gate Value', fontsize=9)
    ax.set_ylabel('Count',      fontsize=9)
    ax.axvline(x=0.01, color='red', linestyle='--', linewidth=1.2, label='Prune threshold')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    EPOCHS     = 30          # increase for better accuracy; 30 is a solid baseline
    BATCH_SIZE = 128

    # ── Three λ values covering low / medium / high sparsity ──
    lambdas = [1e-4, 1e-3, 5e-3]

    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)

    results = {}   # lambda → (accuracy, sparsity, gate_values)

    for lam in lambdas:
        acc, sparsity, gate_vals, _ = run_experiment(
            lam, EPOCHS, train_loader, test_loader, device
        )
        results[lam] = (acc, sparsity, gate_vals)

    # ── Summary Table ──
    print("\n" + "="*55)
    print("  RESULTS SUMMARY")
    print("="*55)
    print(f"  {'Lambda':<12} {'Test Accuracy':>15} {'Sparsity Level':>16}")
    print(f"  {'-'*43}")
    for lam, (acc, sparsity, _) in results.items():
        print(f"  {lam:<12} {acc:>14.2f}%  {sparsity*100:>14.2f}%")
    print("="*55)

    # ── Gate Distribution Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Gate Value Distributions for Different λ Values',
                 fontsize=13, fontweight='bold')

    for ax, lam in zip(axes, lambdas):
        plot_gate_distribution(results[lam][2], lam, ax)

    plt.tight_layout()
    plt.savefig('gate_distributions.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved → gate_distributions.png")
    plt.show()


if __name__ == '__main__':
    main()
