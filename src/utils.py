
import matplotlib.pyplot as plt
import ot
import torch
import numpy as np

def plot_distributions(q0, q1, corrected_q, title):
    # Helper function to ensure data is on CPU for plotting
    def to_numpy(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    q0, q1, corrected_q = to_numpy(q0), to_numpy(q1), to_numpy(corrected_q)

    plt.figure(figsize=(15, 5))

    # Plot Source
    plt.subplot(1, 3, 1)
    plt.scatter(q0[:, 0], q0[:, 1], alpha=0.3, s=10, label='Source')
    plt.title("Source (Biased System)")
    plt.xlabel("Position (x)"); plt.ylabel("Velocity (v)")
    plt.legend(); plt.grid(True); plt.xlim(-4, 4); plt.ylim(-4, 4)

    # Plot Target
    plt.subplot(1, 3, 2)
    plt.scatter(q1[:, 0], q1[:, 1], alpha=0.3, s=10, c='g', label='Target')
    plt.title("Target (Clean System)")
    plt.xlabel("Position (x)"); plt.ylabel("Velocity (v)")
    plt.legend(); plt.grid(True); plt.xlim(-4, 4); plt.ylim(-4, 4)

    # Plot Corrected
    plt.subplot(1, 3, 3)
    plt.scatter(q1[:, 0], q1[:, 1], alpha=0.1, s=10, c='g') # Show target in background
    plt.scatter(corrected_q[:, 0], corrected_q[:, 1], alpha=0.4, s=10, c='r', label='Corrected')
    plt.title(f"Corrected Output ({title})")
    plt.xlabel("Position (x)"); plt.ylabel("Velocity (v)")
    plt.legend(); plt.grid(True); plt.xlim(-4, 4); plt.ylim(-4, 4)

    plt.suptitle(f'Phase Space Comparison: {title}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def calculate_w2_distance(dist1, dist2):
    # Helper function to ensure data is on CPU for calculation
    def to_numpy(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    dist1, dist2 = to_numpy(dist1), to_numpy(dist2)

    # Compute cost matrix
    M = ot.dist(dist1, dist2)
    # Compute OT plan
    a, b = ot.unif(dist1.shape[0]), ot.unif(dist2.shape[0])

    # Return Wasserstein-2 distance
    return ot.emd2(a, b, M)
