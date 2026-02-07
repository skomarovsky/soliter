#!/usr/bin/env python3
"""
Diagnostic script for Fisher Information Matrix and EWC.

Run this to verify:
1. How many parameters are tracked by Fisher
2. Why EWC loss might be zero
3. Fisher accumulation vs decay behavior

Usage:
    python scripts/diagnose_fisher_ewc.py
"""

import torch
import numpy as np
from soliter.core.cfc_network import CfCBrain
from soliter.agents.soliter_agent import SoliterAgent, VitalsConfig
from soliter.environment import World, create_default_resources, Physics, SensorSystem
from soliter.memory import ReplayBuffer, FisherInformationMatrix, Transition
from soliter.training import EWCLoss, SleepWakeTrainer, TrainingConfig


def diagnose_fisher_parameters():
    """Check what parameters Fisher is tracking."""
    print("=" * 60)
    print("DIAGNOSIS 1: Fisher Parameter Tracking")
    print("=" * 60)
    
    device = torch.device('cpu')
    brain = CfCBrain()
    fisher = FisherInformationMatrix(brain, device)
    
    print("\n[A] All CfCBrain parameters:")
    total_params = 0
    param_count = 0
    for name, param in brain.named_parameters():
        print(f"  {name}: {list(param.shape)}, requires_grad={param.requires_grad}")
        total_params += param.numel()
        param_count += 1
    print(f"\n  Total tensors: {param_count}")
    print(f"  Total parameters: {total_params:,}")
    
    print("\n[B] Parameters in Fisher diagonal:")
    fisher_count = 0
    fisher_params = 0
    for name, diag in fisher.fisher_diagonal.items():
        print(f"  {name}: {list(diag.shape)}")
        fisher_count += 1
        fisher_params += diag.numel()
    print(f"\n  Fisher tensors tracked: {fisher_count}")
    print(f"  Fisher parameters tracked: {fisher_params:,}")
    
    # Check if all brain params are in Fisher
    print("\n[C] Missing from Fisher:")
    missing = []
    for name, param in brain.named_parameters():
        if name not in fisher.fisher_diagonal:
            missing.append(name)
            print(f"  MISSING: {name}")
    if not missing:
        print("  None - all parameters tracked ✓")
    
    return brain, fisher


def diagnose_ewc_loss():
    """Check why EWC loss might be zero."""
    print("\n" + "=" * 60)
    print("DIAGNOSIS 2: EWC Loss Behavior")
    print("=" * 60)
    
    device = torch.device('cpu')
    brain = CfCBrain()
    fisher = FisherInformationMatrix(brain, device)
    
    # Check the API signature of get_ewc_loss
    print("\n[A] Checking FisherInformationMatrix.get_ewc_loss() signature:")
    import inspect
    sig = inspect.signature(fisher.get_ewc_loss)
    print(f"  Signature: get_ewc_loss{sig}")
    
    # Initial state - compute EWC loss directly from Fisher
    print("\n[B] Initial EWC loss (before any training):")
    try:
        # Try the 2-arg version (model only, lambda built-in or default)
        ewc_loss = fisher.get_ewc_loss(brain)
        print(f"  EWC loss (2-arg call): {ewc_loss:.6f}")
    except TypeError as e:
        print(f"  Error with 2-arg call: {e}")
        try:
            # Try 3-arg version (model, lambda)
            ewc_loss = fisher.get_ewc_loss(brain, 155000.0)
            print(f"  EWC loss (3-arg call): {ewc_loss:.6f}")
        except Exception as e2:
            print(f"  Error with 3-arg call: {e2}")
    
    # Compute Fisher (simulated)
    print("\n[C] Computing Fisher from random data...")
    dataloader = [(torch.randn(32, 41),) for _ in range(10)]
    fisher.compute_fisher(brain, dataloader, num_samples=320)
    
    # Update optimal weights
    fisher.update_optimal_weights(brain)
    
    print(f"  Fisher mean: {fisher.get_stats()['mean_fisher']:.6f}")
    print(f"  Fisher max:  {fisher.get_stats()['max_fisher']:.6f}")
    
    # Check EWC loss now (should still be ~0 since weights haven't moved)
    print("\n[D] EWC loss after Fisher computation (weights unchanged):")
    try:
        ewc_loss = fisher.get_ewc_loss(brain)
        print(f"  EWC loss: {ewc_loss:.6f}")
    except TypeError:
        ewc_loss = fisher.get_ewc_loss(brain, 155000.0)
        print(f"  EWC loss: {ewc_loss:.6f}")
    print("  (Expected: ~0 because weights == optimal_weights)")
    
    # Perturb weights
    print("\n[E] Perturbing weights and checking EWC loss...")
    with torch.no_grad():
        for name, param in brain.named_parameters():
            param.add_(0.01 * torch.randn_like(param))
    
    try:
        ewc_loss_perturbed = fisher.get_ewc_loss(brain)
    except TypeError:
        ewc_loss_perturbed = fisher.get_ewc_loss(brain, 155000.0)
    
    print(f"  EWC loss after perturbation: {ewc_loss_perturbed:.6f}")
    if ewc_loss_perturbed > 0.001:
        print("  ✓ EWC is working - loss increased after weight perturbation")
    else:
        print("  ✗ WARNING: EWC loss still near zero after perturbation!")
        print("    Check if Fisher diagonal is all zeros")
        
    # Show per-parameter EWC contribution
    print("\n[F] Per-parameter EWC contribution (top 5):")
    contributions = []
    for name, param in brain.named_parameters():
        if name in fisher.fisher_diagonal and name in fisher.optimal_weights:
            f_diag = fisher.fisher_diagonal[name]
            w_opt = fisher.optimal_weights[name]
            diff = (param - w_opt) ** 2
            contrib = (f_diag * diff).sum().item()
            contributions.append((name, contrib))
    
    contributions.sort(key=lambda x: x[1], reverse=True)
    for name, contrib in contributions[:5]:
        print(f"  {name}: {contrib:.6f}")


def diagnose_fisher_accumulation():
    """Check Fisher accumulation vs decay behavior."""
    print("\n" + "=" * 60)
    print("DIAGNOSIS 3: Fisher Accumulation vs Decay")
    print("=" * 60)
    
    device = torch.device('cpu')
    brain = CfCBrain()
    fisher = FisherInformationMatrix(brain, device)
    
    print("\n[A] Initial Fisher (should be zero):")
    stats = fisher.get_stats()
    print(f"  Mean: {stats['mean_fisher']:.8f}")
    
    # First computation
    print("\n[B] After first Fisher computation:")
    dataloader = [(torch.randn(32, 41),) for _ in range(10)]
    fisher.compute_fisher(brain, dataloader, num_samples=320)
    stats_after_compute = fisher.get_stats()
    print(f"  Mean: {stats_after_compute['mean_fisher']:.8f}")
    
    # Apply decay
    print("\n[C] After decay (factor=0.77):")
    fisher.decay_fisher(0.77)
    stats_after_decay = fisher.get_stats()
    print(f"  Mean: {stats_after_decay['mean_fisher']:.8f}")
    print(f"  Ratio: {stats_after_decay['mean_fisher'] / stats_after_compute['mean_fisher']:.4f}")
    
    # Second computation (simulating next sleep cycle)
    print("\n[D] After second Fisher computation:")
    fisher.compute_fisher(brain, dataloader, num_samples=320)
    stats_after_second = fisher.get_stats()
    print(f"  Mean: {stats_after_second['mean_fisher']:.8f}")
    
    # Check if compute_fisher overwrites or accumulates
    print("\n[E] Accumulation behavior check:")
    if stats_after_second['mean_fisher'] > stats_after_compute['mean_fisher']:
        print("  Fisher is ACCUMULATING (new > first)")
    elif stats_after_second['mean_fisher'] < stats_after_decay['mean_fisher']:
        print("  Fisher is being OVERWRITTEN (new < decayed)")
    else:
        print("  Fisher is being MIXED (between decayed and first)")
    
    # Recommendation
    print("\n[F] Recommendation:")
    print("  Current behavior in your logs shows Fisher DECREASING over time:")
    print("    0.000160 → 0.000001")
    print("  This suggests either:")
    print("    1. compute_fisher() OVERWRITES rather than accumulates")
    print("    2. decay is applied but new Fisher values are tiny")
    print("    3. Brain weights are stabilizing (less gradient variance)")


def diagnose_uncertainty_estimation():
    """Check uncertainty estimation behavior."""
    print("\n" + "=" * 60)
    print("DIAGNOSIS 4: Uncertainty Estimation Sanity")
    print("=" * 60)
    
    device = torch.device('cpu')
    brain = CfCBrain()
    fisher = FisherInformationMatrix(brain, device)
    buffer = ReplayBuffer(capacity=100)
    
    # Add some transitions
    for i in range(50):
        transition = Transition(
            state=torch.randn(41),
            action=torch.randn(3),
            reward=np.random.random(),
            next_state=torch.randn(41),
            done=False,
            tick=i,
        )
        buffer.push(transition)
    
    # Compute Fisher
    dataloader = [(torch.randn(32, 41),) for _ in range(10)]
    fisher.compute_fisher(brain, dataloader, num_samples=320)
    
    print("\n[A] Uncertainty with default perturbation_scale=0.1:")
    buffer.update_uncertainties(brain, num_samples=10, fisher_matrix=fisher, perturbation_scale=0.1)
    stats = buffer.get_stats()
    print(f"  Avg uncertainty: {stats['avg_uncertainty']:.4f}")
    print(f"  Min uncertainty: {stats['min_uncertainty']:.4f}")
    print(f"  Max uncertainty: {stats['max_uncertainty']:.4f}")
    
    print("\n[B] Uncertainty with 2x perturbation_scale=0.2:")
    buffer.update_uncertainties(brain, num_samples=10, fisher_matrix=fisher, perturbation_scale=0.2)
    stats_2x = buffer.get_stats()
    print(f"  Avg uncertainty: {stats_2x['avg_uncertainty']:.4f}")
    print(f"  Min uncertainty: {stats_2x['min_uncertainty']:.4f}")
    print(f"  Max uncertainty: {stats_2x['max_uncertainty']:.4f}")
    
    if stats_2x['avg_uncertainty'] > stats['avg_uncertainty']:
        print("  ✓ Uncertainty increases with larger perturbation (expected)")
    else:
        print("  ✗ WARNING: Uncertainty did not increase with larger perturbation")
    
    print("\n[C] Uncertainty with 0.5x perturbation_scale=0.05:")
    buffer.update_uncertainties(brain, num_samples=10, fisher_matrix=fisher, perturbation_scale=0.05)
    stats_half = buffer.get_stats()
    print(f"  Avg uncertainty: {stats_half['avg_uncertainty']:.4f}")
    
    if stats_half['avg_uncertainty'] < stats['avg_uncertainty']:
        print("  ✓ Uncertainty decreases with smaller perturbation (expected)")
    else:
        print("  ✗ WARNING: Uncertainty did not decrease with smaller perturbation")


def main():
    print("=" * 60)
    print("FISHER & EWC DIAGNOSTIC SUITE")
    print("=" * 60)
    
    brain, fisher = diagnose_fisher_parameters()
    diagnose_ewc_loss()
    diagnose_fisher_accumulation()
    diagnose_uncertainty_estimation()
    
    print("\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    print("""
1. FISHER PARAMETERS:
   - 13 tensors is plausible for CfC cell architecture
   - But verify ValueHead parameters are NOT in Fisher (intentional?)
   - Action: Check if you want Fisher on policy only vs policy+value

2. EWC LOSS = 0:
   - Normal if weights haven't moved from optimal_weights
   - Should become > 0 after perturbation
   - Action: Add EWC perturbation test to your test suite

3. FISHER "ACCUMULATING" → DECREASING:
   - Your logs show 0.000160 → 0.000001 (decreasing!)
   - Check if compute_fisher() OVERWRITES or ACCUMULATES
   - Action: Fix logging label or accumulation logic

4. MIN UNCERTAINTY → 0.0043:
   - Very low values may cause over-aggressive pruning
   - Verify uncertainty responds to perturbation scale changes
   - Action: Consider raising prune_uncertainty_threshold
""")


if __name__ == "__main__":
    main()
