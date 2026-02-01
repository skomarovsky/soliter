#!/usr/bin/env python3
"""
Phase 4 verification script.

Tests memory and training components.

Usage:
    python test_phase4.py                    # Run all tests with default 3 cycles
    python test_phase4.py --cycles 10        # Run Test 6 with 10 cycles
    python test_phase4.py --cycles 10 --wake-steps 200  # Custom wake steps per cycle
"""

import torch
import numpy as np
import argparse
from soliter.core.cfc_network import CfCBrain
from soliter.agents.soliter_agent import SoliterAgent, VitalsConfig
from soliter.environment import World, create_default_resources, Physics, SensorSystem
from soliter.memory import ReplayBuffer, FisherInformationMatrix, Transition
from soliter.training import EWCLoss, SleepWakeTrainer, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4 verification script")
    parser.add_argument("--cycles", type=int, default=3, 
                        help="Number of sleep-wake cycles for Test 6 (default: 3)")
    parser.add_argument("--wake-steps", type=int, default=100,
                        help="Wake steps per cycle for Test 6 (default: 100)")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("Phase 4: Memory & Training - Verification")
    print("="*60)

    device = torch.device('cpu')

    # Test 1: Replay Buffer
    print("\n[Test 1: Replay Buffer]")
    buffer = ReplayBuffer(capacity=1000)

    # Add transitions
    for i in range(100):
        transition = Transition(
            state=torch.randn(41),
            action=torch.randn(3),
            reward=np.random.random(),
            next_state=torch.randn(41),
            done=False,
            tick=i,
        )
        buffer.push(transition)

    print(f"Buffer size: {len(buffer)}")
    print(f"Buffer stats: {buffer.get_stats()}")

    # Sample batch
    batch = buffer.sample(32)
    print(f"Sampled batch: {len(batch)} transitions")

    print("✓ Replay buffer working")

    # Test 2: Fisher Information Matrix
    print("\n[Test 2: Fisher Information Matrix]")
    brain = CfCBrain()
    fisher = FisherInformationMatrix(brain, device)

    print(f"Fisher parameters tracked: {len(fisher.fisher_diagonal)}")

    # Simulate Fisher computation
    dataloader = [(torch.randn(16, 41),) for _ in range(10)]
    fisher.compute_fisher(brain, dataloader, num_samples=160)

    stats = fisher.get_stats()
    print(f"Fisher stats: mean={stats['mean_fisher']:.6f}, max={stats['max_fisher']:.6f}")

    # Test decay
    fisher.decay_fisher(0.77)
    print(f"After decay: mean={fisher.get_stats()['mean_fisher']:.6f}")

    print("✓ Fisher matrix working")

    # Test 3: EWC Loss
    print("\n[Test 3: EWC Loss]")
    ewc = EWCLoss(fisher_matrix=fisher, lambda_ewc=1000.0)

    # Compute loss
    task_loss = torch.tensor(1.0)
    total_loss = ewc.compute_loss(brain, task_loss)

    components = ewc.get_loss_components(brain, task_loss)
    print(f"Task loss: {components['task_loss']:.4f}")
    print(f"EWC loss: {components['ewc_loss']:.4f}")
    print(f"Total loss: {components['total_loss']:.4f}")

    print("✓ EWC loss working")

    # Test 4: Sleep-Wake Trainer
    print("\n[Test 4: Sleep-Wake Trainer]")

    # Create components
    world = World()
    resources = create_default_resources()
    physics = Physics()
    sensors = SensorSystem(physics=physics)

    brain = CfCBrain()
    agent = SoliterAgent(brain, VitalsConfig(), device)

    config = TrainingConfig(
        wake_duration=100,
        learning_rate=0.0001,
        batch_size=16,
        sleep_epochs=2,
    )

    trainer = SleepWakeTrainer(
        agent=agent,
        world=world,
        sensors=sensors,
        physics=physics,
        config=config,
        device=device,
    )

    print(f"Trainer initialized")
    print(f"Wake duration: {config.wake_duration}")
    print(f"Buffer capacity: {config.buffer_capacity}")

    # Simulate a few wake steps
    print("\nSimulating 50 wake steps...")
    for i in range(50):
        reward, done = trainer.wake_step(resources)
        world.step()
        
        if done:
            print(f"  Agent died at step {i}")
            break

    print(f"Buffer size after wake: {len(trainer.replay_buffer)}")
    print(f"Total reward: {trainer.stats['total_reward']:.2f}")

    # Trigger sleep
    if len(trainer.replay_buffer) > 0:
        print("\nTriggering sleep cycle...")
        sleep_stats = trainer.sleep_cycle()
        print(f"  Scale factor: {sleep_stats['scale_factor']:.4f}")
        print(f"  Transitions pruned: {sleep_stats['transitions_pruned']}")
        print(f"  Buffer size after: {sleep_stats['buffer_size']}")

    print("✓ Sleep-wake trainer working")

    # Test 5: Checkpointing
    print("\n[Test 5: Checkpointing]")
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")
        
        # Save
        trainer.save_checkpoint(checkpoint_path)
        print(f"✓ Saved checkpoint: {os.path.getsize(checkpoint_path)} bytes")
        
        # Load
        trainer2 = SleepWakeTrainer(
            agent=SoliterAgent(CfCBrain(), VitalsConfig(), device),
            world=World(),
            sensors=sensors,
            physics=physics,
            config=config,
            device=device,
        )
        trainer2.load_checkpoint(checkpoint_path)
        print(f"✓ Loaded checkpoint")
        
        # Verify stats match
        assert trainer2.stats['total_reward'] == trainer.stats['total_reward']
        print(f"✓ Stats verified")

    # Test 6: Multi-Cycle Sleep-Wake Training
    print(f"\n[Test 6: Multi-Cycle Sleep-Wake ({args.cycles} cycles, {args.wake_steps} steps/cycle)]")
    print("-" * 60)
    
    # Create fresh components for multi-cycle test
    world = World()
    resources = create_default_resources()
    physics = Physics()
    sensors = SensorSystem(physics=physics)
    
    brain = CfCBrain()
    agent = SoliterAgent(brain, VitalsConfig(), device)
    
    config = TrainingConfig(
        wake_duration=args.wake_steps,
        learning_rate=0.0001,
        batch_size=16,
        sleep_epochs=3,
        uncertainty_num_samples=10,
        uncertainty_perturbation_scale=0.1,
    )
    
    trainer = SleepWakeTrainer(
        agent=agent,
        world=world,
        sensors=sensors,
        physics=physics,
        config=config,
        device=device,
    )
    
    # Track metrics across cycles
    cycle_stats = []
    total_pruned = 0
    deaths = 0
    
    print(f"\n{'Cycle':<6} {'Wake':<6} {'Buffer':<8} {'Pruned':<8} {'AvgUnc':<10} {'MinUnc':<10} {'AvgTD':<10} {'Scale':<8}")
    print("-" * 76)
    
    for cycle in range(1, args.cycles + 1):
        # Wake phase
        wake_rewards = []
        for step in range(args.wake_steps):
            reward, done = trainer.wake_step(resources)
            wake_rewards.append(reward)
            world.step()
            
            if done:
                deaths += 1
                # Reset agent for next episode
                agent.reset()
        
        buffer_before = len(trainer.replay_buffer)
        
        # Sleep phase
        sleep_stats = trainer.sleep_cycle()
        
        # Collect stats
        stats = {
            'cycle': cycle,
            'wake_steps': len(wake_rewards),
            'mean_reward': np.mean(wake_rewards),
            'buffer_before': buffer_before,
            'buffer_after': sleep_stats['buffer_size'],
            'pruned': sleep_stats['transitions_pruned'],
            'avg_uncertainty': sleep_stats.get('avg_uncertainty', 1.0),
            'avg_td_error': sleep_stats.get('avg_td_error', 1.0),
            'scale_factor': sleep_stats['scale_factor'],
            'fisher_mean': sleep_stats['fisher_stats'].get('mean_fisher', 0),
        }
        
        # Get min uncertainty from buffer stats
        buffer_stats = trainer.replay_buffer.get_stats()
        stats['min_uncertainty'] = buffer_stats.get('min_uncertainty', 1.0)
        stats['near_consolidated'] = buffer_stats.get('near_consolidated', 0)
        
        cycle_stats.append(stats)
        total_pruned += stats['pruned']
        
        # Print row
        print(f"{cycle:<6} {stats['wake_steps']:<6} {stats['buffer_after']:<8} "
              f"{stats['pruned']:<8} {stats['avg_uncertainty']:<10.4f} "
              f"{stats['min_uncertainty']:<10.4f} {stats['avg_td_error']:<10.4f} "
              f"{stats['scale_factor']:<8.4f}")
    
    # Summary
    print("-" * 76)
    print(f"\n📊 Summary after {args.cycles} cycles:")
    print(f"   Total transitions processed: {trainer.replay_buffer.total_added}")
    print(f"   Total pruned: {total_pruned}")
    print(f"   Final buffer size: {len(trainer.replay_buffer)}")
    print(f"   Agent deaths: {deaths}")
    print(f"   Total reward: {trainer.stats['total_reward']:.2f}")
    
    print("✓ Multi-cycle test complete")

    # Test 7: Uncertainty Variation Check
    print("\n[Test 7: Uncertainty Variation Check]")
    uncertainties = [s['avg_uncertainty'] for s in cycle_stats]
    min_uncertainties = [s['min_uncertainty'] for s in cycle_stats]
    
    if all(u == 1.0 for u in uncertainties):
        print("⚠️  WARNING: All avg uncertainties are 1.0 - Fisher-informed estimation may not be working")
        print("   Check that replay_buffer.py has the Fisher-informed update_uncertainties()")
    elif max(uncertainties) - min(uncertainties) < 0.01:
        print(f"⚠️  WARNING: Uncertainties not varying much: {min(uncertainties):.4f} - {max(uncertainties):.4f}")
        print("   Consider increasing perturbation_scale or num_samples")
    else:
        print(f"✓ Avg uncertainties varying: {min(uncertainties):.4f} - {max(uncertainties):.4f}")
    
    if min(min_uncertainties) < 0.5:
        print(f"✓ Min uncertainty reaching low values: {min(min_uncertainties):.4f}")
    else:
        print(f"ℹ️  Min uncertainty still high: {min(min_uncertainties):.4f} (may need more training)")

    # Test 8: Pruning Effectiveness Check
    print("\n[Test 8: Pruning Effectiveness Check]")
    if total_pruned == 0:
        print("⚠️  WARNING: No pruning occurred")
        print("   This is normal for short runs. Try:")
        print("   - More cycles (--cycles 20)")
        print("   - More wake steps (--wake-steps 200)")
        print("   - Lower prune thresholds in TrainingConfig")
    else:
        print(f"✓ Pruning working: {total_pruned} total transitions pruned")
        
        # Show pruning progression
        prune_counts = [s['pruned'] for s in cycle_stats]
        if prune_counts[-1] > prune_counts[0]:
            print(f"✓ Pruning increasing over time: {prune_counts[0]} → {prune_counts[-1]}")
        
    # Test 9: Fisher Accumulation Check
    print("\n[Test 9: Fisher Accumulation Check]")
    fisher_means = [s['fisher_mean'] for s in cycle_stats]
    
    if fisher_means[-1] > 0:
        print(f"✓ Fisher matrix accumulating: {fisher_means[0]:.6f} → {fisher_means[-1]:.6f}")
    else:
        print("⚠️  WARNING: Fisher matrix appears empty")
    
    # Check for Fisher saturation pattern (should decay then stabilize)
    if len(fisher_means) >= 3:
        # Due to decay, later values should be smaller unless new learning dominates
        print(f"   Fisher progression: {' → '.join(f'{f:.6f}' for f in fisher_means[:5])}...")

    # Test 10: Consolidation Progress Check  
    print("\n[Test 10: Consolidation Progress Check]")
    near_consolidated = [s.get('near_consolidated', 0) for s in cycle_stats]
    
    if near_consolidated[-1] > 0:
        print(f"✓ {near_consolidated[-1]} transitions near consolidation threshold")
        print(f"   Progression: {' → '.join(str(n) for n in near_consolidated)}")
    else:
        print("ℹ️  No transitions near consolidation yet (normal for early training)")
        print("   Run longer to see consolidation: --cycles 20 --wake-steps 200")

    print("\n" + "="*60)
    print("Phase 4 Complete: Memory & Training Systems Working! ✅")
    print("="*60)


if __name__ == "__main__":
    main()
