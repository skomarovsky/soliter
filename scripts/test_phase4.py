#!/usr/bin/env python3
"""
Phase 4 verification script.

Tests memory and training components.
"""

import torch
import numpy as np
from soliter.core.cfc_network import CfCBrain
from soliter.agents.soliter_agent import SoliterAgent, VitalsConfig
from soliter.environment import World, create_default_resources, Physics, SensorSystem
from soliter.memory import ReplayBuffer, FisherInformationMatrix, Transition
from soliter.training import EWCLoss, SleepWakeTrainer, TrainingConfig

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

print("\n✓ Sleep-wake trainer working")

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

print("\n" + "="*60)
print("Phase 4 Complete: Memory & Training Systems Working! ✅")
print("="*60)
