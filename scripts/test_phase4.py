#!/usr/bin/env python3
"""
Phase 4 Verification: Replay-Driven Pruning + Detailed Vitals.

Tests:
- Synthetic: buffer basics, prune_by_surprise ranking, min_buffer_size
- Integration: multi-cycle training with vitals, deaths, surprise tracking

Usage:
    python test_phase4.py                              # Quick (3 cycles)
    python test_phase4.py --cycles 50 --wake-steps 500 # Full experiment
"""

import argparse
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict
from collections import defaultdict

from soliter.core.cfc_network import CfCBrain
from soliter.agents.soliter_agent import SoliterAgent, VitalsConfig
from soliter.environment import World, create_default_resources, Physics, SensorSystem
from soliter.memory import FisherInformationMatrix, Transition
from soliter.memory.replay_buffer import EpistemicReplayBuffer, PruningStats
from soliter.training import EWCLoss, SleepWakeTrainer, TrainingConfig


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class DeathRecord:
    cycle: int
    tick: int
    cause: str
    energy_at_death: float
    hydration_at_death: float
    temperature_at_death: float
    wakefulness_at_death: float


@dataclass
class CycleStats:
    cycle: int
    # Vitals
    energy_min: float = 100.0
    energy_max: float = 0.0
    energy_avg: float = 0.0
    hydration_min: float = 100.0
    hydration_avg: float = 0.0
    temperature_min: float = 100.0
    temperature_avg: float = 0.0
    wakefulness_avg: float = 0.0
    # Deaths
    deaths: int = 0
    death_causes: List[str] = field(default_factory=list)
    # Rewards
    reward_total: float = 0.0
    # Replay surprise
    surprise_min: float = 0.0
    surprise_mean: float = 0.0
    surprise_max: float = 0.0
    surprise_cutoff: float = 0.0
    # Buffer
    buffer_size: int = 0
    pruned: int = 0
    # PPO
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0


class DetailedTracker:
    def __init__(self):
        self.all_deaths: List[DeathRecord] = []
        self.cycle_stats: List[CycleStats] = []
        self._e: List[float] = []
        self._h: List[float] = []
        self._t: List[float] = []
        self._w: List[float] = []
        self._r: List[float] = []
        self._cycle_deaths: List[DeathRecord] = []
        self._cycle: int = 0

    def start_cycle(self, cycle: int):
        self._cycle = cycle
        self._e.clear(); self._h.clear()
        self._t.clear(); self._w.clear()
        self._r.clear(); self._cycle_deaths.clear()

    def record_step(self, agent, reward):
        self._e.append(agent.energy)
        self._h.append(agent.hydration)
        self._t.append(agent.temperature)
        self._w.append(agent.wakefulness)
        self._r.append(reward)

    def record_death(self, agent, tick):
        cause = getattr(agent, 'cause_of_death', None) or 'unknown'
        d = DeathRecord(self._cycle, tick, cause, agent.energy,
                        agent.hydration, agent.temperature, agent.wakefulness)
        self.all_deaths.append(d)
        self._cycle_deaths.append(d)

    def end_cycle(self, **kw) -> CycleStats:
        e = self._e or [0.]; h = self._h or [0.]
        t = self._t or [0.]; w = self._w or [0.]
        r = self._r or [0.]
        cs = CycleStats(
            cycle=self._cycle,
            energy_min=min(e), energy_max=max(e), energy_avg=float(np.mean(e)),
            hydration_min=min(h), hydration_avg=float(np.mean(h)),
            temperature_min=min(t), temperature_avg=float(np.mean(t)),
            wakefulness_avg=float(np.mean(w)),
            deaths=len(self._cycle_deaths),
            death_causes=[d.cause for d in self._cycle_deaths],
            reward_total=float(sum(r)),
            **kw,
        )
        self.cycle_stats.append(cs)
        return cs

    def print_summary(self):
        # Deaths
        total_d = len(self.all_deaths)
        causes = defaultdict(int)
        for d in self.all_deaths:
            causes[d.cause] += 1
        print(f"\n{'─'*55}")
        print(f"  DEATHS: {total_d}")
        for c, n in sorted(causes.items(), key=lambda x: -x[1]):
            print(f"    {c:<14} {n:>4} ({100*n/max(1,total_d):.0f}%)")

        n = len(self.cycle_stats)
        if n >= 6:
            t1 = sum(s.deaths for s in self.cycle_stats[:n//3])
            t3 = sum(s.deaths for s in self.cycle_stats[2*n//3:])
            trend = '↓ learning' if t3 < t1 else ('→ stable' if t3 == t1 else '↑ investigate')
            print(f"  Trend: early={t1} late={t3} {trend}")

        # Vitals
        if self.cycle_stats:
            first = self.cycle_stats[:max(1,n//5)]
            last = self.cycle_stats[-max(1,n//5):]
            print(f"\n{'─'*55}")
            print(f"  VITALS (early → late)")
            for name, attr in [('Energy','energy_avg'),('Hydration','hydration_avg'),
                               ('Temperature','temperature_avg')]:
                e = np.mean([getattr(s,attr) for s in first])
                l = np.mean([getattr(s,attr) for s in last])
                a = '↑' if l>e else '↓' if l<e else '→'
                print(f"    {name:<14} {e:5.1f} → {l:5.1f} {a}")

        # Surprise & pruning
        if len(self.cycle_stats) >= 4:
            first = self.cycle_stats[:max(1,n//5)]
            last = self.cycle_stats[-max(1,n//5):]
            print(f"\n{'─'*55}")
            print(f"  REPLAY SURPRISE (early → late)")
            for name, attr in [('Mean surprise','surprise_mean'),
                               ('Pruned/cycle','pruned'),
                               ('Buffer size','buffer_size'),
                               ('Reward/cycle','reward_total')]:
                e = np.mean([getattr(s,attr) for s in first])
                l = np.mean([getattr(s,attr) for s in last])
                a = '↑' if l>e else '↓' if l<e else '→'
                print(f"    {name:<14} {e:8.2f} → {l:8.2f} {a}")


# ======================================================================
# Synthetic tests
# ======================================================================

def run_synthetic_tests():
    print("=" * 60)
    print("SYNTHETIC TESTS")
    print("=" * 60)

    # Test 1: Push / sample / stats
    print("\n[Test 1: Buffer basics]")
    buf = EpistemicReplayBuffer(capacity=500, prune_fraction=0.2, min_buffer_size=10)
    for i in range(100):
        buf.push(Transition(
            state=torch.randn(41), action=torch.randn(3),
            reward=float(i)/100, next_state=torch.randn(41),
            done=False, tick=i,
        ))
    assert len(buf) == 100
    assert len(buf.sample(32)) == 32
    stats = buf.get_stats()
    print(f"  Size={stats['size']}, AvgSurprise={stats['avg_surprise']:.2f}")
    print("  ✓ OK")

    # Test 2: prune_by_surprise respects ranking
    print("\n[Test 2: Prune by surprise ranking]")
    buf = EpistemicReplayBuffer(capacity=500, prune_fraction=0.5, min_buffer_size=10)
    # 50 low-surprise (consolidated) + 50 high-surprise (needed)
    for i in range(50):
        t = Transition(state=torch.randn(41), action=torch.randn(3),
                       reward=0.0, next_state=torch.randn(41), done=False, tick=i)
        t.replay_surprise = 0.01 * (i + 1)  # 0.01 .. 0.50
        buf.push(t)
    for i in range(50):
        t = Transition(state=torch.randn(41), action=torch.randn(3),
                       reward=1.0, next_state=torch.randn(41), done=False, tick=i+50)
        t.replay_surprise = 5.0 + i  # 5.0 .. 54.0
        buf.push(t)

    ps = buf.prune_by_surprise()
    print(f"  Before={ps.buffer_before}, Pruned={ps.pruned}, After={ps.buffer_after}")
    print(f"  Cutoff={ps.surprise_cutoff:.2f}")
    # Should have pruned 50 (50% of 100), all from the low-surprise group
    assert ps.pruned == 50
    assert ps.buffer_after == 50
    # All remaining should be high-surprise
    remaining_min = min(t.replay_surprise for t in buf.buffer)
    assert remaining_min >= 4.0, f"Expected high-surprise kept, got min={remaining_min}"
    print(f"  Remaining min surprise={remaining_min:.2f} (all high-surprise kept)")
    print("  ✓ Ranking correct")

    # Test 3: min_buffer_size respected
    print("\n[Test 3: Min buffer floor]")
    buf = EpistemicReplayBuffer(capacity=500, prune_fraction=0.9, min_buffer_size=30)
    for i in range(50):
        t = Transition(state=torch.randn(41), action=torch.randn(3),
                       reward=0.0, next_state=torch.randn(41), done=False, tick=i)
        t.replay_surprise = 0.001
        buf.push(t)
    ps = buf.prune_by_surprise()
    print(f"  50 transitions, prune_fraction=0.9, min=30: After={ps.buffer_after}")
    assert ps.buffer_after >= 30
    print("  ✓ Floor respected")

    # Test 4: iter_batches covers all
    print("\n[Test 4: iter_batches full coverage]")
    buf = EpistemicReplayBuffer(capacity=500)
    for i in range(73):  # Non-round number
        buf.push(Transition(state=torch.randn(41), action=torch.randn(3),
                            reward=0.0, next_state=torch.randn(41), done=False, tick=i))
    all_idx = []
    for batch, indices in buf.iter_batches(32):
        all_idx.extend(indices)
    assert sorted(all_idx) == list(range(73))
    print(f"  73 transitions in batches of 32: covered {len(all_idx)} indices")
    print("  ✓ Full coverage")

    print("\n  All synthetic tests passed ✅\n")


# ======================================================================
# Integration test
# ======================================================================

def run_integration(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print(f"INTEGRATION: {args.cycles} cycles × {args.wake_steps} steps  [{device}]")
    print("=" * 70)

    world = World()
    resources = create_default_resources()
    physics = Physics()
    sensors = SensorSystem(physics=physics)
    brain = CfCBrain().to(device)
    agent = SoliterAgent(brain, VitalsConfig(), device)

    config = TrainingConfig(
        wake_duration=args.wake_steps,
        learning_rate=0.0001,
        batch_size=16,
        sleep_epochs=3,
        prune_fraction=0.2,
        min_buffer_size=64,
    )

    trainer = SleepWakeTrainer(
        agent=agent, world=world, sensors=sensors,
        physics=physics, config=config, device=device,
    )

    tracker = DetailedTracker()

    hdr = (f"{'Cyc':<5} {'D':<3} {'Cause':<10} "
           f"{'Enrg':<8} {'Hydr':<8} {'Temp':<8} "
           f"{'Buf':<7} {'Prn':<5} "
           f"{'SurpMin':<8} {'SurpMn':<8} {'SurpMax':<8} {'Cut':<8} "
           f"{'Reward':<8}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for cycle in range(1, args.cycles + 1):
        tracker.start_cycle(cycle)

        for step in range(args.wake_steps):
            reward, done = trainer.wake_step(resources)
            world.step()
            tracker.record_step(agent, reward)
            if done:
                tracker.record_death(agent, world.tick)
                agent.reset()

        sleep_stats = trainer.sleep_cycle()

        ps = sleep_stats['transitions_pruned']
        cs = tracker.end_cycle(
            surprise_min=ps.surprise_min,
            surprise_mean=ps.surprise_mean,
            surprise_max=ps.surprise_max,
            surprise_cutoff=ps.surprise_cutoff,
            buffer_size=sleep_stats['buffer_size'],
            pruned=ps.pruned,
            policy_loss=sleep_stats.get('policy_loss', 0),
            value_loss=sleep_stats.get('value_loss', 0),
            entropy=sleep_stats.get('entropy', 0),
        )

        d_str = str(cs.deaths) if cs.deaths > 0 else '.'
        c_str = ','.join(cs.death_causes)[:9] if cs.death_causes else ''

        print(f"{cycle:<5} {d_str:<3} {c_str:<10} "
              f"{cs.energy_avg:>5.1f}/{cs.energy_min:<.0f} "
              f"{cs.hydration_avg:>5.1f}/{cs.hydration_min:<.0f} "
              f"{cs.temperature_avg:>5.1f}/{cs.temperature_min:<.0f} "
              f"{cs.buffer_size:<7} {cs.pruned:<5} "
              f"{cs.surprise_min:<8.4f} {cs.surprise_mean:<8.4f} "
              f"{cs.surprise_max:<8.2f} {cs.surprise_cutoff:<8.4f} "
              f"{cs.reward_total:<8.1f}")

    print("-" * len(hdr))
    print(f"\n📊 Buffer: {trainer.replay_buffer.total_added} added, "
          f"{trainer.replay_buffer.total_pruned} pruned, "
          f"final={len(trainer.replay_buffer)}, "
          f"deaths={len(tracker.all_deaths)}")

    tracker.print_summary()

    # Diagnostic checks
    print(f"\n{'='*60}")
    print("DIAGNOSTIC CHECKS")
    print(f"{'='*60}")

    # Pruning happening?
    total_pruned = sum(s.pruned for s in tracker.cycle_stats)
    print(f"\n[Pruning Active]")
    if total_pruned == 0:
        print("  ❌ No pruning — replay surprise may not be set")
    else:
        print(f"  ✓ {total_pruned} total pruned across {args.cycles} cycles")

    # Buffer bounded?
    sizes = [s.buffer_size for s in tracker.cycle_stats]
    print(f"\n[Buffer Growth]")
    print(f"  Range: {min(sizes)} → {max(sizes)}")
    if max(sizes) > args.cycles * args.wake_steps * 0.9:
        print("  ⚠️  Buffer barely pruned — growing near linearly")
    else:
        print("  ✓ Buffer regulated")

    # Surprise decreasing?
    if len(tracker.cycle_stats) >= 6:
        n = len(tracker.cycle_stats)
        early = np.mean([s.surprise_mean for s in tracker.cycle_stats[:n//3]])
        late = np.mean([s.surprise_mean for s in tracker.cycle_stats[2*n//3:]])
        print(f"\n[Surprise Trend]")
        print(f"  Mean surprise: {early:.4f} → {late:.4f} "
              f"({'↓ consolidating' if late < early else '→ flat/up'})")

    print(f"\n{'='*60}")
    print("Phase 4 Complete ✅")
    print(f"{'='*60}")

    return tracker


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Replay-Driven Pruning")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--wake-steps", type=int, default=100)
    args = parser.parse_args()

    run_synthetic_tests()
    run_integration(args)


if __name__ == "__main__":
    main()