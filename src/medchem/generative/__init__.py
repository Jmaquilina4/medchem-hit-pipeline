"""Generative design: a multi-objective reward, and the interfaces a real generator would implement.

WHAT RUNS: the reward engine and the scorer, on CPU. ``MockSampler`` replays curated compounds so the
wiring is exercised end to end, and both the constrained (geometric-mean) and naive (sum) reductions are
scored for the reward-hacking comparison. The applicability-domain term is a real distance to the training
set.

WHAT DOES NOT: ``Reinvent4Sampler`` and ``Boltz2Scorer`` are INTERFACES WHOSE SHIPPED METHODS RAISE. They
define the contract an external implementation must satisfy; this package contains no REINVENT4 or Boltz-2
integration, and no published result depends on either. The reward specification is required from config —
there is no default, because a reward is a per-target scientific claim.
"""
