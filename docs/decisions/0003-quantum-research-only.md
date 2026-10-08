# 0003: Quantum optimisation is research only

- **Status:** accepted
- **Date:** 2026-10-07

## Context

The project began as a hybrid quantum-classical portfolio optimiser, with
asset selection written as a QUBO problem. The new goal is a live trading
system. Today, cloud QPUs are slow and queued, and classical solvers find
equal or better portfolios in milliseconds on problems of this size.

## Decision

Keep quantum optimisation out of the live path. The optimiser in production is
classical (CVXPY). A QUBO formulation may be compared with classical solvers
offline, under `research/quantum`, on the same problem and time budget.

## Consequences

- The live system has no dependency on QPU availability or queue times.
- The quantum comparison, if done, is reported honestly, including a loss to
  the classical baseline (roadmap Phase 6).
