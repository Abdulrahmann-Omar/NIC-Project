# Chapter 2: The Algorithms

## Overview

We implemented 11 algorithms across two phases:

| Phase | Purpose | Algorithms |
|-------|---------|------------|
| 1 | Model Optimization | 7 algorithms |
| 2 | Meta-Optimization + XAI | 4 algorithms |

## Swarm Intelligence

### Particle Swarm Optimization (PSO)

**Inspiration**: Bird flocking, fish schooling

Each particle:
- Has a position (hyperparameters)
- Has a velocity (search direction)
- Remembers its personal best
- Knows the global best

```
velocity = w * velocity + c1 * r1 * (p_best - pos) + c2 * r2 * (g_best - pos)
position = position + velocity
```

### Grey Wolf Optimizer (GWO)

**Inspiration**: Wolf pack hierarchy

Pack structure:
- **Alpha**: Best solution (leader)
- **Beta**: Second best (advisor)
- **Delta**: Third best (scout)
- **Omega**: Rest of pack (followers)

Wolves encircle and attack prey (optimum).

### Whale Optimization Algorithm (WOA)

**Inspiration**: Humpback whale bubble-net hunting

Two phases:
1. **Encircling prey**: Shrink circle around optimum
2. **Bubble-net attack**: Spiral approach

## Single-Solution Methods

### Simulated Annealing (SA)

**Inspiration**: Metal annealing process

Key idea: Accept worse solutions with probability proportional to "temperature"

- High temp → Explore more
- Low temp → Exploit best

### Tabu Search

**Inspiration**: Human memory

Maintains a "tabu list" of recently visited solutions to avoid cycling.

## Meta-Optimization (Phase 2)

### Cuckoo Search

**Inspiration**: Cuckoo bird brood parasitism

Uses Levy flights for global search + local intensification.

We used it to optimize PSO's parameters!
