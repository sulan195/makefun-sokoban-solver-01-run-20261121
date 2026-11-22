# Sokoban Solver — Push-Optimal per-Room A*

[![make.fun bounty](https://img.shields.io/badge/make.fun-bounty-blue)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

Entry for make.fun's **"Build the best Sokoban solver, 21 November 2026"**
bounty ($100 in USDC, top 10 × $10). A single Node 22 program (`main.ts`,
stdlib only) that answers every instance of a set in one run.

## How it works

The generator builds each level as a hallway with 1–4 rooms hanging off
it (one door each, 3–4 boxes per room), and a level's push-optimal count
P\* is the sum of its rooms' optima. The solver exploits that structure:

1. **Decompose** — the hallway is the row the player starts on; each
   door is a floor square in a neighbouring row touching hallway floor;
   a room is the flood fill from its door without crossing the hallway.
2. **Solve each room** with a push-optimal A\* search: a state is the
   sorted box squares plus the player's reachable region; the heuristic
   is the exact min-cost assignment of boxes to goals (BFS distances);
   pruning removes dead squares (reverse pull-reachability from the
   goals) and 2×2 box blocks with no goal inside. If the optimal search
   exhausts its work budget, weighted A\* / greedy fallbacks still find
   a solving (suboptimal) answer instead of nothing.
3. **Stitch** — rooms are joined with plain walks through the hallway
   (walks are free; only pushes count toward the score). If the
   decomposition ever fails validation, a whole-level push-A\* runs as
   a fallback.

All budgets are counts of work (search expansions), never the wall
clock, so two runs write byte-identical outputs.

## Results (practice set, kit's own grader)

| | Score |
|---|---|
| This solver | **1,000,000** (all 5 instances solved in exactly P\* pushes) |
| House baseline `plain-astar 1.0.0` | 200,000 (solves i001 only, gives up on the rest) |

Verified with the kit's grader (`grade-local`, process driver): all 5
instances 1000/1000, `passed: 1`, L1 regrade reproduced byte for byte.
Full run takes ~0.5 s for the 5-instance set (limit: 60 s).

## Run

```sh
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads `/input/<id>/level.txt`, writes `/output/<id>/solution.txt`
(LURD moves, one letter per step) for every instance.

## Files

- `main.ts` — the whole solver (no dependencies beyond Node's stdlib)
- `LICENSE` — MIT
