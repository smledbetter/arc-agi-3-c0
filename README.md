# arc-agi-3-c0

Structured-exploration agent for ARC-AGI-3. Canonical plan: [/Users/stevo/Sites/Thinking/Projects/ARC-AGI-3/](../../../Sites/Thinking/Projects/ARC-AGI-3/) on laptop.

## Week 1 status (2026-04-16)

- SDK verified: `arc-agi==0.9.7`, `arcengine==0.9.3`, 25 public games fetched via anonymous API key.
- Seed-replay validator (`tools/verify_seed_replay.py`) passes on ls20 + ft09, 200/2000 steps, multiple seeds — bitwise-identical trajectories modulo wall_ms (OS jitter).
- Trace logger + cost/wall instrumentation in `agent/{trace_logger,instrumentation}.py`.
- Held-out wrapper in `eval/held_out_wrapper.py` raises `HeldOutGameError` on non-selected games and logs every access to `logs/game_access.log`.
- 25-game inventory with preview PNGs in `tools/game_inventory.md` + `games/previews/`. A/B/C selection awaiting visual inspection.

## Layout

- `agent/` — C0 layers, seed-replay runner, logging, instrumentation
- `agent/layers/` — Layer 0-4 implementations (TBD: Stage 1)
- `eval/` — eval runners + held-out wrapper
- `games/previews/` — PNG previews of all 25 public games
- `tools/` — validator, preview renderer, trace parser, game inventory
- `traces/` — JSONL trace output (gitignored)
- `logs/` — session logs, game access audit (gitignored)
- `paper/` — paper drafts (Stage 1 onward)

## Running the validator

```bash
source .venv/bin/activate
python tools/verify_seed_replay.py --seed 42 --game ls20 --steps 2000
```

Exits 0 iff two runs with the same master seed produce bitwise-identical JSONL traces modulo wall_ms. Exit 1 on any trajectory-data mismatch.
