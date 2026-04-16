# ARC-AGI-3 Public Game Inventory (25 games)

**Source:** `Arcade().get_environments()` via `arc-agi==0.9.7`. Cached to `tools/games.json` on VPS (mirror in `game-previews/games.json`). Preview PNGs in `game-previews/<short>.png`.

**Purpose:** Stage 1 game selection per execution-plan.md §3.4. Three games picked: A (convention-aligned), B (convention-shifted), C (mixed). The 22 not selected are held out until Stage 4 (wrapper enforces this in `eval/held_out_wrapper.py`).

**Metadata columns:**
- `short` — 4-char prefix used throughout code
- `game_id` — full SDK identifier
- `tags` — SDK-declared input style (`keyboard`, `click`, `keyboard_click`)
- `levels` — number of levels in the public version
- `baseline_sum` — sum of human baseline actions across all levels (difficulty proxy)

## All games, grouped by input style

### Keyboard-only (4)
| short | game_id | levels | baseline_sum |
|---|---|---|---|
| ls20 | ls20-9607627b | 7 | 776 |
| wa30 | wa30-ee6fef47 | 9 | 1843 |
| g50t | g50t-5849a774 | 7 | 879 |
| tr87 | tr87-cd924810 | 6 | 414 |

### Click-only (7)
| short | game_id | levels | baseline_sum |
|---|---|---|---|
| vc33 | vc33-5430563c | 7 | 447 |
| lf52 | lf52-271a04aa | 10 | 1339 |
| lp85 | lp85-305b61c3 | 8 | 388 |
| su15 | su15-1944f8ab | 9 | 361 |
| tn36 | tn36-ef4dde99 | 7 | 317 |
| s5i5 | s5i5-18d95033 | 8 | 638 |
| r11l | r11l-495a7899 | 6 | 233 |

### Keyboard+Click hybrid (13)
| short | game_id | levels | baseline_sum |
|---|---|---|---|
| ar25 | ar25-0c556536 | 8 | 748 |
| dc22 | dc22-fdcac232 | 6 | 1228 |
| tu93 | tu93-0768757b | 9 | 462 |
| cd82 | cd82-fb555c5d | 6 | 171 |
| sb26 | sb26-7fbdac44 | 8 | 213 |
| sc25 | sc25-635fd71a | 6 | 350 |
| sk48 | sk48-d8078629 | 8 | 1070 |
| sp80 | sp80-589a99af | 6 | 518 |
| cn04 | cn04-2fe56bfb | 6 | 789 |
| m0r0 | m0r0-492f87ba | 6 | 1107 |
| ka59 | ka59-38d34dbb | 7 | 730 |
| re86 | re86-8af5384d | 8 | 1255 |
| bp35 | bp35-0a0ad940 | 9 | 651 |

### Untagged / docs demo (1)
| short | game_id | levels | baseline_sum |
|---|---|---|---|
| ft09 | ft09-0d8bbf25 | 6 | 208 |

## Selection criteria (from testing-plan.md §4.6)

- **Game A — convention-aligned:** button-like clickable regions + standard status-bar. Prediction: Layers 0-3 work, Layer 4 gives modest lift.
- **Game B — convention-shifted:** non-button clickable regions OR status-bar that looks like gameplay state. Prediction: Layers 0-3 struggle, Layer 4 provides lift if rule-inference works.
- **Game C — mixed:** partial convention shift. Tests Layer 4 robustness.

**Guidance for picking:**
- A and C likely come from the `click` or `keyboard_click` buckets (so the "button-like clickable" criterion has teeth).
- B is the stress test — the one that looks most unusual among the previews.
- Prefer games with moderate baseline_sum (100-500). Extremes (>1000) are either too hard to clear any level or too easy to inform.
- Avoid `ft09` — it is the SDK documentation demo game and is therefore heavily discussed online; substrate-contamination risk for Stage 3 C1 ablation.

## Selection (committed 2026-04-16)

- **Game A — `sb26` (SB26)** — convention-aligned.
  Textbook button-UI layout: top row of four distinct button-like colored squares (grey/maroon/teal/green), central cyan-bordered display panel with four red dot indicators, a red horizontal divider (status-bar-like), bottom row of four more colored squares. Button-shaped clickable regions top + bottom, status-bar analog in the middle. Baseline 213 steps across 8 levels. Tag: `keyboard_click`. Pixel-salience priors should latch easily onto the rectangular button regions.

- **Game B — `r11l` (R11L)** — convention-shifted.
  Pure non-button clickables: a black cross sprite, a white ring, a green cross, scattered across an irregular gray play area bordered by red. No UI region, no status bar, no rectangular button shapes anywhere. The clickable targets ARE gameplay objects. Textbook visual-salience stress test — the Layer 0 BCE classifier has no dedicated "UI quadrant" to anchor on and must learn click affordances from gameplay sprites. Baseline 233 steps across 6 levels. Tag: `click`. If Layer 4 doesn't rescue this, the StochasticGoose failure mode reproduces.

- **Game C — `su15` (SU15)** — mixed (one axis shifted).
  Yellow status strip at top with a distinct white button is clearly UI — that convention aligns. But the clickable objects on the playfield (a maroon target-ring, a black cross, a green diagonal dotted path, a white square) are gameplay objects, not buttons — that convention is shifted. Status ✓, click ✗. Baseline 361 steps across 9 levels. Tag: `click`. Tests whether Layer 4 provides partial lift when only one convention is off.

**Shared properties:** all three are click-capable (so ACTION6 coordinate reasoning applies to all), baseline_sum clustered in 213–361 (comparable difficulty), level counts 6–9 (enough reps for actions-to-first-level-up measurement), none of them is `ft09` (docs demo).

## Held-out (22 games)

`eval/held_out_wrapper.py` is configured with `allowed_games={"sb26-7fbdac44", "r11l-495a7899", "su15-1944f8ab"}` and raises `HeldOutGameError` for any of:

`ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 re86 s5i5 sc25 sk48 sp80 tn36 tr87 tu93 vc33 wa30`

Blinding lifts at Stage 4 (envelope validation on all 25).
