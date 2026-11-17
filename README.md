# ML_CIV6

A small Civ-inspired hex-grid combat environment and tabular Q-learning experiment, originally inspired by [this Civilization VI machine-learning video](https://www.youtube.com/watch?v=kjA9LxLx9IQ). Units move, heal, fight, and attempt to capture a defended city. The simulation supports configurable terrain, city walls, ranged units, and difficulty presets. It implements a limited combat scenario rather than the full Civilization VI ruleset.

The default learner is a tactical squad Q-learner. After 1,000 training episodes it captures the city in almost every evaluation game on seeds it never trained on, including hard and walled sieges; see [How the AI works](#how-the-ai-works).

An optional [real Civilization VI integration](docs/REAL_CIV6.md) now provides structured observations, saved snapshots, a basic tactical controller, and individual unit commands through FireTuner. Start with `python run_civ6.py doctor`. Orders are previewed until `--execute` is supplied. This first milestone controls a local player's units; full-game strategy and training the learner against the real game remain future work.

## Setup

Python **3.12** is supported. The development environment uses Python 3.12.4, NumPy 2.4.6, and the optional Pygame 2.6.1 renderer. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, create the environment with `python3.12 -m venv .venv`, then activate it with `source .venv/bin/activate`. The commands below assume the virtual environment is active. Install `requirements-render.txt` instead of `requirements.txt` when using `--render` or `--render-training`:

```powershell
python -m pip install -r requirements-render.txt
```

Headless training requires NumPy only. Importing or running the simulation headlessly does not initialize Pygame or load image assets.

## Train, evaluate, and resume

Run a short smoke test:

```powershell
python run_game.py --episodes 2 --max-steps 4 --eval-episodes 2 --fresh --no-save --metrics-dir runs/smoke
```

Train a model, save periodic checkpoints, and evaluate all three policies:

```powershell
python run_game.py --episodes 1000 --max-steps 80 --eval-episodes 50 --seed 7 --eval-seed 100007 --model-path runs/baseline/model.json --metrics-dir runs/baseline --checkpoint-every 50 --fresh
```

Continue the same model for another 500 training episodes:

```powershell
python run_game.py --episodes 500 --model-path runs/baseline/model.json --metrics-dir runs/resumed
```

`--episodes` is the number of additional training episodes. An existing model loads automatically and restores its learner, scenario, learning settings, progress, seeds, and policy random-generator state. Compatible learning options can be explicitly overridden; learner, scenario, seed, precision, and validation-set changes require `--fresh`. Resumption starts at an episode boundary. `--no-save` disables checkpoint writes while retaining metrics output. Joint Q-table models saved before the tactical learner still resume with their original learner; train a tactical model with `--fresh`. Models older than the joint learner's current observation format require `--fresh`.

Evaluate a saved model without further training:

```powershell
python run_game.py --episodes 0 --eval-episodes 100 --model-path runs/baseline/model.json --metrics-dir runs/evaluation
```

By default, evaluation reports **learned**, **heuristic**, and **random** policies separately. Each sees the same scenario seeds, and each policy has an independent random stream. Learned evaluation uses the Q-table with guidance disabled, so its score measures the learner. Training can still use heuristic guidance; use `--no-guidance` to disable it. `--eval-policies learned heuristic` selects baselines to run, while `--eval-guidance` adds a separately labelled guided-policy result. `--eval-episodes 0` skips evaluation.

Use `--save-best --validation-episodes 20` to also select a model using a fixed validation seed set. Validation uses `--validation-seed` (default 200007), independently of the final evaluation seed set (`--eval-seed`, default 100007). Keep final evaluation seeds reserved when comparing training settings.

## Scenario configuration

Start with a preset, then apply explicit overrides:

```powershell
python run_game.py --episodes 500 --difficulty easy --width 7 --height 7 --unit-count 2 --mountain-density 0.05 --forest-density 0.1 --wall-hp 0 --model-path runs/easy/model.json --metrics-dir runs/easy --fresh
```

| Option | Purpose |
| --- | --- |
| `--difficulty easy\|normal\|hard\|curriculum` | Choose city strength: 18, 28, 38, or a 1,000-episode curriculum targeting 38. |
| `--width`, `--height` | Set map dimensions from 3 to 32. |
| `--unit-count` | Choose one to four starting units. |
| `--unit-types warrior spearman heavy_chariot slinger archer battering_ram ...` | Set one type per unit; omitted types default to warriors. |
| `--city-position X Y` | Set the city's map coordinate. |
| `--unit-strengths N [N ...]`, `--city-strength` | Override unit and city combat strengths. |
| `--ranged-strength` | Configure the city's ranged attack strength. |
| `--wall-hp` | Set the city's initial wall health. |
| `--mountain-density`, `--forest-density` | Control terrain generation. |
| `--no-city-healing` | Disable city healing. |
| `--curriculum-episodes N` | Progress from easier scenarios to the selected difficulty over N training episodes. |

Each unit type has default strengths; `--unit-strengths` overrides the melee value only (for a battering ram, its defense).

| Unit | Class | Melee | Ranged | Shooting range |
| --- | --- | --- | --- | --- |
| `warrior` | melee | 20 | — | — |
| `spearman` | anti-cavalry | 25 | — | — |
| `heavy_chariot` | heavy cavalry | 28 | — | — |
| `slinger` | ranged | 5 | 15 | 1 (adjacent) |
| `archer` | ranged | 12 | 20 | 2 |
| `battering_ram` | support | — | — | — |

Spearmen, heavy chariots, slingers, and battering rams use their Civ VI base strengths. Every unit moves one hex per turn, so the heavy chariot's extra movement on flat terrain is not modeled. There are no enemy units, so the spearman's anti-cavalry bonus does not apply either. Slingers and archers have a `SHOOT` action within their range, with no melee retaliation. A slinger's weak melee strength makes attacking the city in melee costly, so it should shoot from an adjacent tile. Ranged fire leaves the city at a minimum of one HP, so capture requires a melee action.

Walls absorb damage before the city does. Melee attacks deal half damage to walls; once the walls break, the rest of the attack hits the city at full strength. A living battering ram next to the city restores full wall damage for warriors and spearmen, but not for heavy chariots or for ranged units attacking in melee. Ranged attacks always deal full damage to walls. Battering rams cannot attack or capture the city, have no combat strength when the city shoots them, and a squad whose only survivors are rams has lost. Mountains block movement. Forests provide a defense bonus and add a movement-cost penalty. Terrain generation places units in the city's reachable component.

The environment offers two observations. The tactical learner uses the egocentric encoding described in [How the AI works](#how-the-ai-works). The joint learner (`--agent joint`) uses the raw encoding: normalized absolute city/unit coordinates, row parity, health, survival, melee/ranged strengths, attack range, walls, and the terrain grid. Legal-action masks account for sequential movement, exclude impossible choices, and restrict dead units to waiting. Joint actions grow exponentially: three warriors produce 343 combinations before masking, and four produce 2,401. Each archer or slinger has eight possible commands instead of seven. The tactical learner still chooses among these joint actions, but its tables store values per unit and tactic, so their size does not grow with the number of joint actions. Use the reported baselines to judge whether training improves the learned policy.

For a mixed squad:

```powershell
python run_game.py --episodes 100 --unit-count 3 --unit-types warrior archer warrior --wall-hp 50 --forest-density 0.2 --mountain-density 0.1 --model-path runs/mixed/model.json --metrics-dir runs/mixed --fresh
```

For a walled siege with a battering ram:

```powershell
python run_game.py --episodes 200 --unit-count 3 --unit-types warrior spearman battering_ram --wall-hp 50 --model-path runs/siege/model.json --metrics-dir runs/siege --fresh
```

## How the AI works

Evaluation compares three policies: the **learned** squad policy, a rule-based **heuristic**, and **random** play.

**What the learner sees.** For each unit, the tactical observation records its type, its fewest turns to the city along a route around the squad's other units (capped at six), its health level, and its exact chance of dying to the city's retaliation if it attacks now (none, unlikely, plausible, likely). That chance is computed from the combat formula, including forest defense. It also records the city's health in fifths, the walls (none, damaged, intact), how many allies stand next to the city, how many attackers are still away, whether a battering ram is in place, and whether the city shoots and heals. For each command, the observation also records the tactic it performs from the unit's current position: `ATTACK`, `SHOOT`, `HEAL` (holding while wounded), `WAIT` (holding at full health), `ADVANCE`, `SIDESTEP`, or `RETREAT`. Nothing depends on absolute coordinates, the map size, or the seed, so experience from one generated map applies to the next.

**How it learns.** A joint action's value is the sum of each living unit's value for its tactic: a tabular value-decomposition network. The squad still picks the best *legal joint* action, so collisions and sequential movement are respected, while units of a type share what they learn. Each unit's value is a sum of five tables over subsets of its features (tile coding), so situations never seen in full still get estimates from coarser tables. Each update moves the joint value the same fraction of the way to its target however many units share table entries, which keeps four-unit squads stable. Several training details keep the greedy policy from stalling:

- Rewards are shaped with the squad's health as a potential, which credits healing immediately without changing which policy is optimal.
- An episode cut off by `--max-steps` bootstraps from its last state instead of being treated as a terminal state, so running out the clock does not look free.
- A turn that changes nothing, or a loop the policy's own greedy choices would repeat, is valued as repeating forever. This replaces a slow self-referential update.
- Once the scenario stops changing, the learning rate decays from 0.1 by 0.998 per episode to a floor of 0.01, so the learned policy settles. A curriculum keeps full-sized updates until it reaches its final city strength.

`--shaping`, `--learning-rate-decay`, and `--learning-rate-min` adjust these (a decay of 1 disables annealing), and `--agent joint` trains the original joint-action Q-table for comparison. Hard sieges typically need 500–1,000 training episodes; small scenarios learn in a few hundred.

**The heuristic** uses the same combat odds. Units close in by the fewest turns, archers and slingers shoot when in range, and a melee attack is made only when the retaliation cannot kill the attacker while the city is encircled (a small risk is accepted while the city can still heal), unless a capture is nearly certain. Otherwise units heal in place, hold next to the city until enough of them stop it from healing, and wait outside a ranged city's reach until the squad closes up. It also serves as the guide for heuristic-guided exploration during training.

**Results.** Win rates over 100 held-out evaluation games (seeds 100007–100106) after 1,000 training episodes with default settings. "Learned" is the mean over four training seeds, with the worst seed's rate in parentheses when below 100%. "Original heuristic" is the rule-based policy this project shipped before; losses are shown where any occurred.

| Scenario | Settings | Learned | Heuristic | Original heuristic | Random |
| --- | --- | --- | --- | --- | --- |
| default | 8×8, three warriors, normal city | 100% | 100% | 93%, 7% lost | 0% |
| hard | `--difficulty hard` | 98.5% (94%) | 100% | 0%, 100% lost | 0% |
| curriculum | `--difficulty curriculum` | 96.3% (87%) | 100% | 0%, 100% lost | 0% |
| walled siege | warrior, spearman, battering ram; `--wall-hp 50` | 100% | 99%, 1% lost | 58%, 22% lost | 0% |
| mixed terrain | warrior, archer, warrior; walls 50; forest 0.2, mountains 0.1 | 100% | 100% | 87% | 0% |
| hard terrain | hard; forest 0.2, mountains 0.1 | 99.8% (99%) | 93% | 0%, 100% lost | 0% |
| chariot and ram | heavy chariot, battering ram; walls 30 | 100% | 99%, 1% lost | 99%, 1% lost | 0% |
| slingers | warrior, slinger, slinger | 100% | 100% | 90% | 0% |
| four units | four warriors, 10×10 | 100% | 100% | 100% | 0% |
| easy small | easy, 7×7, two warriors, light terrain | 100% | 100% | 100% | 3% |
| ranged city | `--ranged-strength 25` | 0%, none lost | 0%, 100% lost | 6%, 94% lost | 0%, 94% lost |
| ranged city, archers | ranged 25; warrior, archer, archer | 0%, none lost | 67%, 33% lost | 26%, 74% lost | 0%, 96% lost |

The learned policy lost no evaluation games in any scenario. The original joint Q-table won none of its evaluation games in the default, hard, walled-siege and mixed-terrain scenarios, which were the ones measured; it is still available as `--agent joint`. A default model file is now about 0.5 MB instead of about 67 MB.

Against a city with a ranged attack, the learner declines the fight and waits out the turn limit. With three warriors that is the better choice under the game's rewards (average reward −4.0 against −26.1 for the heuristic's assault). With archers the heuristic's assault is worth more (−1.2 against −4.6), so the learner misses a winnable fight there; a coordinated assault under fire is not yet discovered by its exploration.

## Metrics and playback

Each run writes to `--metrics-dir` (default `training_metrics`):

- `training.csv`: episode-level training metrics, including reward, win/termination outcome, length, Q-table size (table rows for the tactical learner), and guidance usage.
- `metrics.json`: scenario and learning settings, seeds, training history, and evaluation summaries for comparing runs.
- `learning_curve.svg`: training curves that open in a browser without extra plotting packages.

Use separate output directories for experiments. Checkpoints save atomically; periodic saves preserve recoverable progress during long runs. `--checkpoint-every N` controls their interval.

Render evaluation after training or replay a saved policy:

```powershell
python run_game.py --episodes 0 --eval-episodes 3 --eval-policies learned --model-path runs/baseline/model.json --metrics-dir runs/playback --render --render-delay-ms 100
```

During playback, **Space** pauses/resumes, **Right Arrow** advances one turn while paused, **+ / -** change playback speed, and **Esc** exits. `--render-training` also displays training turns.

Run `python game.py` to play manually. **Tab** selects a unit, **Q/E/A/D/Z/X** move, **Space** heals, **F** shoots with an archer or slinger, **R** restarts, and **Esc** exits. Each command advances the whole squad's turn; the other units wait and heal.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The tests cover simulation rules, independent environments, terminal rewards, legal actions, Q-learning updates, checkpoint validation/resumption, and runner behavior. `tests/test_tactical.py` checks the combat odds against simulated attacks, the tactical encoding, the heuristic's rules, and the tactical learner's arithmetic and checkpoints. A short training regression checks that the learner wins held-out games without guidance. GitHub Actions runs the suite on Python 3.12 with headless dependencies and executes a short CLI training/evaluation run. [UPGRADE_STATUS.md](UPGRADE_STATUS.md) tracks the upgrade areas and their integration evidence.
