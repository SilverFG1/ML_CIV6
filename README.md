# ML_CIV6

A small Civ-inspired hex-grid combat environment and tabular Q-learning experiment, originally inspired by [this Civilization VI machine-learning video](https://www.youtube.com/watch?v=kjA9LxLx9IQ). Units move, heal, fight, and attempt to capture a defended city. The simulation supports configurable terrain, city walls, ranged units, and difficulty presets. It implements a limited combat scenario rather than the full Civilization VI ruleset.

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

`--episodes` is the number of additional training episodes. An existing model loads automatically and restores its scenario, learning settings, progress, seeds, and policy random-generator state. Compatible learning options can be explicitly overridden; scenario, seed, precision, and validation-set changes require `--fresh`. Resumption starts at an episode boundary. `--no-save` disables checkpoint writes while retaining metrics output. Legacy models require `--fresh` because their observations are incompatible with the upgraded environment.

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
| `--city-count N` | Choose one to four enemy cities; capture all of them for a domination victory. |
| `--city-position X Y` | Set the enemy capital's map coordinate (the first city; other cities are placed randomly). |
| `--player-capital` | Give the squad its own capital to defend against raiders; losing it loses the game. |
| `--raid-interval N` | Turns between raids on the player's capital (default 6); 0 disables raids. |
| `--capital-strength`, `--raider-strength` | Set the player's capital (default 28) and raider (default 20) combat strengths. |
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

The observation includes normalized absolute city/unit coordinates, row parity, health, survival, melee/ranged strengths, attack range, walls, and the terrain grid. Legal-action masks account for sequential movement, exclude impossible choices, and restrict dead units to waiting. Joint actions still grow exponentially: three warriors produce 343 combinations before masking, and four produce 2,401. Each archer or slinger has eight possible commands instead of seven. Larger scenarios therefore make this tabular learner substantially more expensive. Start small and use the reported baselines to judge whether training improves the learned policy.

For a mixed squad:

```powershell
python run_game.py --episodes 100 --unit-count 3 --unit-types warrior archer warrior --wall-hp 50 --forest-density 0.2 --mountain-density 0.1 --model-path runs/mixed/model.json --metrics-dir runs/mixed --fresh
```

For a walled siege with a battering ram:

```powershell
python run_game.py --episodes 200 --unit-count 3 --unit-types warrior spearman battering_ram --wall-hp 50 --model-path runs/siege/model.json --metrics-dir runs/siege --fresh
```

## Domination victory

Each enemy city is a rival civilization's capital. Capturing every one is a domination victory and ends the game as a win; with the default single city, this is the same as capturing it. Use `--city-count 2` to `4` for a longer campaign:

```powershell
python run_game.py --episodes 500 --city-count 3 --difficulty easy --max-steps 200 --model-path runs/domination/model.json --metrics-dir runs/domination --fresh
```

- The enemy capital sits at `--city-position` (or the map center). Other cities are placed randomly at least three hexes apart, in reach of the squad, and with enough open tiles around them for a full siege.
- Every city that has not been captured shoots and heals on its own; all cities share the strength, wall, and ranged settings.
- A captured city stops fighting and its tile opens to your units. Capturing a city before the last earns +5, and the last one earns the +10 victory reward.
- Ranged units shoot the closest city in range; ties go to the city with the least health plus walls. A battering ram helps only against the city it stands next to.
- Evaluation summaries add the average number of cities captured, which shows progress even when games time out.

Domination is much harder than a single siege, because damage carries over between cities. With three units on normal difficulty, the built-in heuristic wins 65–73% of games against two to four cities, and most other games time out after a unit is lost. On easy difficulty, it wins every three-city game. Allow more turns with `--max-steps` for larger campaigns.

## Defending your capital

Add `--player-capital` to give the squad its own capital, Otterholt, which the enemy raids. Losing it loses the game. It combines with every other option, including domination:

```powershell
python run_game.py --episodes 500 --player-capital --model-path runs/capital/model.json --metrics-dir runs/capital --fresh
```

- The capital is placed as far from the enemy cities as the map allows, at least three hexes from each, and the squad starts around it. Your units cannot enter it.
- It has 100 HP and strength 28 (`--capital-strength`). It hits back when attacked and heals 10 HP at the end of any turn with no raider next to it.
- Every 6 turns (`--raid-interval`; 0 disables raids), an enemy city sends a raider (strength 20, `--raider-strength`) from a tile beside it. With several enemy cities, they take turns, so each city you capture slows the raids. At most four raiders are on the map at once.
- Raiders march toward the capital and attack it once they reach it. They step around your units when they can, and attack a unit that blocks every way forward.
- Move into a raider to attack it. Archers and slingers can also shoot raiders, and their shots can kill. Battering rams cannot fight raiders.
- Damage to the capital costs 3 reward per 100 HP, damage dealt to raiders earns 0.5 per 100 HP, and losing the capital is a −10 loss.
- The observation adds the capital's position, health, and strength, and the position and health of up to four raiders. Models trained with and without `--player-capital` are not interchangeable.
- Evaluation summaries show how many losses came from losing the capital.

Left undefended, the capital falls in every game, around turn 31. One raider cannot take a healthy capital, because the capital's counterattacks kill it first, so the built-in heuristic keeps attacking. It turns back only when nearby raiders could destroy the capital, unless the city it is besieging is nearly taken. With default settings it wins 84% of games against one enemy city and loses its capital in 7%; against three cities over 200 turns, it wins 41% (97% on easy difficulty).

## Metrics and playback

Each run writes to `--metrics-dir` (default `training_metrics`):

- `training.csv`: episode-level training metrics, including reward, win/termination outcome, length, Q-table size, and guidance usage.
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

The tests cover simulation rules, independent environments, terminal rewards, legal actions, Q-learning updates, checkpoint validation/resumption, and runner behavior. GitHub Actions runs the suite on Python 3.12 with headless dependencies and executes a short CLI training/evaluation run. [UPGRADE_STATUS.md](UPGRADE_STATUS.md) tracks the ten upgrade areas and their integration evidence.
