# ML_CIV6

A small Civ-inspired hex-grid combat environment and tabular Q-learning experiment, originally inspired by [this Civilization VI machine-learning video](https://www.youtube.com/watch?v=kjA9LxLx9IQ). Units move, heal, fight, and attempt to capture a defended city. The simulation supports Civ VI terrain types with their yields, rivers, city walls (including Civ VI Ancient Walls), ranged units, and difficulty presets. It implements a limited combat scenario rather than the full Civilization VI ruleset.

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
| `--city-position X Y` | Set the city's map coordinate. |
| `--unit-strengths N [N ...]`, `--city-strength` | Override unit and city combat strengths. |
| `--ranged-strength` | Configure the city's ranged attack strength. |
| `--wall-hp` | Set the city's initial wall health. |
| `--walls none\|ancient` | Give the city Civ VI Ancient Walls: 100 wall health and a ranged strike. |
| `--mountain-density`, `--forest-density` | Control terrain generation. |
| `--land-types grassland plains desert tundra snow` | Fill the map with these Civ VI land types, in regions (default: grassland). |
| `--hills-density`, `--water-density` | Turn this share of land into hills, or of all tiles into water. |
| `--river-count N` | Generate up to 10 rivers along tile edges (default 0). |
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

### Terrain and yields

The map uses Civ VI's base-game terrain rules, taken from the same catalog as the [real-game bridge](docs/REAL_CIV6.md) (`civ6_bridge/data/terrain_types.json`). There are five land types, each flat, hills, or mountain, plus coast and ocean. Forest is a separate feature on top of the terrain, as in Civ VI.

| Terrain | Flat | Hills | Mountain |
| --- | --- | --- | --- |
| Grassland | 2 food | 2 food, 1 production | none |
| Plains | 1 food, 1 production | 1 food, 2 production | none |
| Desert | none | 1 production | none |
| Tundra | 1 food | 1 food, 1 production | none |
| Snow | none | 1 production | none |
| Coast | 1 food, 1 gold | — | — |
| Ocean | 1 food | — | — |

- **Hills** cost two movement points and give units +3 defense.
- **Forest** adds 1 production, one movement point, and +3 defense. It grows on grassland, plains, and tundra, flat or hills, but not desert or snow. A forest on hills costs three movement points and gives +6 defense.
- **Mountains** are impassable. **Water** is impassable too, because units cannot embark.
- **Yields** are attached to every tile (`cell.yields`, e.g. `{'food': 2, 'production': 1}`) and shown as dots near the top of each tile: green food, orange production, yellow gold. Press **Y** to hide or show them. The simulation has no economy yet, so yields do not change combat, rewards, or the observation.

By default every map is grassland, forest, and mountains, exactly as before. `--land-types` splits the map into regions around randomly placed centres. `--hills-density` raises that share of flat land into hills; the city's tile stays flat. `--water-density` turns that share of tiles into water, which counts against the same 0.8 limit as mountains and forests. Water next to land becomes coast and water further out becomes ocean. Rivers never run alongside water.

Ancient Walls (`--walls ancient`) follow Civ VI. They give the city 100 wall health ("outer defense") and a ranged strike that hits one unit within two tiles each turn. In Civ VI that strike matches the defender's strongest ranged unit, or 15 without one; this city has no units, so it uses 15. `--wall-hp` and `--ranged-strength` override these values. Ancient Walls add no combat strength to the city, and battering rams work against them. They make a siege much harder: over 200 heuristic games at normal difficulty, three warriors never win, while two spearmen, an archer, and a ram win 133. The renderer draws intact walls as a stone ring around the city and broken walls as ruined towers.

Rivers run along the edges between tiles, not across tiles. Each river starts at a random edge and winds across the map until it leaves the map or joins another river. Rivers never block movement. As in Civ VI, a city attacked in melee from across a river gets +5 combat strength for that attack ("river defense"). The attacker deals less damage and takes more in return. Ranged attacks are unaffected. In Civ VI, crossing a river uses all of a unit's remaining movement. Every unit here already moves one hex per turn, so a crossing instead costs two movement points, like entering a forest. That cost feeds the movement penalty and the heuristic's path costs. The heuristic also prefers to attack from a tile that is on the city's side of the river.

The observation includes normalized absolute city/unit coordinates, row parity, health, survival, melee/ranged strengths, attack range, walls, and the terrain grid. With any of `--land-types`, `--hills-density`, or `--water-density`, each tile instead reports three combat properties: impassable, extra movement cost, and defense bonus, and the observation schema includes `:terrain`. With rivers enabled, it also has three river flags per tile, for the tile's east, southeast, and southwest edges, and the observation schema ends in `:rivers`. Scenarios without these options keep the earlier observation format, so existing models still load. Legal-action masks account for sequential movement, exclude impossible choices, and restrict dead units to waiting. Joint actions still grow exponentially: three warriors produce 343 combinations before masking, and four produce 2,401. Each archer or slinger has eight possible commands instead of seven. Larger scenarios therefore make this tabular learner substantially more expensive. Start small and use the reported baselines to judge whether training improves the learned policy.

For a mixed squad:

```powershell
python run_game.py --episodes 100 --unit-count 3 --unit-types warrior archer warrior --wall-hp 50 --forest-density 0.2 --mountain-density 0.1 --model-path runs/mixed/model.json --metrics-dir runs/mixed --fresh
```

For varied Civ VI terrain:

```powershell
python run_game.py --episodes 200 --unit-count 2 --land-types grassland plains desert --hills-density 0.2 --water-density 0.1 --forest-density 0.15 --model-path runs/terrain/model.json --metrics-dir runs/terrain --fresh
```

For a siege across rivers:

```powershell
python run_game.py --episodes 200 --unit-count 2 --river-count 2 --forest-density 0.1 --model-path runs/rivers/model.json --metrics-dir runs/rivers --fresh
```

For a city with Ancient Walls:

```powershell
python run_game.py --episodes 200 --unit-count 4 --unit-types spearman spearman archer battering_ram --walls ancient --model-path runs/ancient-walls/model.json --metrics-dir runs/ancient-walls --fresh
```

For a walled siege with a battering ram:

```powershell
python run_game.py --episodes 200 --unit-count 3 --unit-types warrior spearman battering_ram --wall-hp 50 --model-path runs/siege/model.json --metrics-dir runs/siege --fresh
```

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

During playback, **Space** pauses/resumes, **Right Arrow** advances one turn while paused, **+ / -** change playback speed, **Y** toggles yield dots, and **Esc** exits. `--render-training` also displays training turns.

Run `python game.py` to play manually. **Tab** selects a unit, **Q/E/A/D/Z/X** move, **Space** heals, **F** shoots with an archer or slinger, **Y** toggles yield dots, **R** restarts, and **Esc** exits. Each command advances the whole squad's turn; the other units wait and heal.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The tests cover simulation rules, independent environments, terminal rewards, legal actions, Q-learning updates, checkpoint validation/resumption, and runner behavior. GitHub Actions runs the suite on Python 3.12 with headless dependencies and executes a short CLI training/evaluation run. [UPGRADE_STATUS.md](UPGRADE_STATUS.md) tracks the ten upgrade areas and their integration evidence.
