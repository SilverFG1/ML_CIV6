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
| `--city-position X Y` | Set the city's map coordinate. |
| `--unit-strengths N [N ...]`, `--city-strength` | Override unit and city combat strengths. |
| `--ranged-strength` | Configure the city's ranged attack strength. |
| `--wall-hp` | Set the city's initial wall health. |
| `--mountain-density`, `--forest-density` | Control terrain generation. |
| `--no-city-healing` | Disable city healing. |
| `--city-population N` | Start the city with N citizens (0–20) and simulate worked tiles and growth; 0 (the default) turns this off. |
| `--city-builders N`, `--city-settlers N` | Give the city up to two builders and two settlers; requires `--city-population`. |
| `--city-districts ...`, `--city-buildings ...` | Ancient-era districts and buildings the city starts with; requires `--city-population`. |
| `--build-queue ...` | Districts and buildings the city produces, in order, during the game. |
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

## City population and growth

`--city-population N` gives the defending city N citizens. Each turn the city collects food, grows, or starves, using Civ VI yields:

| Source | Food | Production |
| --- | --- | --- |
| City center (always worked) | 2 | 1 |
| Grassland | 2 | 0 |
| Forest (grassland woods) | 2 | 1 |
| Mountain | cannot be worked | |

- **Worked tiles.** The city owns the six tiles around its center, like a newly founded Civ VI city. Each citizen works one free tile, choosing the most food, then the most production. A tile with a unit on it cannot be worked, and citizens are reassigned whenever units move. A citizen with no free tile still eats.
- **Growth.** Each citizen eats 2 food. The surplus is stored, and the city gains a citizen when the store reaches the Civ VI threshold of 15 + 8n + n^1.5 food, where n is the population minus one (15, 24, 33, 44, …). Leftover food carries over.
- **Starvation.** If the city's food store would drop below zero, it empties and the city loses a citizen (never going below one).
- **Reward.** The attacker loses 1 reward each time the city grows and gains 1 each time it starves. Units standing next to the city therefore blockade it as well as besiege it.

Housing, amenities, and territory expansion are not modeled. When growth is enabled, the observation adds the population, food progress (in quarters), and food surplus, and the observation schema ends with `:city_growth`, so turning growth on for an existing model requires `--fresh`. Evaluation summaries also report `average_city_population`, the city's size at the end of each game. Without `--city-population`, scenarios, observations, and existing checkpoints are unchanged.

```powershell
python run_game.py --episodes 200 --unit-count 3 --city-population 3 --model-path runs/growth/model.json --metrics-dir runs/growth --fresh
```

During playback, the city's territory is outlined, gold dots mark worked tiles, red dots mark tiles a unit is blocking, and the status bar shows population, stored food against the growth threshold, and the food surplus.

## Builders and settlers

`--city-builders N` and `--city-settlers N` (up to two each) give a populated city civilians that start on free tiles in its territory. Civilians move one hex per turn after the squad moves, walking around units, other civilians, and the city itself.

- **Builders** have Civ VI's three build charges. A builder farms the unfarmed grassland tiles in the city's territory, using one charge per farm, and is used up when the charges run out. A farm adds 1 food, so a farmed tile yields 3 food, and citizens work farms first. Civ VI does not allow farms on woods, so forests are not farmed.
- **Settlers** walk to the nearest tile at least 4 hexes from every city (Civ VI's minimum spacing) and found a city there on the following turn. The new city is recorded and drawn but is not otherwise simulated.
- **Capture.** A unit that moves onto a civilian captures it and takes its tile. Battering rams cannot capture civilians, just as they cannot capture the city.
- **Reward.** Capturing a builder is worth +2 and capturing a settler +4. A city founded by a settler costs the attacker 5.

The squad therefore has to look for the city's civilians as well as besiege it: a settler that gets away founds a new city, and a builder left alone speeds up the city's growth. The heuristic hunts any civilian within 3 hexes of a healthy unit that can capture, and besieges otherwise. Units still move one hex per turn, so the Civ VI civilians' two movement points and the population cost of training a settler are not modeled.

The observation adds each civilian's position, whether it is still in play, and a builder's remaining charges. The schema ends with `:civilians=...`, so models need `--fresh` when civilians are added. Evaluation summaries also report `average_civilians_captured` and `average_cities_founded`.

```powershell
python run_game.py --episodes 300 --unit-count 3 --city-population 2 --city-builders 1 --city-settlers 1 --model-path runs/civilians/model.json --metrics-dir runs/civilians --fresh
```

During playback, civilians are purple discs marked B or S (with a builder's remaining charges beside its name), farms show yellow furrows, founded cities are outlined in purple, and the status bar counts captured civilians and founded cities.

## Districts and buildings (ancient era)

A populated city can have the ancient-era districts and buildings from Civ VI's Gathering Storm ruleset. `--city-districts` and `--city-buildings` set what the city starts with, and `--build-queue` lists what it produces during the game.

| District | Unlocked by | Cost | Placement | Adjacency bonus |
| --- | --- | --- | --- | --- |
| `campus` | Writing | 54 | Next to the city center | Science: +1 per adjacent mountain, +1 per 2 adjacent districts |
| `holy_site` | Astrology | 54 | Next to the city center | Faith: +1 per adjacent mountain, +1 per 2 adjacent woods, +1 per 2 adjacent districts |
| `encampment` | Bronze Working | 54 | Two hexes from the city center | None |

| Building | District | Cost | Effect in this simulation |
| --- | --- | --- | --- |
| `monument` | City center | 60 | +2 culture |
| `granary` | City center | 65 | +1 food |
| `ancient_walls` | City center | 80 | +100 wall HP and a city ranged strike |
| `library` | Campus | 80 | +2 science |
| `shrine` | Holy Site | 65 | +2 faith |
| `barracks` | Encampment | 80 | +1 production |

- **Placement.** A district goes on the free tile with the best adjacency bonus, then the lowest yields. The city center counts as a district for adjacency. A district tile is no longer worked by citizens, and a district placed on a farm replaces it. A tile with a unit or civilian on it cannot take a district.
- **District limit.** A city can have one district, plus one more for every three citizens beyond the first (1 at population 1–3, 2 at 4–6, and so on). A starting configuration that exceeds the limit is rejected.
- **Production.** The city's production (1 from the city center, 1 from each worked forest, and 1 from barracks) goes to the first queued item it can build. An item waits while its district is missing, the district limit is reached, or no tile is free; partial progress is kept. Leftover production carries over to the next item, and production is lost when nothing in the queue can be built. Blockading forest tiles therefore slows construction.
- **Combat.** Ancient Walls add 100 wall HP and give the city a ranged strike, as walls do in Civ VI. The encampment shoots a unit within 2 hexes of it each turn. Units cannot enter the encampment, which cannot be attacked. Both strikes use the city's ranged strength, or 15 (Civ VI's value for a civilization without ranged units) when the city has none. Battering rams help against these walls as they do against `--wall-hp`.
- **Reward.** The attacker loses 1 each time the city completes a district or building.

Culture, science and faith are tracked and shown, but nothing spends them. Several things are not modeled: border growth, rising district costs as Civ VI's tech and civics progress, housing, building maintenance, and pillaging. The Water Mill is left out because these maps have no rivers. The Government Plaza is left out because its buildings need Classical-era governments. In Civ VI, an Encampment can be placed only on a tile the city owns that is not next to the city center. Without border growth, this simulation places it two hexes out. Building costs come from community references, since the game's data files were not available here. Check them against your game's Civilopedia if exact values matter.

The observation adds a built flag for each listed district and building, plus the current item's production progress in quarters. An encampment appears as impassable in the terrain grid. The schema ends with `:built=...` and `:queue=...`, so changing them requires `--fresh`. Evaluation summaries report `average_builds_completed` when there is a queue.

```powershell
python run_game.py --episodes 300 --unit-count 3 --unit-types warrior spearman battering_ram --city-population 4 --forest-density 0.3 --city-districts campus --city-buildings monument library --build-queue ancient_walls encampment barracks --model-path runs/districts/model.json --metrics-dir runs/districts --fresh
```

During playback, districts are outlined in their own color (blue campus, gold holy site, red encampment), and the status bar lists completed buildings and the item in production with its progress.

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
