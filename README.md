# ML_CIV6

A small Civ-inspired hex-grid combat environment and tabular Q-learning experiment, originally inspired by [this Civilization VI machine-learning video](https://www.youtube.com/watch?v=kjA9LxLx9IQ). Units move, heal, fight, and attempt to capture a defended city. The simulation supports Civ VI terrain types with their yields, rivers, city walls (including Civ VI Ancient Walls), ranged units, and difficulty presets. It implements a limited combat scenario rather than the full Civilization VI ruleset.

The default learner is a tactical squad Q-learner. After 1,000 training episodes it captures the city in almost every evaluation game on seeds it never trained on, including hard and walled sieges. Part of its training watches the rule-based heuristic play, which teaches it assaults it would not find on its own; see [How the AI works](#how-the-ai-works).

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

`--episodes` is the number of additional training episodes. An existing model loads automatically and restores its learner, scenario, learning settings, progress, seeds, and policy random-generator state. Compatible learning options can be explicitly overridden; learner, scenario, seed, precision, demonstration-share, and validation-set changes require `--fresh`. Resumption starts at an episode boundary. `--no-save` disables checkpoint writes while retaining metrics output. Models saved before demonstrations existed resume without them. Joint Q-table models saved before the tactical learner still resume with their original learner; train a tactical model with `--fresh`. Models older than the joint learner's current observation format require `--fresh`.

Evaluate a saved model without further training:

```powershell
python run_game.py --episodes 0 --eval-episodes 100 --model-path runs/baseline/model.json --metrics-dir runs/evaluation
```

By default, evaluation reports **learned**, **heuristic**, and **random** policies separately. Each sees the same scenario seeds, and each policy has an independent random stream. Learned evaluation uses the Q-table with guidance disabled, so its score measures the learner. Training can still use the heuristic, both to guide some exploratory moves and to play [demonstration episodes](#how-the-ai-works) that the learner learns from; `--demonstrations 0` turns off demonstrations, and `--no-guidance` turns off both. `--eval-policies learned heuristic` selects baselines to run, while `--eval-guidance` adds a separately labelled guided-policy result. `--eval-episodes 0` skips evaluation.

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
| `--walls none\|ancient` | Give the city Civ VI Ancient Walls: 100 wall health and a ranged strike. |
| `--buildings monument granary` | Give the city a Monument (+2 culture) and/or a Granary (+1 food, +2 housing). |
| `--economy` | Run the city's economy: it grows, builds, and claims tiles while you besiege it. |
| `--city-population N`, `--build-queue ...` | With `--economy`, set the starting population (1–20, default 1) and what the city builds, in order (default: `monument granary ancient_walls`). |
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

Ancient Walls (`--walls ancient`) follow Civ VI. They give the city 100 wall health ("outer defense") and a ranged strike that hits one unit within two tiles each turn. In Civ VI that strike matches the defender's strongest ranged unit, or 15 without one; this city has no units, so it uses 15. `--wall-hp` and `--ranged-strength` override these values. Ancient Walls add no combat strength to the city, and battering rams work against them. They make a siege much harder: over 200 heuristic games at normal difficulty, three warriors never win, while two spearmen, an archer, and a ram win 192. The renderer draws intact walls as a stone ring around the city and broken walls as ruined towers.

Rivers run along the edges between tiles, not across tiles. Each river starts at a random edge and winds across the map until it leaves the map or joins another river. Rivers never block movement. As in Civ VI, a city attacked in melee from across a river gets +5 combat strength for that attack ("river defense"). The attacker deals less damage and takes more in return. Ranged attacks are unaffected. In Civ VI, crossing a river uses all of a unit's remaining movement. Every unit here already moves one hex per turn, so a crossing instead costs two movement points, like entering a forest. That cost feeds the movement penalty and breaks ties between the heuristic's routes. When a tile beside the city on its side of the river is one step away, the heuristic moves there before attacking, and the tactical learner's odds of dying in an attack include the river defense.

The environment offers two observations. The tactical learner uses the egocentric encoding described in [How the AI works](#how-the-ai-works). Terrain, rivers, walls, and the economy reach it only through the features and odds they change, so its format never varies. The joint learner (`--agent joint`) uses the raw encoding: normalized absolute city/unit coordinates, row parity, health, survival, melee/ranged strengths, attack range, walls, and the terrain grid. With any of `--land-types`, `--hills-density`, or `--water-density`, each tile instead reports three combat properties: impassable, extra movement cost, and defense bonus, and the raw observation schema includes `:terrain`. With rivers enabled, it also has three river flags per tile, for the tile's east, southeast, and southwest edges, and the schema ends in `:rivers`. `--economy` adds city features and `:economy` (see [City economy](#city-economy)). Scenarios without these options keep the earlier observation format, so existing models still load. Legal-action masks account for sequential movement, exclude impossible choices, and restrict dead units to waiting. Joint actions grow exponentially: three warriors produce 343 combinations before masking, and four produce 2,401. Each archer or slinger has eight possible commands instead of seven. The tactical learner still chooses among these joint actions, but its tables store values per unit and tactic, so their size does not grow with the number of joint actions. Use the reported baselines to judge whether training improves the learned policy.

### City buildings

`--buildings` gives the city Civ VI city-centre buildings, with their base-game values:

| Building | Effect | Production cost | Requires |
| --- | --- | --- | --- |
| Monument | +2 culture | 60 | — |
| Granary | +1 food, +2 housing | 65 | Pottery |

Some wikis list a later cost of 50 for both; the simulator records cost but has no production yet. The city exposes the totals as `city.building_yields` (e.g. `{'food': 1, 'culture': 2}`) and `city.building_housing`. The renderer draws an obelisk (Monument) and a small barn (Granary) on the city tile and lists the buildings and their effects in the status line. As in Civ VI, neither building affects combat. Without `--economy` they change nothing else either; with it, their yields and housing feed the city's growth and borders.

### City economy

`--economy` runs a Civ VI-style economy for the city you are besieging, at the end of each of its turns:

- **Yields.** The city collects yields from its centre tile (always at least 2 food and 1 production), one worked tile per citizen, and its buildings.
- **Citizens** work the owned tiles with the most food and production, preferring food on ties. They swap production for food when the city would otherwise starve. They cannot work mountains, the centre, or **a tile one of your units stands on**, so a siege can starve the city or slow its building.
- **Food.** Each citizen eats 2 food. Surplus fills the food store, and the city grows when it holds 15, 24, 33, 44, … food (15 + 8(n−1) + (n−1)^1.5). If the store would go negative, the city loses a citizen.
- **Housing** is 2, or 3 next to water, or 5 next to a river; a Granary adds 2. Growth halves one citizen below housing, drops to a quarter at or above it, and stops five above it.
- **Production** builds the `--build-queue` in order: Monument (60), Granary (65), Ancient Walls (80). Overflow carries over, and items the city already has are skipped. Finished walls arrive at full health and switch on the city's ranged strike.
- **Culture** (2 per turn from a Monument) claims a new tile every 10, then 10 + (6t)^1.3 culture, where t is the number of tiles claimed so far. Each claim takes the best unowned tile next to the territory, up to 3 tiles from the centre. A new city owns its centre and the six tiles around it.

The economy sets a clock on the siege. Left alone on the default grassland map, a size-1 city building walls first finishes them in a median 40 turns. Standing on three of its tiles doubled that to a median 80 in testing, because grassland has no production and the forests that do are easy to block. On plains with hills it needs about 27 turns. The rule-based baseline usually takes the city within about 6 turns, before anything is built, so the clock mostly pressures slower policies such as the learner. Rewards are unchanged. The observation gains the city's population, progress towards growth, production and the next tile, its housing, and which item it is building; the schema ends in `:economy`.

The renderer outlines the city's territory in red, marks worked tiles with a white citizen token, and adds an economy line to the status panel, for example `Pop 3   Housing 5   Food 21/33 (+2)   Granary 0/65 (+2)   Culture 0/10 (+2)`.

Simplifications: there are no districts, specialists, gold or loyalty; the city cannot buy tiles or repair walls; a city's own culture comes only from its buildings; and the city never builds units. The 2-food-1-production centre minimum and the rule that enemy units block worked tiles come from Civ VI play rather than a source found while writing this.

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
- **Yields** are attached to every tile (`cell.yields`, e.g. `{'food': 2, 'production': 1}`) and shown as dots near the top of each tile: green food, orange production, yellow gold. Press **Y** to hide or show them. Yields matter only with `--economy`, where the city collects them; they never change combat or rewards.

By default every map is grassland, forest, and mountains, exactly as before. `--land-types` splits the map into regions around randomly placed centres. `--hills-density` raises that share of flat land into hills; the city's tile stays flat. `--water-density` turns that share of tiles into water, which counts against the same 0.8 limit as mountains and forests. Water next to land becomes coast and water further out becomes ocean. Rivers never run alongside water.

### Example scenarios

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

For a city that grows and builds while you besiege it:

```powershell
python run_game.py --episodes 200 --unit-count 3 --economy --city-population 2 --build-queue ancient_walls monument --land-types plains grassland --hills-density 0.2 --model-path runs/economy/model.json --metrics-dir runs/economy --fresh
```

For a city with Ancient Walls:

```powershell
python run_game.py --episodes 200 --unit-count 4 --unit-types spearman spearman archer battering_ram --walls ancient --model-path runs/ancient-walls/model.json --metrics-dir runs/ancient-walls --fresh
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

## How the AI works

Evaluation compares three policies: the **learned** squad policy, a rule-based **heuristic**, and **random** play.

**What the learner sees.** For each unit, the tactical observation records its type, its fewest turns to the city along a route around the squad's other units (capped at six), its health level, and its exact chance of dying to the city's retaliation if it attacks now (none, unlikely, plausible, likely). That chance is computed from the combat formula, including the defense of forests and hills and a river's defense bonus for the city. It also records the city's health in fifths, the walls (none, damaged, intact), how many allies stand next to the city, how many attackers are still away, whether a battering ram is in place, and whether the city shoots and heals. For each command, the observation also records the tactic it performs from the unit's current position: `ATTACK`, `SHOOT`, `HEAL` (holding while wounded), `WAIT` (holding at full health), `ADVANCE`, `SIDESTEP`, or `RETREAT`. Nothing depends on absolute coordinates, the map size, or the seed, so experience from one generated map applies to the next.

**How it learns.** A joint action's value is the sum of each living unit's value for its tactic: a tabular value-decomposition network. The squad still picks the best *legal joint* action, so collisions and sequential movement are respected, while units of a type share what they learn. Each unit's value is a sum of five tables over subsets of its features (tile coding), so situations never seen in full still get estimates from coarser tables. Each update moves the joint value the same fraction of the way to its target however many units share table entries, which keeps four-unit squads stable. Several training details keep the greedy policy from stalling:

- Rewards are shaped with the squad's health as a potential, which credits healing immediately without changing which policy is optimal.
- An episode cut off by `--max-steps` bootstraps from its last state instead of being treated as a terminal state, so running out the clock does not look free.
- A turn that changes nothing, or a loop the policy's own greedy choices would repeat, is valued as repeating forever. This replaces a slow self-referential update.
- Once the scenario stops changing, the learning rate decays from 0.1 by 0.998 per episode to a floor of 0.01, so the learned policy settles. A curriculum keeps full-sized updates until it reaches its final city strength.

`--shaping`, `--learning-rate-decay`, and `--learning-rate-min` adjust these (a decay of 1 disables annealing), and `--agent joint` trains the original joint-action Q-table for comparison. Hard sieges typically need 500–1,000 training episodes; small scenarios learn in a few hundred.

**Demonstrations.** By default, 30% of training episodes are demonstrations: the heuristic described below plays them from start to finish, and the learner updates from every move with its usual Q-learning step. Q-learning is off-policy, so the learner learns the value of play it only watches, including coordinated assaults that its own exploration rarely completes. Demonstrations pause while the last 100 of them scored less on average than the learner's own last 100 episodes (judged once there are at least 20 of each), so a heuristic that plays worse than the learner stops teaching it. The share does not decay: in experiments, once demonstrations stopped, the learner's own exploratory play pulled the values they had taught back down within a few hundred episodes. Whether an episode is a demonstration is drawn from a random stream derived from its seed, so the draw uses none of the learner's or the game's randomness, and a resumed run makes the same choices. `--demonstrations` sets the share; `--demonstrations 0` trains exactly as before demonstrations existed, and `--no-guidance` turns them off together with guided exploration. The joint learner trains without demonstrations unless asked. The learned policy's evaluation never uses the heuristic.

**The heuristic** uses the same combat odds. Units close in by the fewest turns, archers and slingers shoot when in range, and a melee attack is made only when the retaliation cannot kill the attacker while the city is encircled (a small risk is accepted while the city can still heal), unless a capture is nearly certain. Otherwise units heal in place, hold next to the city until enough of them stop it from healing, and wait outside a ranged city's reach until the squad closes up. During training it also guides a share of exploratory moves and plays the demonstrations.

**Results.** Win rates over 100 held-out evaluation games (seeds 100007–100106) after 1,000 training episodes with default settings. "Learned" is the mean over training seeds 1–4, with the worst seed's rate in parentheses when it is lower. "Without demonstrations" is the same learner trained with `--demonstrations 0`, which reproduces the previous version exactly. "Original heuristic" is the rule-based policy this project shipped before; losses are shown where any occurred.

| Scenario | Settings | Learned | Without demonstrations | Heuristic | Original heuristic | Random |
| --- | --- | --- | --- | --- | --- | --- |
| default | 8×8, three warriors, normal city | 100% | 100% | 100% | 93%, 7% lost | 0% |
| hard | `--difficulty hard` | 100% | 98.5% (94%) | 100% | 0%, 100% lost | 0% |
| curriculum | `--difficulty curriculum` | 98% (97%) | 96.3% (87%) | 100% | 0%, 100% lost | 0% |
| walled siege | warrior, spearman, battering ram; `--wall-hp 50` | 100% | 100% | 99%, 1% lost | 58%, 22% lost | 0% |
| mixed terrain | warrior, archer, warrior; walls 50; forest 0.2, mountains 0.1 | 100% | 100% | 100% | 87% | 0% |
| hard terrain | hard; forest 0.2, mountains 0.1 | 99.5% (99%) | 99.8% (99%) | 93% | 0%, 100% lost | 0% |
| chariot and ram | heavy chariot, battering ram; walls 30 | 99%, 1% lost | 100% | 99%, 1% lost | 99%, 1% lost | 0% |
| slingers | warrior, slinger, slinger | 100% | 100% | 100% | 90% | 0% |
| four units | four warriors, 10×10 | 100% | 100% | 100% | 100% | 0% |
| easy small | easy, 7×7, two warriors, light terrain | 100% | 100% | 100% | 100% | 3% |
| ranged city | `--ranged-strength 25` | 0%, none lost | 0%, none lost | 0%, 100% lost | 6%, 94% lost | 0%, 94% lost |
| ranged city, archers | ranged 25; warrior, archer, archer | 40.8% (37%), 1.5% lost | 0%, none lost | 67%, 33% lost | 26%, 74% lost | 0%, 96% lost |

Outcomes vary between training seeds, so the hard, curriculum, and both ranged-city scenarios were also trained on seeds 5–8. Over all eight seeds, demonstrations raise hard sieges from 95.9% (worst seed 87%) to 99.6% (97%) and the curriculum from 94.9% (85%) to 96.1% (81%). The one cost is in the chariot-and-ram scenario: every model now loses the single evaluation game that the heuristic also loses, and over 1,000 further games one model lost 0.8% of them against none without demonstrations. With the city on its last hit point, it attacks at the same small risk of dying to the retaliation that the heuristic accepts, where the learner on its own heals one more turn first. Without demonstrations the learned policy lost no evaluation games in any scenario. The original joint Q-table won none of its evaluation games in the default, hard, walled-siege and mixed-terrain scenarios, which were the ones measured; it is still available as `--agent joint`. A default model file is now about 0.5 MB instead of about 67 MB.

Against a city with a ranged attack and three warriors, the learner still declines the fight and waits out the turn limit. That is the better choice under the game's rewards (average reward −4.0 against −26.1 for the heuristic's assault, which loses every game). Demonstrations stop there within the first 275 episodes, once the learner's own episodes outscore the heuristic's, and no evaluation game was lost on any of the eight seeds. With archers, the heuristic's assault is worth more (−1.2 against −4.8 for waiting), but the learner never finds it on its own: near a city that shoots, random exploration is punished, so it learns to stay out of range before it ever completes an assault. Watching complete assaults changes that. Over eight training seeds it now wins 40.4% of evaluation games (worst seed 37%) and loses 2.4%, and its average reward rises from −4.8 to −3.2. It still trails the heuristic (67% won, 33% lost). When an assault stalls, it pulls back wounded and waits out the clock (those games average −11.9). Its units coordinate only through their own features, which cannot tell an archer shooting from two hexes away from one far off, and it picks up some of the heuristic's habits, such as healing in place next to a city that is shooting.

## Metrics and playback

Each run writes to `--metrics-dir` (default `training_metrics`):

- `training.csv`: episode-level training metrics, including reward, win/termination outcome, length, Q-table size (table rows for the tactical learner), guidance usage, and whether the heuristic played the episode as a demonstration.
- `metrics.json`: scenario and learning settings, seeds, training history, and evaluation summaries for comparing runs.
- `learning_curve.svg`: training curves that open in a browser without extra plotting packages. The reward, win-rate, and length panels show only the learner's own episodes, not demonstrations; the training log's average reward and win count leave demonstrations out too.

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

The tests cover simulation rules, independent environments, terminal rewards, legal actions, Q-learning updates, checkpoint validation/resumption, and runner behavior. `tests/test_tactical.py` checks the combat odds against simulated attacks, the tactical encoding, the heuristic's rules, and the tactical learner's arithmetic and checkpoints. A short training regression checks that the learner wins held-out games without guidance. Runner tests check that demonstrations are played entirely by the heuristic and train the learner through its normal update, that they pause while they score below the learner's own episodes, that a run with demonstrations split by a checkpoint matches an uninterrupted one, and that older checkpoints resume without them. GitHub Actions runs the suite on Python 3.12 with headless dependencies and executes a short CLI training/evaluation run. [UPGRADE_STATUS.md](UPGRADE_STATUS.md) tracks the upgrade areas and their integration evidence.
