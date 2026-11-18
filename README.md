# ML_CIV6

A small Civ-inspired hex-grid environment and tabular Q-learning experiment, originally inspired by [this Civilization VI machine-learning video](https://www.youtube.com/watch?v=kjA9LxLx9IQ). It has two modes. In the original **siege**, units move, heal, fight, and attempt to capture a defended city, with Civ VI terrain types and their yields, rivers, city walls (including Civ VI Ancient Walls), ranged units, and difficulty presets. In **[empire mode](#empire-mode)**, you play a whole civilization against a rival: settlers found cities, which grow within their housing and amenities, build units and buildings, and claim tiles; gold, science and a 15-technology tree drive the economy; builders improve tiles, traders lay roads, and units embark or sail the coast. Both are limited simulations rather than the full Civilization VI ruleset.

The default siege learner is a tactical squad Q-learner. After 1,000 training episodes it captures the city in almost every evaluation game on seeds it never trained on, including hard and walled sieges. Part of its training watches the rule-based heuristic play, which teaches it assaults it would not find on its own; see [How the AI works](#how-the-ai-works). Empire mode has its own learner, in which every city, unit and the research queue decide each turn; see [How the empire learner works](#how-the-empire-learner-works).

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
| `--city-count N` | Choose one to four enemy cities; capture all of them for a domination victory (see [Domination victory](#domination-victory)). |
| `--city-position X Y` | Set the enemy capital's map coordinate (default: the map center). Further cities are placed randomly. |
| `--player-capital` | Give the squad its own capital to defend against raiders; losing it loses the game (see [Defending your capital](#defending-your-capital)). |
| `--raid-interval N` | Turns between raids on the player's capital (default 6); 0 disables raids. |
| `--capital-strength`, `--raider-strength` | Set the player's capital (default 28) and raider (default 20) combat strengths. |
| `--unit-strengths N [N ...]`, `--city-strength` | Override unit and city combat strengths. |
| `--ranged-strength` | Configure the city's ranged attack strength. |
| `--wall-hp` | Set the city's initial wall health. |
| `--walls none\|ancient` | Give the city Civ VI Ancient Walls: 100 wall health and a ranged strike. |
| `--buildings ...` | Give the city buildings: Monument, Granary, or with `--economy` and their district, Library, Shrine and Barracks (see [City buildings and districts](#city-buildings-and-districts)). |
| `--economy` | Run the city's economy: it grows, builds, and claims tiles while you besiege it. |
| `--city-population N`, `--build-queue ...` | With `--economy`, set the starting population (1–20, default 1) and what the city builds, in order (default: `monument granary ancient_walls`); districts can be queued too. |
| `--districts campus holy_site encampment` | With `--economy`, districts the city starts with. |
| `--city-luxuries N`, `--city-war-weariness POINTS`, `--city-loyalty N`, `--foreign-city POPULATION DISTANCE` | With `--economy`, the city's amenities, its civilization's war weariness, its loyalty, and a city of yours that pressures it (see [City economy](#city-economy)). |
| `--city-governor NAME`, `--governor-promotion NAME`, `--pantheon BELIEF`, `--city-faith N` | With `--economy`, the city's governor and promotion, and the pantheon its civilization founds once it stores 25 faith. |
| `--city-builders N`, `--city-settlers N` | With `--economy`, up to two builders and two settlers that you can capture. |
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

The environment offers two observations. The tactical learner uses the egocentric encoding described in [How the AI works](#how-the-ai-works). Terrain, rivers, walls, and the economy reach it only through the features and odds they change, so its format never varies. The joint learner (`--agent joint`) uses the raw encoding: normalized absolute city/unit coordinates, row parity, health, survival, melee/ranged strengths, attack range, walls, and the terrain grid. With any of `--land-types`, `--hills-density`, or `--water-density`, each tile instead reports three combat properties: impassable, extra movement cost, and defense bonus, and the raw observation schema includes `:terrain`. With rivers enabled, it also has three river flags per tile, for the tile's east, southeast, and southwest edges, and the schema ends in `:rivers`. `--economy` adds city features and `:economy` (see [City economy](#city-economy)). Each further enemy city adds its own city features (schema `:cities=N`), and `--player-capital` adds the capital and up to four raiders (schema `:capital`). Scenarios without these options keep the earlier observation format, so existing models still load. Legal-action masks account for sequential movement, exclude impossible choices, and restrict dead units to waiting. Joint actions grow exponentially: three warriors produce 343 combinations before masking, and four produce 2,401. Each archer or slinger has eight possible commands instead of seven. The tactical learner still chooses among these joint actions, but its tables store values per unit and tactic, so their size does not grow with the number of joint actions. Use the reported baselines to judge whether training improves the learned policy.

### City buildings and districts

`--buildings` gives the city Civ VI buildings, and `--districts` (with `--economy`) its ancient-era districts. Values follow Gathering Storm:

| Building | Stands in | Effect | Production cost | Requires |
| --- | --- | --- | --- | --- |
| Monument | City centre | +2 culture, +1 loyalty | 60 | — |
| Granary | City centre | +1 food, +2 housing | 65 | Pottery |
| Ancient Walls | City centre | 100 wall health and a ranged strike (see above) | 80 | Masonry |
| Library | Campus | +2 science | 80 | Writing |
| Shrine | Holy Site | +2 faith | 65 | Astrology |
| Barracks | Encampment | +1 production, +1 housing | 80 | Bronze Working |

| District | Cost | Placement | Adjacency bonus |
| --- | --- | --- | --- |
| `campus` | 54 | An owned land tile within 3 hexes of the centre | Science: +1 per adjacent mountain, +1 per 2 adjacent districts |
| `holy_site` | 54 | An owned land tile within 3 hexes of the centre | Faith: +1 per adjacent mountain, +1 per 2 adjacent woods, +1 per 2 adjacent districts |
| `encampment` | 54 | An owned land tile 2–3 hexes from the centre | None |

The city exposes its building totals as `city.building_yields` (e.g. `{'food': 1, 'culture': 2}`), `city.building_housing` and `city.building_loyalty`. Buildings never affect combat, and without `--economy` they change nothing else either. With several enemy cities, every city gets the city-centre buildings and walls; only the enemy capital has districts. A building in a district needs that district, given with `--districts` or queued before it.

- **Placement.** A district goes on the free tile with the best adjacency bonus, then the lowest yields; the city centre counts as a district for adjacency. Citizens no longer work its tile, a district placed on a farm replaces it, and a tile with a unit, civilian or raider on it cannot take one. A starting district claims the land it needs, as if the borders had grown earlier.
- **District limit.** One district, plus one for every three citizens beyond the first (1 at population 1–3, 2 at 4–6, and so on). A starting configuration over the limit is rejected; a queued district waits.
- **The Encampment** blocks movement like a mountain and cannot be attacked, but it is never placed where it would cut any tile off from the city. Each turn it strikes a unit within 2 hexes of it at the city's ranged strength, or 15 (Civ VI's value without ranged units) when the city has none. Battering rams help against Ancient Walls as they do against `--wall-hp`.

The renderer draws an obelisk (Monument) and a small barn (Granary) on the city tile and lists the buildings and their effects in the status line. Each district is outlined in its own colour with a badge (an open book, a star, a shield) and the icons of its buildings.

### City economy

`--economy` runs a Civ VI-style economy for the city you are besieging (with several cities, the enemy capital), at the end of each of its turns:

- **Yields.** The city collects yields from its centre tile (always at least 2 food and 1 production), one worked tile per citizen, its buildings, and its districts' adjacency bonuses.
- **Citizens** work the owned tiles with the most food and production, preferring food on ties. They swap production for food when the city would otherwise starve. They cannot work mountains, the centre, districts, tiles more than 3 hexes out, or **a tile one of your units stands on**, so a siege can starve the city or slow its building.
- **Food.** Each citizen eats 2 food. Surplus fills the food store, and the city grows when it holds 15, 24, 33, 44, … food (15 + 8(n−1) + (n−1)^1.5). If the store would go negative, the city loses a citizen.
- **Housing** is 2, or 3 next to water, or 5 next to a river; a Granary adds 2, Barracks 1, and each farm half a house. Growth halves one citizen below housing, drops to a quarter at or above it, and stops five above it.
- **Production** goes to the first item in the `--build-queue` (default `monument granary ancient_walls`) that the city can build now. An item waits while its district is missing, the district limit is reached or no tile is free, keeping its progress. Overflow carries over to the next item, production is lost when nothing can be built, and items the city already has are skipped. Finished walls arrive at full health and switch on the city's ranged strike.
- **Culture** claims a new tile every 10, then 10 + (6t)^1.3 culture, where t is the number of tiles claimed so far. Borders grow ring by ring out to 5 hexes, workable land with the best yields first within a ring, but never onto another city or your capital. A new city owns its centre and the six tiles around it. A city with a Monument claims its first tile on turn 5 and four tiles in 60 turns.
- **Amenities and mood.** A capital needs no amenities for its first two citizens, one for the third, and one more per two citizens after that; each `--city-luxuries` resource gives one. The balance sets the mood, which scales growth and every yield but food. These are Gathering Storm's tiers; sources differ between expansions on some yield changes.

  | Amenities vs. needed | Mood | Growth | Other yields |
  | --- | --- | --- | --- |
  | +5 or more | Ecstatic | +20% | +20% |
  | +3 to +4 | Happy | +10% | +10% |
  | 0 to +2 | Content | — | — |
  | −1 to −2 | Displeased | −15% | −5% |
  | −3 to −4 | Unhappy | −30% | −10% |
  | −5 to −6 | Unrest | stops | −15% |
  | −7 or fewer | Revolt | stops | −20% |

- **War weariness.** Every combat at the enemy capital adds 16 points to its civilization's weariness (`--city-war-weariness` sets a starting value): a melee attack or shot at it, its ranged strike, or its encampment's. Fighting the other enemy cities does not count, since they belong to other civilizations. Weariness fades by 50 a turn, and every 400 points cost one amenity, but never more than the city needs.
- **Loyalty** (0–100, `--city-loyalty`, default 100) changes every turn. Citizens within 9 tiles add 1 a turn, 10% less per tile away; the capital's own citizens count double, and cities its settlers found add theirs. `--foreign-city POPULATION DISTANCE` adds a city of your civilization, 4–9 hexes away and not drawn on the map, whose citizens subtract loyalty the same way. Pressure is capped at 20 either way. The mood adds +6 (ecstatic) to −6 (unhappy or worse), a Monument +1, and an established governor +8. Below 75 loyalty, growth and yields fall by 25%; below 25, growth by 75% and yields by 50%. At 0 and still falling, the city declares independence: the game ends with the outcome `free_city`, you gain 5 reward, and evaluation summaries count `free_cities`. Civ VI's exact conversion from pressure to loyalty is not documented, so 1 per citizen is an approximation.
- **Governors.** `--city-governor` appoints one at the start, with up to one ancient-era promotion (`--governor-promotion`). It takes effect once established, after 5 turns (3 for Victor). Only effects that can matter here are modelled: Victor's `defense_logistics` lets the city heal while surrounded; Magnus's `surplus_logistics` adds 20% food; Liang's `zoning_commissioner` adds 20% production toward districts; Pingala adds 15% science and culture, and `connoisseur` or `researcher` 1 culture or science per citizen. The other titles (Reyna, Amani, Moksha) only add loyalty. The establishment times and the 20% values come from community sources.
- **Pantheon.** With `--pantheon BELIEF`, the city's faith (holy site adjacency, a Shrine) is stored from `--city-faith`, and once 25 is stored the civilization founds the pantheon. `fertility_rites` adds 10% growth and a free builder; `city_patron_goddess` adds 25% production toward a district while the city has none; `religious_settlements` grows borders 15% faster and brings a free settler. The free civilian appears next to the city when the pantheon is founded.
- **Builders and settlers.** `--city-builders N` and `--city-settlers N` (up to two each) start on free tiles in the territory and move one hex a turn after the squad, around units, raiders, cities and each other. A builder has three charges and farms flat grassland and plains without woods (+1 food, half a house). A settler walks to the nearest tile at least 4 hexes from every city, your capital and earlier settlements, and founds a city there; the new city is drawn and adds loyalty but is not otherwise simulated. Moving onto a civilian captures it; battering rams cannot, and raiders walk around them. The heuristic hunts a civilian within 3 hexes with any unit that can capture it and has at least a third of its health.
- **Rewards.** The economy now feeds the reward: you lose 1 each time the city grows, completes a district or building, or founds its pantheon, and gain 1 each time it starves. Capturing a builder earns 2 and a settler 4, a city founded by a settler costs 5, and independence earns 5.

The economy sets a clock on the siege. Left alone on a map with forests (`--forest-density 0.2 --mountain-density 0.1`), a size-1 city building walls first finishes them in a median 40 turns; on pure grassland, with no production but its centre's, it takes 80. On plains with hills it needs about 27 turns. The rule-based baseline usually takes the city within about 6 turns, before anything is built, so the clock mostly pressures slower policies such as the learner. The raw observation gains the city's population, progress towards growth, production and the next tile, its housing, which item it is building, its amenity balance, loyalty, war weariness, governor and pantheon progress, which items it has built, and each civilian's position, whether it is in play and a builder's charges. The schema ends in `:economy`, plus `:civilians=...` with civilians. Raw models trained before districts and the city systems existed need `--fresh`; the tactical learner's format does not change.

The renderer outlines the city's territory in red, marks worked tiles with a white citizen token and tiles a unit is blocking with a red one, draws farms, civilians (with a builder's charges) and settlers' new cities, and adds an economy line with an icon for each field, for example `Pop 3   Housing 5   Food 21/33 (+2)   Granary 0/65 (+2)   Culture 0/10 (+2)   Amenities 0/1 Displeased   Loyalty 100 (+3)`.

Simplifications: there are no specialists or gold; the city cannot buy tiles or repair walls, never builds units and never trains civilians; Classical-era religion, the Water Mill, the Government Plaza, rising district costs, maintenance and pillaging are not modelled; and only the enemy capital has an economy. Civilians move one hex a turn rather than Civ VI's two. The 2-food-1-production centre minimum and the rule that enemy units block worked tiles come from Civ VI play rather than a source found while writing this. Building costs come from community references, so check them against your game's Civilopedia if exact values matter.

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

### Domination victory

Each enemy city is a rival civilization's capital. Capturing every one is a domination victory and ends the game as a win; with the default single city, this is the same as capturing it. Use `--city-count 2` to `4` for a longer campaign:

```powershell
python run_game.py --episodes 1000 --city-count 3 --max-steps 200 --model-path runs/domination/model.json --metrics-dir runs/domination --fresh
```

- The enemy capital sits at `--city-position` (or the map center). Other cities are placed randomly at least three hexes apart, in reach of the squad, with enough open tiles around them for a full siege. City tiles stay flat, open land on every map.
- Every city that has not been captured shoots and heals on its own. All cities share the strength, wall, building and ranged settings. With `--economy`, the economy runs in the enemy capital.
- A captured city stops fighting and its tile opens to your units. Capturing a city before the last earns +5, and the last one earns the +10 victory reward.
- Ranged units shoot the closest city in range; ties go to the city with the least health plus walls. A battering ram helps only against the city it stands next to.
- The squad besieges one city at a time: the weakest, then the one closest to the squad. The tactical observation describes each unit relative to that city, and routes go around the other cities, so a model trained against one city can play several.
- Evaluation summaries add the average number of cities captured, which shows progress even when games time out.

### Defending your capital

Add `--player-capital` to give the squad its own capital, Otterholt, which the enemy raids. Losing it loses the game. It combines with every other siege option, including domination:

```powershell
python run_game.py --episodes 1000 --player-capital --model-path runs/capital/model.json --metrics-dir runs/capital --fresh
```

- The capital is placed as far from the enemy cities as the map allows, at least three hexes from each, and the squad starts around it. Your units cannot enter it.
- It has 100 HP and strength 28 (`--capital-strength`). It hits back when attacked and heals 10 HP at the end of any turn with no raider next to it.
- Every 6 turns (`--raid-interval`; 0 disables raids), an enemy city sends a raider (strength 20, `--raider-strength`) from a tile beside it. With several enemy cities, they take turns, so each city you capture slows the raids. At most four raiders are on the map at once.
- Raiders march toward the capital by the fewest turns and attack it once they reach it. They step around your units when they can, and attack a unit that blocks every way forward.
- Move into a raider to attack it. Archers and slingers can also shoot raiders, and their shots can kill. Battering rams cannot fight raiders.
- Damage to the capital costs 3 reward per 100 HP, damage dealt to raiders earns 0.5 per 100 HP, and losing the capital is a −10 loss.
- A raider threatens the capital when the raiders within three hexes of it could together destroy it, with a 25% margin, unless the city being besieged is nearly taken. Then the heuristic's fighting units turn back for the nearest one, and the tactical observation describes them relative to that raider instead of the city. A raider has no walls and neither shoots nor heals, and attacking it is `ATTACK`. The tactical schema ends in `:capital`, so models trained with and without a capital are not interchangeable.
- Evaluation summaries show how many losses came from losing the capital.

Over 100 games of up to 200 turns with default settings otherwise, the heuristic wins 99% against one city with a capital to defend, 99% against three cities, and 65% against three cities with a capital to defend, where it loses the capital in 16% of games and runs out of turns in 12% (94% and 4% on `--difficulty easy`). Left undefended by random play, the capital falls in every game, around turn 23. After 1,000 training episodes (training seed 7, 100 held-out games), the tactical learner wins all 100 games against three cities in 200 turns, as the heuristic does, and all 100 with a capital to defend, where the heuristic wins 98 and loses its capital once. Against three cities with a capital to defend it wins 68, loses 16 (13 of them by losing the capital) and runs out of turns in 16, where the heuristic wins 60 on the same seeds.

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

For a city with districts, civilians, a governor and a pantheon:

```powershell
python run_game.py --episodes 300 --unit-count 3 --unit-types warrior spearman battering_ram --economy --city-population 4 --districts campus --buildings monument library --build-queue ancient_walls encampment barracks --city-builders 1 --city-settlers 1 --city-governor pingala --pantheon fertility_rites --forest-density 0.3 --model-path runs/districts/model.json --metrics-dir runs/districts --fresh
```

For a city whose loyalty a large city of yours wears down:

```powershell
python run_game.py --episodes 300 --unit-count 3 --economy --city-population 3 --foreign-city 12 4 --model-path runs/loyalty/model.json --metrics-dir runs/loyalty --fresh
```

For a city with Ancient Walls:

```powershell
python run_game.py --episodes 200 --unit-count 4 --unit-types spearman spearman archer battering_ram --walls ancient --model-path runs/ancient-walls/model.json --metrics-dir runs/ancient-walls --fresh
```

For a walled siege with a battering ram:

```powershell
python run_game.py --episodes 200 --unit-count 3 --unit-types warrior spearman battering_ram --wall-hp 50 --model-path runs/siege/model.json --metrics-dir runs/siege --fresh
```

## Empire mode

`--mode empire` plays a whole civilization instead of a single siege. You (the learner) and a rival played by the rule-based heuristic each start with a settler and a warrior, found cities, grow them, research technologies, trade, and fight. With a rival, the map is mirrored top to bottom, so neither start is favoured, and the two civilizations alternate moving first each turn.

```powershell
python run_game.py --mode empire --episodes 1000 --eval-episodes 100 --model-path runs/empire/model.json --metrics-dir runs/empire --fresh
```

A game ends when one side holds every rival's original capital (domination), when a civilization loses its last city and settler, or at the turn limit (default 100), where the higher score wins and an exact tie is reported as a timeout. The score is 5 per city, 1 per citizen, 2 per technology and 1 per building, loosely after Civ VI's. Each turn's reward is a tenth of the change in your score, ±0.5 for each military unit killed or lost (±0.25 for civilians), ±3 for each city captured or lost, and ±10 for winning or losing.

| Option | Purpose |
| --- | --- |
| `--mode empire` | Play the empire game (a checkpoint keeps its mode). |
| `--rivals 0\|1` | Play against one rival (default) or alone. A rival needs an odd `--height`. |
| `--map-type continents\|islands` | One landmass connecting both starts (default), or one island each, separated by a row of coast. |
| `--turn-limit N` | Turns before the higher score wins (10–500, default 100). `--max-steps` defaults to it. |
| `--speed online\|quick\|standard` | Civ VI game speed: production, research, growth and border costs at 50%, 67% or 100% (default online, which suits 100 turns). |
| `--difficulty easy\|normal\|hard\|curriculum` | The rival's production, science and gold multiplier: 0.75, 1.0, 1.25, or a ramp from 0.5 to 1.25 over `--curriculum-episodes` (default 1,000). `--rival-bonus` sets it directly. |
| `--resource-density` | Share of tiles with a resource (0–0.5, default 0.15). |
| `--width`, `--height`, `--land-types`, `--hills-density`, `--forest-density`, `--mountain-density`, `--water-density`, `--river-count` | Map generation, as in the siege (defaults 14×13, grassland and plains, 0.15 hills, 0.15 forest, 0.05 mountains, 0.05 water, 2 rivers). With a rival, the top half, its rivers included, is copied onto the bottom half. |

Siege-only options such as `--unit-types` or `--economy` are rejected in empire mode, and empire options in a siege.

### Cities, housing and amenities

- **Settlers** found cities on open land at least four tiles from every city and outside foreign territory. A city starts with its tile and the six around it; the first is the capital, whose Palace adds 2 production, 2 science, 5 gold and 1 culture. A settler can only be started in a city of two or more citizens, costs a citizen when it completes, and costs 30 more (at standard speed) for each settler the civilization has already trained.
- **Citizens** work the city's best tiles, as in the siege economy, and cannot work a tile an enemy unit stands on. Each citizen also produces half a science.
- **Housing** is 2, 3 next to water, or 5 next to a river, plus 2 for a Granary and half a house for every farm, pasture, plantation or fishing boats in the city's territory. Growth slows near the housing limit as in the siege economy, counting half houses: with 5.5 housing, a city of 4 grows at the full rate and one of 5 at half.
- **Amenities.** A city needs one amenity for every two citizens, the first at three. Each improved luxury type supplies one amenity to up to four of its owner's cities, neediest first. The surplus sets the city's mood:

| Surplus | Mood | Growth | Non-food yields |
| --- | --- | --- | --- |
| +3 or more | Ecstatic | +20% | +10% |
| +1 to +2 | Happy | +10% | +5% |
| 0 | Content | — | — |
| −1 to −2 | Displeased | −15% | −5% |
| −3 to −4 | Unhappy | −30% | −10% |
| −5 to −6 | Unrest | none | −15% |
| −7 or less | Revolt | none | −20% |

The tiers follow Civ VI's base game; the percentages are approximate, and unrest and revolt raise no rebels here.

- **Production** completes the city's chosen item; overflow carries over. Gold buys the item in production outright at 4 gold per point of its cost, and production already spent on it is lost.
- **Culture** claims tiles up to three away as in the siege economy; a tile belongs to one city.
- **Gold** pays 1 per military unit and 1 per Granary, Water Mill and Library each turn. A civilization whose treasury goes negative disbands one unit that costs maintenance each turn.

### Research

Science researches 15 technologies of Civ VI's Ancient and Classical eras, with their base costs (standard speed) and prerequisites. A boost (Eureka) pays half of a technology when its event happens, before or during research. Overflow carries to the next technology.

| Technology | Cost | Needs | Boost | Unlocks here |
| --- | --- | --- | --- | --- |
| Pottery | 25 | — | — | Granary |
| Animal Husbandry | 25 | — | — | Pasture |
| Mining | 25 | — | — | Mine, Quarry |
| Sailing | 50 | — | found a coastal city | Galley, Fishing Boats |
| Irrigation | 50 | Pottery | farm a resource | Plantation |
| Writing | 50 | Pottery | meet another civilization | Library |
| Archery | 50 | Animal Husbandry | kill a unit with a slinger | Archer |
| Masonry | 80 | Mining | build a quarry | Ancient Walls, Battering Ram |
| Bronze Working | 80 | Mining | — | Spearman |
| The Wheel | 80 | Mining | mine a resource | Heavy Chariot, Water Mill |
| Currency | 120 | Writing | establish a trade route | Market, Trader |
| Horseback Riding | 120 | Animal Husbandry | build a pasture | Horseman |
| Iron Working | 120 | Bronze Working | mine iron | Swordsman |
| Shipbuilding | 200 | Sailing | own two galleys | Quadrireme, embarking |
| Engineering | 200 | The Wheel | build Ancient Walls | Roads bridge rivers |

Bronze Working's boost needs barbarians, which this simulation does not have. Two players meet when any of their units or cities come within three tiles.

### Units, buildings and tile improvements

| Unit | Cost | Strength | Ranged (range) | Moves | Needs |
| --- | --- | --- | --- | --- | --- |
| Settler | 80 (+30 each) | — | — | 2 | 2 citizens |
| Builder | 50 | — | — | 2 | 3 charges |
| Trader | 40 | — | — | 2 | Currency |
| Warrior | 40 | 20 | — | 2 | — |
| Slinger | 35 | 5 | 15 (1) | 2 | — |
| Archer | 60 | 15 | 25 (2) | 2 | Archery |
| Spearman | 65 | 25 | — | 2 | Bronze Working |
| Heavy Chariot | 65 | 28 | — | 2 (+1 starting on flat land) | The Wheel |
| Horseman | 80 | 36 | — | 4 | Horseback Riding, improved Horses |
| Swordsman | 90 | 35 | — | 2 | Iron Working, improved Iron |
| Battering Ram | 65 | — | — | 2 | Masonry |
| Galley | 65 | 30 | — | 3 | Sailing, coastal city |
| Quadrireme | 120 | 20 | 25 (1) | 3 | Shipbuilding, coastal city |

Costs are standard-speed values; the default online speed halves them. In this mode the archer uses Civ VI's 15/25 instead of the siege's 12/20.

| Building | Cost | Effect | Needs |
| --- | --- | --- | --- |
| Monument | 60 | +2 culture | — |
| Granary | 65 | +1 food, +2 housing | Pottery |
| Water Mill | 80 | +1 food, +1 production | The Wheel, a river next to the city |
| Library | 90 | +2 science | Writing |
| Market | 120 | +3 gold, +1 trade route | Currency |
| Ancient Walls | 80 | 100 wall health and a ranged strike | Masonry |

Civ VI builds Libraries and Markets in Campus and Commercial Hub districts; there are no districts here, so they are city buildings.

Builders spend a charge on an improvement in their civilization's territory: a **farm** (+1 food) on flat or hilly grassland or plains without a feature, a **mine** (+1 production) on hills without a feature (Mining), or the improvement a resource needs. Farms, pastures, plantations and fishing boats add half a house.

| Resource | Kind | Yield | Improvement |
| --- | --- | --- | --- |
| Wheat, Rice | bonus | +1 food | Farm |
| Cattle; Sheep (hills) | bonus | +1 food | Pasture (+1 production) |
| Stone | bonus | +1 production | Quarry (+1 production) |
| Fish; Crabs | bonus | +1 food; +2 gold | Fishing Boats (+1 food) |
| Horses | strategic | +1 food, +1 production | Pasture |
| Iron (hills) | strategic | +1 science | Mine |
| Citrus; Cotton; Silk (forest) | luxury | +2 food; +3 gold; +1 culture | Plantation (+2 gold) |
| Salt | luxury | +1 food, +1 gold | Mine |
| Pearls | luxury | +1 faith | Fishing Boats |

A builder places fishing boats on a coastal tile next to the land it stands on.

### Movement, roads and ships

Units have Civ VI's movement points: flat land costs 1, hills and forest one more each, and crossing a river without a bridge, embarking or disembarking uses all remaining movement. A unit with full movement can always enter one tile. Entering a tile next to an enemy military unit ends the move (zone of control). One military and one civilian unit may share a tile; a military unit that enters a tile holding only enemy civilians captures them.

**Roads** cost one movement point between road tiles, whatever the terrain, and bridge rivers once Engineering is known. City tiles count as roads. Civ VI lays roads along trade routes, and so does this simulation: a **trader** opens a domestic route from its nearest city to the nearest other city it can reach over land, lays a road along the way, and becomes the route. Each route adds 1 food and 1 production to its origin city for the rest of the game. A civilization has one route, plus one per Market.

**Ships and embarking.** Galleys and quadriremes sail coast tiles and their own coastal cities; nothing enters ocean, which needs Cartography in Civ VI. With Shipbuilding, land units embark onto the coast. Embarked units defend with strength 10, cannot attack, and do not heal. Ships can attack units on the shore and coastal cities, and a galley can capture a coastal city. On an islands map, an invasion needs Shipbuilding or a fleet.

### Combat and cities

Combat uses the siege's damage formula. A unit loses one strength for every 10 hit points of damage, as in Civ VI. Defenders gain the terrain's defense (hills and forest +3 each) and +5 against melee attacks across a river; spearmen gain +10 against cavalry. Ranged attacks draw no retaliation. A melee unit that destroys a land unit moves into its tile.

A city has 100 health and defends with its owner's strongest available melee unit's strength plus 8 (28 with warriors, like the siege's normal city). It heals 10 health in any turn it is not damaged. Walls, rams and ranged attacks work as in the siege: melee attacks deal half damage to walls unless a ram stands next to the city, and ranged attacks cannot take a city's last hit point. Walled cities strike the weakest enemy unit within two tiles each turn with the owner's strongest ranged strength (15, or 25 with archers). A melee unit, or a galley, that brings a city to zero health captures it: the city keeps its territory, buildings and improvements, loses a citizen and its walls, drops to 50 health, and any garrison dies. A captured capital no longer has a Palace.

Units heal 20 a turn in their own city, 15 in friendly territory, 10 in neutral and 5 in enemy territory if they neither moved nor attacked; ships heal only in friendly territory.

### What the empire learner decides

Each turn, every entity of your civilization makes one decision, so an observation is a list of decisions whose length changes as cities are founded and units trained or lost. Each decision has a handful of integer features and a menu of orders; the environment carries out each order with pathfinding:

| Decision | Features | Options |
| --- | --- | --- |
| Research | turns left, technologies known, cities, coastal, threat, military balance, separated by sea | the available technologies (only "carry on" while researching) |
| City | turns left, whether it is choosing, capital, population, housing room, amenities, food surplus, production, nearby sites, settlers, builder need, defenders, threats, coastal, cities, walls, army size, treasury (short of the item's price, within 100 gold above it, or richer) | any buildable item while choosing; otherwise carry on or buy |
| Settler | turns left, this tile's quality, best site's quality and distance, threat, escort, cities | found here, head for the 1st/2nd/3rd best site, hold |
| Builder | turns left, what it could improve here, targets, distance, charges, threat | improve, head for a resource, head for a tile, hold |
| Land unit | turns left, type, health, threat, what it can attack now, the odds, distance to an enemy city, military balance, territory, whether it is a city's only defender, unescorted settlers | fortify, garrison, defend, hunt, siege, escort |
| Ship | turns left, type, health, threat, contact, odds, enemy coastal city, balance | fortify, hunt, raid, patrol |
| Trader | turns left, whether a route is available | open a route, hold |

The odds are the exact chances that an attack succeeds and that the retaliation kills the attacker, computed from the combat formula. Nothing depends on map coordinates.

### How the empire learner works

`EmpireQAgent` is the default learner in empire mode. Like the tactical learner, it is tabular and tile-coded, but each decision estimates the whole civilization's return, as independent learners sharing one target would.

- **Values.** A decision's value for an option is a situation value V, which every visit updates, plus an advantage A for the option, which only choosing it updates; each is a sum of table rows over a few subsets of the decision's features. Options are ranked by advantage, so the value level can follow the situation even for options seldom taken. A new row starts V at the target that created it and every advantage half a point below it (`--untried-penalty`), so options never tried look slightly worse, not better.
- **Targets.** The target is the turn's reward plus the discounted mean, over next turn's decisions, of each decision's best value, leaving out the decision's own entity. A unit that holds still therefore cannot bootstrap from its own unchanged estimate. A sum over entities, as in the tactical learner, does not suit a civilization whose number of units changes: it credits training any unit with one more term and blames a builder for spending its last charge or a settler for founding a city. The first version did exactly that, building builders that never improved anything.
- **Demonstrations and teacher labels.** As in the siege, 30% of training episodes are demonstrations played by the heuristic. Thirty or more entities share one reward, so a single unit's choice barely moves its target, and the learned advantages are noisy. The heuristic therefore also teaches with DQfD's large-margin loss (Hester et al. 2018): every option valued within `--demonstration-margin` (default 1.0) of the heuristic's choice is pushed down, while the chosen option keeps its learned value. Demonstrations label their own moves. In the learner's own episodes, the heuristic labels the states the learner actually reaches, as in DAgger (Ross et al. 2011), because the learner's play reaches states that demonstrations never show. The greedy policy follows these labels unless the learner's own experience values another option by more than the margin. `--margin-on demonstrations` limits the margin to demonstrations.
- **Exploration** is per decision: each explores with probability epsilon (35% of those follow the heuristic), so a turn mixes exploration and greedy play instead of randomizing a whole empire at once.

**The heuristic** expands while there are good sites, improves tiles, builds a Monument and the economic buildings its technologies allow, keeps one defender per city and answers threats, then masses an army, with a battering ram against walls and archers in support, and besieges the nearest rival city. It buys settlers, builders and urgently needed defenders, and spends any gold above a 100-gold reserve on whatever a city is building. Its melee units assault a city only when the retaliation is unlikely to kill them or a capture is likely. It plays the rival, the demonstrations and the "heuristic" baseline.

### Empire results

Held-out results over 100 evaluation games (seeds 100007–100106) against the heuristic rival with default settings (14×13 map, 100 turns, online speed, normal difficulty). "Learned" was trained for 500 episodes, on training seeds 1–3 for continents and seed 1 for islands; ties at the turn limit count as neither a win nor a loss.

These results were measured before a fix to the housing growth rule, which slowed cities with fractional housing (the half houses from improvements) too early: a city of 4 with 5.5 housing grew at a quarter of the full rate. The learned rows have not been re-measured. Re-running the heuristic and random rows on the same seeds after the fix raised the heuristic's average population from 28.1 to 30.7 on continents and from 26.6 to 29.7 on islands, and its score and the rival's by 3–4 points; its win rate moved within noise (56% and 44%), and random play barely changed.

| Map | Policy | Wins | Losses | Score | Rival score | Cities | Population | Gold left |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| continents | learned (3 seeds) | 17%, 20%, 19% | 82%, 78%, 80% | 81.7, 83.2, 85.2 | 97.7–98.4 | 4.8–5.0 | 22.5–23.7 | 164–227 |
| continents | heuristic | 55% | 44% | 99.4 | 97.4 | 5.6 | 28.1 | 105 |
| continents | random | 0% | 100% | 41.3 | 101.0 | 2.0 | 7.3 | 71 |
| islands | learned | 17% | 82% | 80.9 | 97.2 | 4.5 | 21.6 | 323 |
| islands | heuristic | 48% | 49% | 94.8 | 96.7 | 4.8 | 26.6 | 105 |
| islands | random | 0% | 100% | 50.5 | 109.4 | 2.4 | 9.8 | 58 |

The learned policy plays a whole civilization: it expands to about five cities, researches nearly all 15 technologies, improves tiles, trades and fights, and wins about one game in five against the heuristic. That is far better than random play but still below the heuristic it learns from, which wins about half its games against an identical rival. The heuristic's roughly even record against itself is consistent with the mirrored map and alternating turn order being fair. The learned cities grow more slowly (about 23 citizens against 28), and the learner still ends with more unspent gold.

How the learner got here, measured on held-out games against the heuristic rival:

- With values summed over entities, as in the tactical learner, cities trained builders that never improved anything and units only fortified. After 400 episodes it won none of 30 games and ended with almost no cities.
- Averaging targets and splitting values into situation values and advantages raised the score from about 17 to 51–66 (300 episodes, two seeds), but it still won at most 2 of 30 games.
- Leaving each entity's own successor out of its target did not change these results by itself. It removes a feedback loop in which a unit that holds still keeps an overestimate alive.
- The demonstration margin raised wins to 9–11 of 30 (scores 83–89) with a margin of 1.0, and to 14–16% of 100 games after 1,000 episodes. With a margin of 0.5, wins were 2–10 of 30, or 5–10 of 30 with half of all episodes as demonstrations.
- Labelling the learner's own states with the heuristic's choices raised wins on continents to 23–30% after 500 episodes (14% on islands), against the heuristic as it then was, which spent gold only on settlers, builders and emergency defenders. That learner ended games with 540–680 unspent gold.
- With the heuristic also spending its surplus gold, the rival grew stronger, and the learner still hoarded about 580 gold, because a city's features did not show its treasury. The treasury feature brought unspent gold down to 164–227, giving the results above.

Each learned evaluation takes the greedy policy, without the heuristic. A training episode takes about 0.6 seconds.

### Simplifications

There are no civics, policies, governments, religion, great people, districts, wonders, barbarians, diplomacy, trade with rivals, fog of war or tile purchases. Rivals are always at war. Traders become their route instead of travelling it, and routes cannot be plundered. Builders place fishing boats from the shore. Purchases forfeit production already spent. Movement, healing and city values not listed above as Civ VI's are this simulation's own. Map generation is this project's own. With one rival the map is mirrored, so `--height` must be odd.

The renderer draws each civilization's borders in its colour (blue for you, red for the rival), roads, resources (three-letter labels), improvements, cities with their population and health, units and ships with type labels, and an outline round embarked units.

## How the AI works

This section describes the siege. Empire mode's learner is described in [How the empire learner works](#how-the-empire-learner-works).

Evaluation compares three policies: the **learned** squad policy, a rule-based **heuristic**, and **random** play.

**What the learner sees.** For each unit, the tactical observation records its type, its fewest turns to the city along a route around the squad's other units (capped at six; with several cities, the city being besieged, or a raider threatening the player's capital), its health level, and its exact chance of dying to the city's retaliation if it attacks now (none, unlikely, plausible, likely). That chance is computed from the combat formula, including the defense of forests and hills and a river's defense bonus for the city. It also records the city's health in fifths, the walls (none, damaged, intact), how many allies stand next to the city, how many attackers are still away, whether a battering ram is in place, and whether the city shoots and heals. For each command, the observation also records the tactic it performs from the unit's current position: `ATTACK`, `SHOOT`, `HEAL` (holding while wounded), `WAIT` (holding at full health), `ADVANCE`, `SIDESTEP`, or `RETREAT`. Nothing depends on absolute coordinates, the map size, or the seed, so experience from one generated map applies to the next.

**How it learns.** A joint action's value is the sum of each living unit's value for its tactic: a tabular value-decomposition network. The squad still picks the best *legal joint* action, so collisions and sequential movement are respected, while units of a type share what they learn. Each unit's value is a sum of five tables over subsets of its features (tile coding), so situations never seen in full still get estimates from coarser tables. Each update moves the joint value the same fraction of the way to its target however many units share table entries, which keeps four-unit squads stable. Several training details keep the greedy policy from stalling:

- Rewards are shaped with the squad's health as a potential, which credits healing immediately without changing which policy is optimal.
- An episode cut off by `--max-steps` bootstraps from its last state instead of being treated as a terminal state, so running out the clock does not look free.
- A turn that changes nothing, or a loop the policy's own greedy choices would repeat, is valued as repeating forever. This replaces a slow self-referential update.
- Once the scenario stops changing, the learning rate decays from 0.1 by 0.998 per episode to a floor of 0.01, so the learned policy settles. A curriculum keeps full-sized updates until it reaches its final city strength.

`--shaping`, `--learning-rate-decay`, and `--learning-rate-min` adjust these (a decay of 1 disables annealing), and `--agent joint` trains the original joint-action Q-table for comparison. Hard sieges typically need 500–1,000 training episodes; small scenarios learn in a few hundred.

**Demonstrations.** By default, 30% of training episodes are demonstrations: the heuristic described below plays them from start to finish, and the learner updates from every move with its usual Q-learning step. Q-learning is off-policy, so the learner learns the value of play it only watches, including coordinated assaults that its own exploration rarely completes. Demonstrations pause while the last 100 of them scored less on average than the learner's own last 100 episodes (judged once there are at least 20 of each), so a heuristic that plays worse than the learner stops teaching it. The share does not decay: in experiments, once demonstrations stopped, the learner's own exploratory play pulled the values they had taught back down within a few hundred episodes. Whether an episode is a demonstration is drawn from a random stream derived from its seed, so the draw uses none of the learner's or the game's randomness, and a resumed run makes the same choices. `--demonstrations` sets the share; `--demonstrations 0` trains exactly as before demonstrations existed, and `--no-guidance` turns them off together with guided exploration. The joint learner trains without demonstrations unless asked. The learned policy's evaluation never uses the heuristic.

**The heuristic** uses the same combat odds. Units close in by the fewest turns, archers and slingers shoot when in range, and a melee attack is made only when the retaliation cannot kill the attacker while the city is encircled (a small risk is accepted while the city can still heal), unless a capture is nearly certain. Otherwise units heal in place, hold next to the city until enough of them stop it from healing, and wait outside a ranged city's reach until the squad closes up. Against several cities it takes them one at a time, and its fighting units turn back for a raider that could take the player's capital. During training it also guides a share of exploratory moves and plays the demonstrations.

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

- `training.csv`: episode-level training metrics, including reward, win/termination outcome (`win`, `loss`, `timeout`, `quit`, or `free_city` when the city declares independence), length, Q-table size (table rows for the tactical learner), guidance usage, and whether the heuristic played the episode as a demonstration.
- `metrics.json`: scenario and learning settings, seeds, training history, and evaluation summaries for comparing runs.
- `learning_curve.svg`: training curves that open in a browser without extra plotting packages. The reward, win-rate, and length panels show only the learner's own episodes, not demonstrations; the training log's average reward and win count leave demonstrations out too.

Use separate output directories for experiments. Checkpoints save atomically; periodic saves preserve recoverable progress during long runs. `--checkpoint-every N` controls their interval.

Siege evaluations with `--economy` also report the city's average final population and builds completed, and with civilians how many were captured and how many cities they founded; free cities are counted when there are any. Siege evaluations against several cities also report the average number captured, and with `--player-capital` how many losses came from losing the capital, for example `heuristic: wins 65/100, losses 23 (capital lost 16), ..., cities captured 2.4`.

In empire mode, `steps` counts turns, and the `city_strength` column records the episode's rival bonus, which a curriculum ramps. Evaluation summaries also average each policy's final score, the rival's score, cities, population, technologies and gold, for example `learned: wins 9/30, ..., score 83.2, rival score 97.2, cities 4.8, ...`.

Render evaluation after training or replay a saved policy:

```powershell
python run_game.py --episodes 0 --eval-episodes 3 --eval-policies learned --model-path runs/baseline/model.json --metrics-dir runs/playback --render --render-delay-ms 100
```

During playback, **Space** pauses/resumes, **Right Arrow** advances one turn while paused, **+ / -** change playback speed, **Y** toggles yield dots, and **Esc** exits. `--render-training` also displays training turns.

Playback works the same way for empire games (`--mode empire --render`), without manual play.

Run `python game.py` to play a siege manually. **Tab** selects a unit, **Q/E/A/D/Z/X** move, **Space** heals, **F** shoots with an archer or slinger, **Y** toggles yield dots, **R** restarts, and **Esc** exits. Each command advances the whole squad's turn; the other units wait and heal.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The tests cover simulation rules, independent environments, terminal rewards, legal actions, Q-learning updates, checkpoint validation/resumption, and runner behavior. `tests/test_tactical.py` checks the combat odds against simulated attacks, the tactical encoding, the heuristic's rules, and the tactical learner's arithmetic and checkpoints. Game tests cover domination (city placement, capture rewards, every city shooting and healing, shot targeting, rams, the target city) and the player's capital (placement, raids taking turns, raider movement and blockades, healing, losing it, fighting raiders, and when the heuristic defends), and check that every legal command has a tactic in the tactical encoding throughout heuristic games with both. `tests/test_city_systems.py` covers the enemy capital's districts and their placement, adjacency, limit and Encampment, production overflow and district bonuses, builders, settlers and capture, the pantheon's civilian, fractional housing, growth rewards, amenities and moods, war weariness, loyalty and independence, governors, faith and pantheons, and border growth; `tests/test_icons.py` checks that the code-drawn icons cover every unit, district, building, civilian and mood. A short training regression checks that the learner wins held-out games without guidance. Runner tests check that demonstrations are played entirely by the heuristic and train the learner through its normal update, that they pause while they score below the learner's own episodes, that a run with demonstrations split by a checkpoint matches an uninterrupted one, and that older checkpoints resume without them. `tests/test_empire.py` covers the empire's map mirroring, settlers and city sites, housing, amenities, purchases, bankruptcy, borders, research and boosts, strategic resources, improvements, trade routes and roads, movement points, rivers, zones of control, embarking, ships, combat odds against simulated fights, walls, city strikes, capture and domination, and whole random and heuristic games; `tests/test_empire_learning.py` checks the empire learner's arithmetic, its own-entity exclusion, the demonstration margin and its checkpoints; and runner tests check that empire training resumes exactly. GitHub Actions runs the suite on Python 3.12 with headless dependencies and executes a short CLI training/evaluation run. [UPGRADE_STATUS.md](UPGRADE_STATUS.md) tracks the upgrade areas and their integration evidence.
