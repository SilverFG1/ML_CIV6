# Real Civilization VI: first integration milestone

The optional `run_civ6.py` entry point connects to a loaded Civilization VI game, captures structured observations, and previews or executes one unit order. The first tactical baseline fortifies a wounded unit or moves a healthy unit to a supplied adjacent tile. It uses local Python logic and requires no model API key.

The existing simulator and `run_game.py` continue to work with their original dependencies. Its trained Q-table uses a different observation/action schema. Real-game training, autonomous empire management, turn completion, scenario resets and replacement of native AI opponents are outside this milestone.

## Install and connect

The supported bridge setup is **Steam Civilization VI with Gathering Storm**, Python 3.12+, and the Civilization VI SDK on Windows. Install the optional dependency into your activated Python environment:

```powershell
python -m pip install -r requirements-civ6.txt
python run_civ6.py doctor
```

The requirements file pins [civ6-mcp](https://github.com/lmwilki/civ6-mcp) to [commit dd2019056371b92ea4854e879ddf05a8cad95e8a](https://github.com/lmwilki/civ6-mcp/commit/dd2019056371b92ea4854e879ddf05a8cad95e8a), version 1.1.11. It is an MIT-licensed external dependency. This adapter imports only its connection and Lua query modules; it does not start the MCP server, web dashboard, popup handlers, or AI providers.

1. Install **Sid Meier's Civilization VI SDK** from the Steam Library's **Tools** filter.
2. Enable **Tuner (disables achievements)** in Civ VI's gameplay options, then restart the game.
3. Load a single-player Gathering Storm game and reach your active turn.
4. Close the SDK's `FireTuner.exe` and any other Civ VI bridge. The game accepts one tuner connection at a time.
5. Run these commands from the repository using native Windows Python, rather than WSL.

The adapter connects only to `127.0.0.1:4318`. It does not launch the game or change game configuration. A failed handshake can leave FireTuner stuck; if a correctly configured connection still fails, restart Civ VI before trying again. `doctor` reports Python dependency status and setup steps without opening a tuner connection.

## Capture the real game

```powershell
python run_civ6.py observe --output runs/civ6/snapshot.json
python run_civ6.py observe --center 4 4 --radius 2
```

Each snapshot contains the current turn, local player, turn activity, owned units, empire overview, and currently visible map tiles around the selected center. The default center is the first owned unit. Radius is bounded to 0–4. Hidden and previously revealed but currently unseen tiles are omitted to avoid exposing current information through the debug interface.

Use `units[].index` for commands. This is Civ VI's per-player unit index, which differs from the composite `unit_id`. Unit health is normalized to a percentage in `hp`.

Owned units also retain their native `unit_type`, runtime `max_moves`, and `base_combat_strength`. Scouts expose `base_sight_range` and a list of current exploration candidates in `scout_moves`. Native IDs identify units even when names are localized or a unit has been renamed.

Snapshots use the versioned `civ6.live.v1` JSON schema. They are data captures, not game save files, and cannot restore a game.

## Terrain types

Visible tiles now include a `terrain_info` object alongside the original `terrain`, `feature`, yields and other plot fields. The bundled catalog imports all **17 base-game terrain types**: grassland, plains, desert, tundra and snow in flat, hill and mountain variants, plus coast and ocean. Features remain a separate layer, so a forest on plains hills retains both the hills and forest information. The catalog also includes the 18 feature definitions in the base game's `Features.xml`.

The annotation exposes the exact terrain/feature identifiers, base terrain family, hills, mountains, water, shallow water, static impassability, and separate terrain/feature movement and defense values. Mountains remain marked impassable even though their base movement cost is 1. Coastal land is distinguished from coast water. Unknown terrain and feature identifiers from DLC or mods are retained with `known_terrain`/`known_feature` set to false and unknown rule values left null.

Import a fresh catalog directly from your installation without running Civ VI:

```powershell
python run_civ6.py import-terrain
```

The default Steam installation path is used on Windows. For a different installation, or a direct terrain XML file:

```powershell
python run_civ6.py import-terrain --game-path "D:\SteamLibrary\steamapps\common\Sid Meier's Civilization VI" --output runs/civ6/terrain_types.json
python run_civ6.py observe --terrain-catalog runs/civ6/terrain_types.json --output runs/civ6/snapshot.json
```

`--terrain-catalog` is also accepted by `play` and `order`. Without it, the bridge uses the bundled base-game catalog automatically. Existing `civ6.live.v1` snapshots remain valid; the added tile field does not change the simulator's terrain encoding or saved Q-tables.

Catalog properties are **base-game reference data**, labeled with `rules_scope`. Importing installed XML does not establish which DLC or mods are active and does not apply their SQL/XML updates. Movement costs can also change with roads, rivers, unit abilities and other game rules. The bridge preserves observed plot fields separately and continues to let the live engine decide whether an order is legal. Hidden tiles remain excluded before terrain annotation.

## Preview and execute a decision

After inspecting the unit's real coordinates, supply an **adjacent, currently visible** destination. The coordinates and index below are examples; replace them with values from your game:

```powershell
python run_civ6.py play --unit-index 0 --target 5 4
python run_civ6.py play --unit-index 0 --target 5 4 --execute
```

The baseline fortifies at or below 50% HP, fortifies a unit already on the goal, and otherwise proposes movement. `--heal-below 30` changes the health threshold. This is a first scripted baseline, not a learned combat policy or a pathfinding system.

Explicit commands use the same preview/execute flow:

```powershell
python run_civ6.py order --kind move --unit-index 0 --target 5 4
python run_civ6.py order --kind attack --unit-index 0 --target 5 4 --execute
python run_civ6.py order --kind fortify --unit-index 0 --execute
python run_civ6.py order --kind skip --unit-index 0 --execute
```

Move commands are limited to one adjacent visible tile and exclude city destinations and foreign units; combat requires an explicit attack order against a visible hostile unit. City attacks and capture are future work. The game engine performs the final operation checks. Turn, local player and exact unit identity are checked again when executing, and guarded inside the submitted Lua. Attack legality and range come from the engine. Units on other players' turns, stale observations, hidden targets and incompatible orders are rejected.

The command previews express intent; they do not reserve a move or prove that the engine will accept it. A command can become invalid between observation and execution.

The controller never ends turns. Resolve production, research, popups, and other decisions in the game yourself during this milestone.

## Scouts

The bridge recognizes owned **`UNIT_SCOUT`** units and provides a single-decision exploration policy. The base game's scout has 3 movement, 2 base sight and 10 base combat strength. Snapshots read the unit's runtime movement and game definitions rather than replacing them with these reference values. Base sight is reported separately from actual visibility, which promotions and terrain can change.

Capture state, find the scout's `index`, then preview a decision:

```powershell
python run_civ6.py observe --output runs/civ6/snapshot.json
python run_civ6.py scout --unit-index 0
python run_civ6.py scout --unit-index 0 --execute
```

The command controls an existing scout; city production and spawning units are outside its scope. It returns one ordinary move or fortify order. It does not enable native automated exploration, queue a route across multiple turns, attack, or end a turn.

The game supplies up to six adjacent, currently visible land destinations that the scout can reach with its remaining movement. Candidates exclude impassable plots, cities and incompatible occupants. Movement eligibility uses the live engine, including current terrain costs, instead of treating every scout as able to move exactly three tiles.

For each candidate, the bridge counts unrevealed tiles within the scout's base sight radius (bounded to four for scoring) using visibility masks. This is a discovery estimate; the query does not read hidden terrain, resources or units. Visible hostile military units on or next to a candidate make it unsuitable for the exploration policy. Peaceful and invisible units are not reported as threats. The policy prefers the largest positive discovery estimate, with coordinate order breaking ties consistently.

A scout at or below `--heal-below` HP (default 50) fortifies. It also fortifies when no candidate offers a suitable unexplored frontier. A unit with no movement remaining produces an error instead of attempting an action. If the game's movement-query APIs are unavailable, the bridge reports the failure rather than guessing a route.

Every executed scout move is checked again against fresh game state and uses an ordinary movement request without the upstream helper's automatic attack modifier. Completion still requires readback; a request acknowledgement alone does not prove movement occurred.

Use a saved snapshot for an offline preview:

```powershell
python run_civ6.py scout --snapshot runs/civ6/snapshot.json --unit-index 0
python run_civ6.py scout --snapshot examples/civ6_scout_snapshot.json --unit-index 0
```

The example is synthetic and includes a higher-scoring destination with a visible threat, demonstrating why the policy chooses a different tile. `--snapshot` cannot be combined with `--execute`. Older snapshots still load, but a fresh capture is required to identify scouts and obtain engine-derived candidates. Renaming a warrior to "Scout" does not make it eligible. Other recon unit types remain available through generic unit commands; this policy currently selects `UNIT_SCOUT` specifically.

## Offline planning

Planning from a snapshot works without Civ VI or the optional bridge dependency:

```powershell
python run_civ6.py plan --snapshot runs/civ6/snapshot.json --unit-index 0 --target 5 4
```

For an immediately runnable demonstration, use the clearly labeled synthetic fixture:

```powershell
python run_civ6.py plan --snapshot examples/civ6_snapshot.json --unit-index 0 --target 5 4
```

Offline plans are never executable through this command. Live commands always capture current game state.

## Results and logs

Commands write JSON to stdout and append timestamped events to `runs/civ6/transitions.jsonl`. Override the location with `--journal PATH`. Observation records contain the snapshot; execution records preserve intent before submission and then the acknowledgement and observed result. This gives a starting dataset for a later real-game learning adapter, without pretending the simulator's action indices apply here.

Result statuses distinguish `preview`, `verified`, `requested`, `rejected`, and `unknown`. A request acknowledgement does not establish that an animation completed or an attack dealt damage. Readback can verify movement and exhausted movement points; attack and fortification outcomes remain explicitly unconfirmed when they cannot be established from the observation.

The adapter disables the upstream transport's automatic command replay, uses per-order Lua tokens as an additional guard, and bounds connections and operations with `--timeout SECONDS` (default 10). If the connection drops after submission, the result can be `unknown`. Inspect the unit before deciding whether to issue another command. There is no automatic resubmission.

## Verification and remaining work

```powershell
python -m unittest discover -s tests -v
python run_civ6.py doctor
python run_civ6.py plan --snapshot examples/civ6_snapshot.json --unit-index 0 --target 5 4
```

The bridge tests use fake connections and deterministic snapshots. They cover JSON contracts, preview behavior, command guards, transport uncertainty, visible-state filtering, and intent logging without requiring the game. A live `observe` plus a single verified move is the next installation-specific acceptance check; automated tests do not establish that the Lua commands work with every game build or mod combination.

Future milestones can add a compact tactical observation encoder, engine-derived legal-action masks, real-game reward/outcome tracking and training, followed by production/research and broader strategy.
