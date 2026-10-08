# Upgrade status

All ten requested upgrade areas are implemented and verified locally on 8 October 2026 with Python 3.12.4. This project remains a small Civ-inspired combat simulation with a tabular learner; it does not implement the full Civilization VI ruleset.

| # | Area | Implementation | Validation |
| --- | --- | --- | --- |
| 1 | Game-ending logic | End on capture or loss of all units; runner enforces turn limits and distinguishes wins, losses, timeouts, and user quits. Completed games yield no further rewards or turns. | Capture prevents healing/ranged retaliation; all-dead loss and repeated-terminal-step tests pass. Runner win/loss/timeout tests pass. |
| 2 | Reward calculation | Record explicit turn events for damage, healing, death, capture, and movement progress. Reward readings do not consume events; dead units have no ongoing distance penalty. | Multiple attack events, death followed by healing, health caps, and repeated reward reads verified in game tests. |
| 3 | AI evaluation | Evaluate learned, heuristic, and random policies separately on matched scenario seeds. Guidance is disabled for learned evaluation; evaluation preserves training state and randomness. | Policy-order invariance, shared seeds, unchanged Q-table/RNG, and zero learned-policy guidance verified. CLI comparison completed. |
| 4 | Automated tests | Add simulation, learner, runner, and optional renderer regression suites plus a headless GitHub Actions workflow. | All 63 local tests pass. The CI workflow is configured; remote CI has not been run. |
| 5 | Environment architecture | Keep state and RNG inside each game. Separate geometry/simulation from the optional renderer; remove graphics import side effects. | Headless subprocess confirms Pygame is not imported. Interleaved independently seeded games produce identical traces. |
| 6 | State and action representation | Encode absolute normalized coordinates, row parity, health, survival, melee/ranged strengths, range, walls, and terrain. Mask edges, mountains, collisions, and inactive commands; account for sequential movement. | Geometry, vacated/contested tile masks, dead-unit waiting, archer range masks, masked exploration/greedy choice/bootstrap, and observation schema tests pass. |
| 7 | Training visibility | Export CSV history, JSON settings/evaluation, and SVG learning curves. Display turn/outcome/health/walls with pause, single-turn, speed, and quit controls. | CSV/JSON contents and four SVG panels verified. Playback events and manual archer control pass. Default and small-map board PNGs visually inspected. |
| 8 | Reproducible setup | Document Python 3.12; pin NumPy 2.4.6 and optional Pygame 2.6.1 separately. Ignore future generated artifacts; enable console output in the executable build specification. | Existing Python 3.12.4 environment runs tests and CLI commands; pip check reports no broken requirements. Existing executable has not been rebuilt. |
| 9 | Checkpoint reliability | Validate versioned models and progress; save atomically and periodically. Store schema, settings, episode history, seeds, and policy RNG. Select best models on a separate, persistent validation set. | Save/load RNG round trips, failed-write preservation, corrupt/nonfinite metadata rejection, exact split-vs-uninterrupted training, periodic/best saves, and validation-setting restoration pass. CLI resumed episode 11 through 13. |
| 10 | Gameplay depth | Configure map/city location, one to four warriors/archers, terrain, walls, city ranged attacks, unit strengths, difficulty, and curriculum. Forest adds defense/cost; mountains block movement; ranged fire cannot capture. | Connected generated starts, wall overflow, forest defense, city/archer range, melee capture, mixed-unit resume, difficulty presets, and curriculum targets verified. Mixed terrain/wall/archer training run completed. |

The richer state and configurable scenarios still make tabular learning expensive as map size and unit count grow. Use small scenarios and compare the learned policy with the separately reported baselines before drawing conclusions about learning quality. Models from the previous observation/action format require fresh training.

Validation commands:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B run_game.py --episodes 2 --max-steps 4 --eval-episodes 2 --fresh --no-save --metrics-dir runs/final-smoke
.\.venv\Scripts\python.exe -B run_game.py --episodes 10 --max-steps 30 --eval-episodes 3 --checkpoint-every 5 --validation-episodes 3 --save-best --width 6 --height 6 --unit-count 3 --unit-types warrior archer warrior --wall-hp 30 --forest-density 0.2 --mountain-density 0.1 --difficulty easy --curriculum-episodes 5 --fresh --model-path runs/verification/model.json --metrics-dir runs/verification
.\.venv\Scripts\python.exe -B run_game.py --episodes 3 --eval-episodes 2 --model-path runs/verification/model.json --metrics-dir runs/verification/resumed
```

The ten-episode experiment is a functionality check, not evidence that the learner is strong: pure learned evaluation timed out in all three games, while the heuristic won all three. Evaluation exposes that difference explicitly. Example metrics and board previews are retained under ignored `runs/verification/` and `runs/final-smoke/` directories.
