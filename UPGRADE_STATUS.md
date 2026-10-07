# Upgrade status

This checklist tracks the ten requested upgrade areas. The validation column is completed after integration checks. This project remains a small Civ-inspired combat simulation with a tabular learner; it does not implement the full Civilization VI ruleset.

| # | Area | Implementation | Validation |
| --- | --- | --- | --- |
| 1 | Game-ending logic | End on capture, destruction of all units, or the turn limit; suppress actions after termination. | Pending integration checks. |
| 2 | Reward calculation | Record explicit turn events for damage, healing, death, and capture; exclude dead units from distance penalties. | Pending integration checks. |
| 3 | AI evaluation | Evaluate learned, heuristic, and random policies separately on matched scenario seeds using independent random streams. | Pending integration checks. |
| 4 | Automated tests | Cover movement, combat, healing, termination, learning updates, checkpoint loading, and runner behavior; add headless CI. | Pending integration checks. |
| 5 | Environment architecture | Own state per game; import and initialize Pygame only when rendering. | Pending integration checks. |
| 6 | State and action representation | Encode normalized positions, health, survival, strengths, walls, and terrain; mask impossible moves and make dead units wait. | Pending integration checks. |
| 7 | Training visibility | Export episode metrics, evaluation summaries, and SVG learning curves; add pause, single-step, and speed controls. | Pending integration checks. |
| 8 | Reproducible setup | Document Python 3.12; pin simulation dependencies separately from the optional renderer. | Pending integration checks. |
| 9 | Checkpoint reliability | Validate versioned models; save atomically and periodically; retain training progress and random-generator state for resuming. | Pending integration checks. |
| 10 | Gameplay depth | Configure map size, starting units, terrain, walls, ranged units, strengths, difficulty, and curriculum. | Pending integration checks. |

The richer state and configurable scenarios still make tabular learning expensive as map size and unit count grow. Use small scenarios and compare the learned policy with the separately reported baselines before drawing conclusions about learning quality. Models from the previous observation/action format require fresh training.
