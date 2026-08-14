# ML_CIV6
Community machine learning algorithm development for the game Civilization 6 found in: https://www.youtube.com/watch?v=kjA9LxLx9IQ

Initial commit only includes the basic enviroment as well as an example of how to tie in a ML algorithm. 

-IO

## Q-learning AI

`run_game.py` now trains and evaluates a tabular Q-learning agent against the existing `Game(ml_ai=True)` environment.

Quick smoke test:

```powershell
.\.venv\Scripts\python.exe run_game.py --episodes 10 --max-steps 20 --eval-episodes 1 --no-save
```

Train and save a reusable model:

```powershell
.\.venv\Scripts\python.exe run_game.py --episodes 1000 --max-steps 80 --eval-episodes 5 --model-path q_learning_model.json
```

Render evaluation games after training:

```powershell
.\.venv\Scripts\python.exe run_game.py --episodes 1000 --render --render-delay-ms 100
```

The learned Q-table is saved as JSON and loaded automatically on future runs unless `--fresh` is supplied.
By default, training uses a distance-based guidance fallback for exploration and unseen states so the small tabular learner does not waste most episodes on random joint moves. Use `--no-guidance` to run the pure Q-table policy.
