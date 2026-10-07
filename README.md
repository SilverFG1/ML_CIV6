# ML_CIV6

A small Civ-inspired hex-grid combat environment and tabular Q-learning experiment, originally inspired by [this Civilization VI machine-learning video](https://www.youtube.com/watch?v=kjA9LxLx9IQ). Units move, heal, fight, and attempt to capture a defended city. The simulation supports configurable terrain, city walls, ranged units, and difficulty presets. It implements a limited combat scenario rather than the full Civilization VI ruleset.

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

`--episodes` is the number of additional training episodes. An existing model loads automatically and restores its scenario, learning settings, progress, and random-generator states unless compatible options are explicitly supplied. `--fresh` starts a new model. `--no-save` disables checkpoint writes while retaining metrics output. The current model and observation formats are versioned; legacy models require `--fresh` because their states and actions are incompatible with the upgraded environment.

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
| `--difficulty easy\|normal\|hard` | Choose terrain and combat settings. |
| `--width`, `--height` | Set the map dimensions. |
| `--unit-count` | Choose one to four starting units. |
| `--city-position X Y` | Set the city's map coordinate. |
| `--unit-strengths N [N ...]`, `--city-strength` | Override unit and city combat strengths. |
| `--ranged-strength` | Configure the ranged attack strength. |
| `--wall-hp` | Set the city's initial wall health. |
| `--mountain-density`, `--forest-density` | Control terrain generation. |
| `--no-city-healing` | Disable city healing. |
| `--curriculum-episodes N` | Progress from easier scenarios to the selected difficulty over N training episodes. |

The observation includes normalized absolute city/unit coordinates, health, survival, combat strengths, walls, and terrain. Legal-action masks exclude impossible choices and restrict dead units to waiting. Joint actions still grow exponentially: three units produce 343 combinations before masking, and four produce 2,401. Larger scenarios therefore make this tabular learner substantially more expensive. Start small and use the reported baselines to judge whether training improves the learned policy.

## Metrics and playback

Each run writes to `--metrics-dir` (default `training_metrics`):

- `training.csv`: episode-level training metrics, including reward, win/termination outcome, length, Q-table size, and guidance usage.
- `metrics.json`: settings and evaluation summaries for comparing runs.
- `learning_curve.svg`: training curves that open in a browser without extra plotting packages.

Use separate output directories for experiments. Checkpoints save atomically; periodic saves preserve recoverable progress during long runs. `--checkpoint-every N` controls their interval.

Render evaluation after training or replay a saved policy:

```powershell
python run_game.py --episodes 0 --eval-episodes 3 --eval-policies learned --model-path runs/baseline/model.json --metrics-dir runs/playback --render --render-delay-ms 100
```

During playback, **Space** pauses/resumes, **Right Arrow** advances one turn while paused, **+ / -** change playback speed, and **Esc** exits. `--render-training` also displays training turns.

## Verification

```powershell
python -m unittest discover -s tests -v
```

The tests cover simulation rules, independent environments, terminal rewards, legal actions, Q-learning updates, checkpoint validation/resumption, and runner behavior. GitHub Actions runs the suite on Python 3.12 with headless dependencies and executes a short CLI training/evaluation run. [UPGRADE_STATUS.md](UPGRADE_STATUS.md) tracks the ten upgrade areas and their integration evidence.
