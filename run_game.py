"""Reproducible training, checkpointing, and policy comparisons for ML_CIV6."""

import argparse
import csv
import html
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import game
from learning_ai import QLearningAgent


DEFAULT_MODEL_PATH = Path("q_learning_model.json")
DIFFICULTY_STRENGTHS = {"easy": 18, "normal": 28, "hard": 38, "curriculum": 38}
METRIC_FIELDS = ("episode", "seed", "reward", "outcome", "steps", "epsilon", "guidance_actions",
                 "random_actions", "policy_actions", "q_table_size", "city_strength")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000, help="additional training episodes")
    parser.add_argument("--max-steps", type=int, help="turn limit (default: 80, or checkpoint value)")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-seed", type=int, default=100007, help="first fixed policy comparison seed")
    parser.add_argument("--eval-policies", nargs="+", choices=("learned", "heuristic", "random"),
                        default=["learned", "heuristic", "random"])
    parser.add_argument("--eval-guidance", action="store_true", help="also measure the guided policy")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="save latest model every N absolute episodes; 0 disables periodic saves")
    parser.add_argument("--save-best", action="store_true", help="save best model using separate validation games")
    parser.add_argument("--validation-episodes", type=int, help="best model validation games (default: 20)")
    parser.add_argument("--validation-seed", type=int, help="first separate validation seed (default: 200007)")
    parser.add_argument("--metrics-dir", type=Path, default=Path("training_metrics"))
    parser.add_argument("--render", action="store_true", help="render evaluation games")
    parser.add_argument("--render-training", action="store_true")
    parser.add_argument("--render-delay-ms", type=int, default=0)
    parser.add_argument("--seed", type=int, help="training seed (default: 7, or checkpoint value)")
    for flag, kind in (("learning-rate", float), ("discount-factor", float), ("epsilon", float),
                       ("epsilon-decay", float), ("epsilon-min", float), ("state-precision", int),
                       ("initial-q", float)):
        parser.add_argument("--" + flag, type=kind)
    parser.add_argument("--guided-exploration", type=float,
                        help="guided fraction of epsilon exploration (default: 0.35)")
    parser.add_argument("--no-guidance", action="store_true", help="train without heuristic fallback")
    parser.add_argument("--log-every", type=int, default=50)
    for flag, kind in (("width", int), ("height", int), ("city-strength", float), ("wall-hp", float),
                       ("ranged-strength", float), ("mountain-density", float), ("forest-density", float)):
        parser.add_argument("--" + flag, type=kind)
    parser.add_argument("--unit-count", type=int, choices=range(1, 5))
    parser.add_argument("--city-position", type=int, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--no-city-healing", action="store_true")
    parser.add_argument("--unit-strengths", type=float, nargs="+")
    parser.add_argument("--unit-types", choices=("warrior", "archer"), nargs="+")
    parser.add_argument("--difficulty", choices=tuple(DIFFICULTY_STRENGTHS))
    parser.add_argument("--curriculum-episodes", type=int,
                        help="ramp city strength from 18 to selected difficulty over N absolute episodes")
    args = parser.parse_args(argv)
    for name in ("episodes", "eval_episodes", "checkpoint_every", "log_every", "render_delay_ms"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("max_steps", "state_precision", "width", "height", "validation_episodes"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.curriculum_episodes is not None and args.curriculum_episodes < 0:
        parser.error("--curriculum-episodes must be non-negative")
    for name in ("epsilon", "epsilon_decay", "epsilon_min", "guided_exploration", "discount_factor",
                 "mountain_density", "forest_density"):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must lie between 0 and 1")
    if args.learning_rate is not None and not 0 < args.learning_rate <= 1:
        parser.error("--learning-rate must lie above 0 and at most 1")
    return args


def config_from_args(args, saved=None):
    """Restore omitted scenario flags and reject accidental changes on resume."""
    saved = saved or {}
    values = asdict(game.GameConfig())
    values.update(saved)
    explicit = {}
    for name in ("width", "height", "unit_count", "city_position", "city_strength", "wall_hp",
                 "ranged_strength", "mountain_density", "forest_density", "unit_strengths", "unit_types", "difficulty"):
        value = getattr(args, name)
        if value is not None:
            explicit[name] = tuple(value) if name in ("city_position", "unit_strengths", "unit_types") else value
    if args.no_city_healing:
        explicit["city_heals"] = False
    if args.difficulty is not None and args.city_strength is None:
        explicit["city_strength"] = DIFFICULTY_STRENGTHS[args.difficulty]
    elif not saved and args.city_strength is None:
        values["city_strength"] = DIFFICULTY_STRENGTHS[values["difficulty"]]
    for name, value in explicit.items():
        old = values.get(name)
        if name in ("city_position", "unit_strengths", "unit_types") and old is not None:
            old = tuple(old)
        if saved and old != value:
            raise ValueError(f"--{name.replace('_', '-')} differs from the checkpoint; use --fresh for a new scenario")
        values[name] = value
    values["city_position"] = None if values["city_position"] is None else tuple(values["city_position"])
    values["unit_strengths"] = tuple(values["unit_strengths"])
    values["unit_types"] = tuple(values["unit_types"])
    return game.GameConfig(**values)


def _restore_setting(args, saved, name, default):
    previous = saved.get(name, default)
    value = getattr(args, name)
    if value is not None and name in saved and value != previous:
        raise ValueError(f"--{name.replace('_', '-')} differs from the checkpoint; use --fresh")
    setattr(args, name, previous if value is None else value)


def _validate_training_state(agent):
    """Check runner-owned progress before using it to resume an experiment."""
    state = agent.training_state
    if not state and agent.episodes_trained == 0:
        return
    required = ('config', 'seed', 'max_steps', 'curriculum_episodes', 'guided_exploration',
                'no_guidance', 'episodes_completed', 'checkpoint_episode', 'metrics')
    if any(name not in state for name in required):
        raise ValueError('Checkpoint is missing training progress; use --fresh')
    if not isinstance(state['config'], dict):
        raise ValueError('Checkpoint config must be an object')
    for name, minimum in [('seed', None), ('max_steps', 1), ('curriculum_episodes', 0),
                          ('episodes_completed', 0), ('checkpoint_episode', 0),
                          ('validation_seed', None), ('validation_episodes', 1)]:
        if name not in state:
            continue
        value = state[name]
        if isinstance(value, bool) or not isinstance(value, int) or (minimum is not None and value < minimum):
            raise ValueError(f'Checkpoint {name} is invalid')
    if state['episodes_completed'] != agent.episodes_trained or state['checkpoint_episode'] != agent.episodes_trained:
        raise ValueError('Checkpoint episode counters disagree')
    rate = state['guided_exploration']
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
        raise ValueError('Checkpoint guided exploration must lie between 0 and 1')
    if not isinstance(state['no_guidance'], bool):
        raise ValueError('Checkpoint no_guidance must be a boolean')
    rows = state['metrics']
    if not isinstance(rows, list) or len(rows) != agent.episodes_trained:
        raise ValueError('Checkpoint training history does not match episode count')
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != set(METRIC_FIELDS):
            raise ValueError('Checkpoint contains an invalid training history row')
        for name in ('reward', 'epsilon', 'city_strength'):
            if isinstance(row[name], bool) or not isinstance(row[name], (int, float)):
                raise ValueError(f'Checkpoint history {name} must be numeric')
        if row['episode'] != index + 1 or row['seed'] != state['seed'] + index:
            raise ValueError('Checkpoint training history has inconsistent episode seeds')
        for name in ('episode', 'seed', 'steps', 'guidance_actions', 'random_actions', 'policy_actions', 'q_table_size'):
            if isinstance(row[name], bool) or not isinstance(row[name], int):
                raise ValueError(f'Checkpoint history {name} must be an integer')
        if not 1 <= row['steps'] <= state['max_steps'] or any(row[name] < 0 for name in
                ('guidance_actions', 'random_actions', 'policy_actions', 'q_table_size')):
            raise ValueError('Checkpoint history contains invalid counts')
        if row['guidance_actions'] + row['random_actions'] + row['policy_actions'] != row['steps']:
            raise ValueError('Checkpoint history action counts disagree')
        if row['outcome'] not in ('win', 'loss', 'timeout', 'quit') or not 0 <= row['epsilon'] <= 1:
            raise ValueError('Checkpoint history contains invalid outcomes or exploration rates')
    best = state.get('best_validation_score')
    if best is not None and (not isinstance(best, list) or len(best) != 2
                             or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in best)
                             or not 0 <= best[0] <= 1):
        raise ValueError('Checkpoint best validation score is invalid')


def build_agent(args, rng):
    loaded = args.model_path.exists() and not args.fresh
    agent = QLearningAgent.load(args.model_path, rng=rng) if loaded else None
    if loaded and agent.observation_schema is not None:
        _validate_training_state(agent)
    saved = agent.training_state if loaded else {}
    _restore_setting(args, saved, "seed", 7)
    _restore_setting(args, saved, "max_steps", 80)
    difficulty = args.difficulty or saved.get('config', {}).get('difficulty', 'normal')
    _restore_setting(args, saved, "curriculum_episodes", 1000 if difficulty == 'curriculum' else 0)
    _restore_setting(args, saved, "guided_exploration", 0.35)
    _restore_setting(args, saved, "validation_episodes", 20)
    _restore_setting(args, saved, "validation_seed", 200007)
    if saved.get("no_guidance") and not args.no_guidance:
        args.no_guidance = True
    elif saved and args.no_guidance != saved.get("no_guidance", False):
        raise ValueError("--no-guidance differs from the checkpoint; use --fresh")
    try:
        config = config_from_args(args, saved.get("config"))
    except (TypeError, KeyError) as error:
        raise ValueError(f'Invalid checkpoint scenario: {error}') from error
    env = game.Game(config=config, seed=args.seed, ml_ai=True, render=False)
    try:
        schema, size, action_count = env.observation_schema, len(env.reset(seed=args.seed)), env.action_count
    finally:
        env.close()
    if loaded:
        if agent.action_count != action_count or agent.observation_schema != schema or agent.observation_size != size:
            raise ValueError("Checkpoint uses a different action/observation format; use --fresh to retrain")
        if args.state_precision is not None and args.state_precision != agent.state_precision:
            raise ValueError("State precision cannot change on resume; use --fresh")
        for name in ("learning_rate", "discount_factor", "epsilon", "epsilon_decay", "epsilon_min", "initial_q"):
            if getattr(args, name) is not None:
                setattr(agent, name, getattr(args, name))
    else:
        rng.seed(args.seed)
        defaults = dict(learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995,
                        epsilon_min=0.05, state_precision=100, initial_q=-100.0)
        options = {name: default if getattr(args, name) is None else getattr(args, name)
                   for name, default in defaults.items()}
        agent = QLearningAgent(action_count, rng=rng, observation_schema=schema, observation_size=size, **options)
    args.config = config
    return agent


def episode_config(args, absolute_episode):
    if not args.curriculum_episodes:
        return args.config
    progress = 1.0 if args.curriculum_episodes == 1 else min(1.0, absolute_episode / (args.curriculum_episodes - 1))
    start_strength = min(18.0, args.config.city_strength)
    return replace(args.config, city_strength=start_strength + (args.config.city_strength - start_strength) * progress)


@dataclass
class EpisodeResult:
    reward: float
    outcome: str
    steps: int
    guidance_actions: int = 0
    random_actions: int = 0
    policy_actions: int = 0

    @property
    def won(self):
        return self.outcome == "win"

    def __iter__(self):
        return iter((self.reward, self.won, self.steps))


def run_episode(env, agent, episode_number, max_steps, train, render_delay_ms=0,
                guided_exploration_rate=0.0, use_guidance=False, seed=None,
                policy="learned", rng=None):
    state = env.reset(seed=seed)
    total_reward = 0.0
    outcome = "timeout"
    counts = {"guided": 0, "random": 0, "policy": 0}
    steps_taken = 0
    policy_rng = rng if rng is not None else agent.rng
    for step in range(max_steps):
        valid = env.valid_actions()
        if policy == "heuristic":
            action, source = env.guided_action(policy_rng), "guided"
        elif policy == "random":
            action, source = policy_rng.choice(valid), "random"
        else:
            fallback = env.guided_action(policy_rng) if use_guidance else None
            action = agent.choose_action(state, explore=train, valid_actions=valid,
                                         fallback_action=fallback,
                                         guided_exploration_rate=guided_exploration_rate)
            source = agent.last_action_source
        next_state, reward, done = env.step(action)
        steps_taken = step + 1
        truncated = steps_taken == max_steps and not done
        if train:
            agent.learn(state, action, reward, next_state, done or truncated,
                        next_valid_actions=[] if done or truncated else env.valid_actions())
        counts[source] += 1
        total_reward += float(reward)
        state = next_state
        if done:
            outcome = env.last_info.get("outcome", "loss")
        if env.render and not env.render_frame(delay_ms=render_delay_ms):
            outcome = "quit"
            break
        if done:
            break
    if train:
        agent.finish_episode()
    return EpisodeResult(total_reward, outcome, steps_taken, counts["guided"], counts["random"], counts["policy"])


def compare_policies(agent, args, *, count=None, seed=None, config=None, policies=None, render=False, verbose=True):
    """Evaluate every policy on the same seeds without consuming training randomness."""
    count = args.eval_episodes if count is None else count
    seed = args.eval_seed if seed is None else seed
    config = args.config if config is None else config
    explicit_policies = policies is not None
    policies = list(args.eval_policies if policies is None else policies)
    if args.eval_guidance and not explicit_policies and "guided" not in policies:
        policies.append("guided")
    summaries = {}
    original_rng = agent.rng
    original_source = agent.last_action_source
    original_size = agent.observation_size
    try:
        for policy in policies:
            env = game.Game(config=config, seed=seed, ml_ai=True, render=render)
            rows = []
            try:
                for index in range(count):
                    local_rng = random.Random(seed + index)
                    agent.rng = local_rng
                    result = run_episode(env, agent, index, args.max_steps, train=False,
                                         seed=seed + index, rng=local_rng, policy=policy,
                                         use_guidance=policy == "guided", render_delay_ms=args.render_delay_ms)
                    rows.append({"seed": seed + index, **asdict(result)})
                    if verbose:
                        print(f"{policy} eval {index + 1}/{count} | reward {result.reward:.2f} | "
                              f"{result.outcome} | steps {result.steps}")
                    if result.outcome == "quit":
                        break
            finally:
                env.close()
            completed = len(rows)
            wins = sum(row["outcome"] == "win" for row in rows)
            summaries[policy] = {
                "episodes": completed, "wins": wins,
                "losses": sum(row["outcome"] == "loss" for row in rows),
                "timeouts": sum(row["outcome"] == "timeout" for row in rows),
                "quits": sum(row["outcome"] == "quit" for row in rows),
                "win_rate": wins / completed if completed else 0.0,
                "average_reward": sum(row["reward"] for row in rows) / completed if completed else 0.0,
                "average_steps": sum(row["steps"] for row in rows) / completed if completed else 0.0,
                "episodes_detail": rows,
            }
            if verbose:
                summary = summaries[policy]
                print(f"{policy}: wins {wins}/{completed}, losses {summary['losses']}, "
                      f"timeouts {summary['timeouts']}, average reward {summary['average_reward']:.2f}")
            if summaries[policy]["quits"]:
                break
    finally:
        agent.rng = original_rng
        agent.last_action_source = original_source
        agent.observation_size = original_size
    return summaries


def _checkpoint_state(agent, args, rows, best_score=None):
    state = dict(agent.training_state)
    state.update(config=asdict(args.config), seed=args.seed, max_steps=args.max_steps,
                 curriculum_episodes=args.curriculum_episodes, guided_exploration=args.guided_exploration,
                 validation_episodes=args.validation_episodes, validation_seed=args.validation_seed,
                 no_guidance=args.no_guidance, episodes_completed=agent.episodes_trained,
                 checkpoint_episode=agent.episodes_trained, metrics=rows)
    if best_score is not None:
        state["best_validation_score"] = list(best_score)
    agent.training_state = state


def _best_path(path):
    return path.with_name(path.stem + ".best" + path.suffix)


def _save_checkpoint(agent, args, rows):
    best_score = agent.training_state.get("best_validation_score")
    if args.save_best:
        summary = compare_policies(agent, args, count=args.validation_episodes, seed=args.validation_seed,
                                   config=args.config, policies=["learned"], verbose=False)["learned"]
        score = (summary["win_rate"], summary["average_reward"])
        if best_score is None or score > tuple(best_score):
            best_score = score
            _checkpoint_state(agent, args, rows, best_score)
            agent.save(_best_path(args.model_path))
    _checkpoint_state(agent, args, rows, best_score)
    agent.save(args.model_path)


def train_agent(agent, args):
    rows = list(agent.training_state.get("metrics", []))
    rewards = []
    wins = 0
    start_episode = agent.episodes_trained
    for offset in range(args.episodes):
        episode = start_episode + offset
        config = episode_config(args, episode)
        env = game.Game(config=config, seed=args.seed + episode, ml_ai=True, render=args.render_training)
        try:
            result = run_episode(env, agent, episode, args.max_steps, train=True,
                                 seed=args.seed + episode, render_delay_ms=args.render_delay_ms,
                                 guided_exploration_rate=args.guided_exploration,
                                 use_guidance=not args.no_guidance)
        finally:
            env.close()
        rewards.append(result.reward)
        wins += int(result.won)
        rows.append(dict(episode=episode + 1, seed=args.seed + episode, reward=result.reward,
                         outcome=result.outcome, steps=result.steps, epsilon=agent.epsilon,
                         guidance_actions=result.guidance_actions, random_actions=result.random_actions,
                         policy_actions=result.policy_actions, q_table_size=len(agent.q_table),
                         city_strength=config.city_strength))
        _checkpoint_state(agent, args, rows)
        if args.log_every and (offset == 0 or (episode + 1) % args.log_every == 0):
            window = rewards[-args.log_every:]
            print(f"episode {episode + 1} | avg reward {sum(window) / len(window):.2f} | "
                  f"wins {wins}/{len(rewards)} | epsilon {agent.epsilon:.3f} | "
                  f"guided {result.guidance_actions}/{result.steps} | Q states {len(agent.q_table)}")
        if args.checkpoint_every and (episode + 1) % args.checkpoint_every == 0:
            if not args.no_save:
                _save_checkpoint(agent, args, rows)
            write_metrics(args.metrics_dir, rows, config=args.config)
        if result.outcome == "quit":
            break
    return rewards, wins


def evaluate_agent(agent, args):
    return compare_policies(agent, args, render=args.render)


def _curve_svg(rows):
    width, height, padding = 960, 640, 65
    panels = (("reward", "Episode reward", "#2864dc"),
              ("win_rate", "Rolling win rate (50 episodes)", "#199663"),
              ("steps", "Episode length", "#b353ce"),
              ("q_table_size", "Q-table states", "#ca7329"))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
             'aria-label="Training learning curves"><rect width="100%" height="100%" fill="white"/>',
             '<style>text{font:14px sans-serif;fill:#222}.axis{stroke:#bbb;stroke-width:1}</style>']
    for index, (field, label, color) in enumerate(panels):
        origin_x = (index % 2) * width / 2 + padding
        origin_y = (index // 2) * height / 2 + padding
        plot_w, plot_h = width / 2 - padding - 30, height / 2 - padding - 55
        parts.append(f'<text x="{origin_x}" y="{origin_y - 25}">{html.escape(label)}</text>')
        parts.append(f'<path class="axis" fill="none" d="M{origin_x},{origin_y}v{plot_h}h{plot_w}"/>')
        if not rows:
            parts.append(f'<text x="{origin_x + 15}" y="{origin_y + 35}">No training episodes</text>')
            continue
        values = []
        for position, row in enumerate(rows):
            if field == "win_rate":
                window = rows[max(0, position - 49):position + 1]
                value = sum(item["outcome"] == "win" for item in window) / len(window)
            else:
                value = float(row[field])
            values.append(value)
        low, high = (0.0, 1.0) if field == "win_rate" else (min(values), max(values))
        if low == high:
            low, high = low - 0.5, high + 0.5
        points = " ".join(f"{origin_x + plot_w * position / max(1, len(values) - 1):.2f},"
                          f"{origin_y + plot_h * (1 - (value - low) / (high - low)):.2f}"
                          for position, value in enumerate(values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for value, y in ((high, origin_y), (low, origin_y + plot_h)):
            parts.append(f'<text x="{origin_x - 8}" y="{y + 4}" text-anchor="end">{value:.3g}</text>')
        parts.append(f'<text x="{origin_x}" y="{origin_y + plot_h + 24}">{rows[0]["episode"]}</text>')
        parts.append(f'<text x="{origin_x + plot_w}" y="{origin_y + plot_h + 24}" '
                     f'text-anchor="end">{rows[-1]["episode"]}</text>')
        parts.append(f'<text x="{origin_x + plot_w / 2}" y="{origin_y + plot_h + 42}" '
                     'text-anchor="middle">Episode</text>')
    return "\n".join(parts + ["</svg>"])


def write_metrics(directory, rows, evaluations=None, config=None, settings=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "training.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    payload = {"format_version": 1, "config": asdict(config) if config is not None else None,
               "settings": settings or {},
               "training": rows, "evaluation": evaluations or {}}
    (directory / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    (directory / "learning_curve.svg").write_text(_curve_svg(rows), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    try:
        agent = build_agent(args, random.Random())
        if args.episodes:
            train_agent(agent, args)
        rows = agent.training_state.get("metrics", [])
        if not args.no_save:
            _save_checkpoint(agent, args, rows)
            print(f"saved model to {args.model_path}")
        evaluations = evaluate_agent(agent, args) if args.eval_episodes else {}
        settings = {name: getattr(agent, name) for name in
                    ('learning_rate', 'discount_factor', 'epsilon', 'epsilon_decay', 'epsilon_min',
                     'state_precision', 'initial_q', 'episodes_trained', 'observation_schema')}
        settings.update({name: getattr(args, name) for name in
                         ('seed', 'max_steps', 'eval_seed', 'eval_episodes', 'eval_policies',
                          'eval_guidance', 'validation_seed', 'validation_episodes',
                          'curriculum_episodes', 'guided_exploration', 'no_guidance')})
        write_metrics(args.metrics_dir, rows, evaluations=evaluations, config=args.config, settings=settings)
        print(f"metrics written to {args.metrics_dir}")
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
