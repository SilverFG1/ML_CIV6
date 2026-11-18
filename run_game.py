"""Reproducible training, checkpointing, and policy comparisons for ML_CIV6."""

import argparse
import collections
import csv
import html
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import empire
import game
from learning_ai import AGENT_TYPES, EmpireQAgent, QLearningAgent, TacticalQAgent, load_agent


DEFAULT_MODEL_PATH = Path("q_learning_model.json")
TUPLE_SETTINGS = ("city_position", "unit_strengths", "unit_types", "land_types", "buildings", "build_queue")
# Defaults for a new model of each learner; checkpoints restore their own values.
LEARNING_DEFAULTS = {
    "tactical": dict(learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995,
                     epsilon_min=0.05, initial_q=0.0, shaping=0.5, learning_rate_decay=0.998,
                     learning_rate_min=0.01),
    "joint": dict(learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995,
                  epsilon_min=0.05, state_precision=100, initial_q=-100.0),
    "empire": dict(learning_rate=0.1, discount_factor=0.97, epsilon=1.0, epsilon_decay=0.995,
                   epsilon_min=0.05, initial_q=0.0, learning_rate_decay=0.998, learning_rate_min=0.01,
                   untried_penalty=0.5, demonstration_margin=1.0, margin_on="all"),
}
MODES = ("siege", "empire")
# Scenario flags that only one mode understands.
SIEGE_SETTINGS = ("unit_count", "unit_types", "unit_strengths", "city_strength", "wall_hp", "walls", "buildings",
                  "ranged_strength", "city_population", "build_queue", "city_position", "city_count",
                  "raid_interval", "capital_strength", "raider_strength")
EMPIRE_SETTINGS = ("rivals", "map_type", "turn_limit", "resource_density", "rival_bonus", "speed")
# The empire curriculum ramps the rival's bonus up from this level.
EMPIRE_CURRICULUM_START = 0.5
DIFFICULTY_STRENGTHS = {"easy": 18, "normal": 28, "hard": 38, "curriculum": 38}
# Share of training episodes that the heuristic plays from start to finish while
# the learner updates from every move, by learner. The joint learner keeps its
# original training. Demonstrations pause while the last DEMONSTRATION_WINDOW of
# them scored less on average than the learner's own last DEMONSTRATION_WINDOW
# episodes, so a heuristic that plays worse than the learner stops teaching it.
# The comparison needs DEMONSTRATION_EVIDENCE of each, so one unlucky early
# demonstration cannot stop them.
DEMONSTRATION_DEFAULTS = {"tactical": 0.3, "joint": 0.0, "empire": 0.3}
DEMONSTRATION_WINDOW = 100
DEMONSTRATION_EVIDENCE = 20
METRIC_FIELDS = ("episode", "seed", "reward", "outcome", "steps", "epsilon", "guidance_actions",
                 "random_actions", "policy_actions", "q_table_size", "city_strength", "demonstration")


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
    parser.add_argument("--mode", choices=MODES,
                        help="siege: a squad besieges a city (default); empire: found cities, grow, research and "
                             "fight a rival civilization (a checkpoint keeps its own)")
    parser.add_argument("--agent", choices=tuple(AGENT_TYPES),
                        help="learner for a new model (default: tactical for sieges, empire for empires; "
                             "a checkpoint keeps its own)")
    for flag, kind in (("learning-rate", float), ("discount-factor", float), ("epsilon", float),
                       ("epsilon-decay", float), ("epsilon-min", float), ("state-precision", int),
                       ("initial-q", float)):
        parser.add_argument("--" + flag, type=kind)
    parser.add_argument("--shaping", type=float,
                        help="tactical learner health-potential shaping weight (default: 0.5)")
    parser.add_argument("--learning-rate-decay", type=float,
                        help="tactical learner per-episode learning-rate decay once any curriculum "
                             "is complete (default: 0.998)")
    parser.add_argument("--learning-rate-min", type=float,
                        help="tactical learner learning-rate floor (default: 0.01)")
    parser.add_argument("--untried-penalty", type=float,
                        help="empire learner: how far below a situation's value untried options start (default: 0.5)")
    parser.add_argument("--demonstration-margin", type=float,
                        help="empire learner: margin by which demonstrated options must lead the others (default: 1.0)")
    parser.add_argument("--margin-on", choices=EmpireQAgent.MARGIN_SCOPES,
                        help="empire learner: apply the margin to demonstrations only, or also to the heuristic's "
                             "choices in the learner's own episodes (default: all)")
    parser.add_argument("--guided-exploration", type=float,
                        help="guided fraction of epsilon exploration (default: 0.35)")
    parser.add_argument("--demonstrations", type=float,
                        help="share of training episodes that the heuristic plays from start to finish while "
                             "the learner learns from its moves, paused while they score below the learner's "
                             "own recent episodes (default: 0.3 for the tactical learner, 0 for the joint learner)")
    parser.add_argument("--no-guidance", action="store_true",
                        help="train without heuristic fallback or demonstrations")
    parser.add_argument("--log-every", type=int, default=50)
    for flag, kind in (("width", int), ("height", int), ("city-strength", float), ("wall-hp", float),
                       ("ranged-strength", float), ("mountain-density", float), ("forest-density", float),
                       ("hills-density", float), ("water-density", float)):
        parser.add_argument("--" + flag, type=kind)
    parser.add_argument("--unit-count", type=int, choices=range(1, 5))
    parser.add_argument("--land-types", nargs="+", choices=game.LAND_TYPES,
                        help="Civ VI land types to fill the map with, in regions (default: grassland)")
    parser.add_argument("--buildings", nargs="+", choices=game.BUILDING_TYPES,
                        help="city buildings: monument (+2 culture), granary (+1 food, +2 housing)")
    parser.add_argument("--economy", action="store_true",
                        help="run the city's economy: growth, production of --build-queue, and border growth")
    parser.add_argument("--city-population", type=int, choices=range(1, game.MAX_CITY_POPULATION + 1), metavar="N",
                        help=f"starting city population with --economy (1-{game.MAX_CITY_POPULATION}, default: 1)")
    parser.add_argument("--build-queue", nargs="+", choices=game.BUILD_ITEMS,
                        help="what the city builds, in order, with --economy (default: "
                             + " ".join(game.DEFAULT_BUILD_QUEUE) + ")")
    parser.add_argument("--walls", choices=game.WALL_TYPES,
                        help="city walls; ancient gives 100 wall HP and a 15-strength ranged strike")
    parser.add_argument("--river-count", type=int, choices=range(0, game.MAX_RIVERS + 1), metavar="N",
                        help=f"rivers to generate along hex sides (0-{game.MAX_RIVERS}, default: 0)")
    parser.add_argument("--city-count", type=int, choices=range(1, game.MAX_CITIES + 1), metavar="N",
                        help=f"enemy capitals (1-{game.MAX_CITIES}); capture them all for a domination victory "
                             "(default: 1)")
    parser.add_argument("--player-capital", action="store_true",
                        help="give the squad its own capital to defend against raiders; losing it loses the game")
    parser.add_argument("--raid-interval", type=int,
                        help="turns between raids on the player's capital; 0 disables raids (default: 6)")
    parser.add_argument("--capital-strength", type=float, help="player's capital combat strength (default: 28)")
    parser.add_argument("--raider-strength", type=float, help="raider combat strength (default: 20)")
    parser.add_argument("--city-position", type=int, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--no-city-healing", action="store_true")
    parser.add_argument("--unit-strengths", type=float, nargs="+")
    parser.add_argument("--unit-types", choices=game.UNIT_TYPES, nargs="+")
    parser.add_argument("--difficulty", choices=tuple(DIFFICULTY_STRENGTHS))
    parser.add_argument("--curriculum-episodes", type=int,
                        help="ramp city strength from 18 (siege) or the rival's bonus from "
                             f"{EMPIRE_CURRICULUM_START} (empire) to the selected difficulty over N absolute episodes")
    empire_group = parser.add_argument_group("empire mode")
    empire_group.add_argument("--rivals", type=int, choices=(0, 1), help="rival civilizations (default: 1)")
    empire_group.add_argument("--map-type", choices=empire.MAP_TYPES,
                              help="continents: one landmass; islands: one island each (default: continents)")
    empire_group.add_argument("--turn-limit", type=int,
                              help="turns before the higher score wins (10-500, default: 100)")
    empire_group.add_argument("--speed", choices=tuple(empire.SPEEDS),
                              help="Civ VI game speed scaling production, research, growth and border costs "
                                   "(default: online, half of standard)")
    empire_group.add_argument("--resource-density", type=float,
                              help="share of tiles with a resource (0-0.5, default: 0.15)")
    empire_group.add_argument("--rival-bonus", type=float,
                              help="rival's production, science and gold multiplier (default: set by --difficulty)")
    args = parser.parse_args(argv)
    for name in ("episodes", "eval_episodes", "checkpoint_every", "log_every", "render_delay_ms"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("max_steps", "state_precision", "width", "height", "validation_episodes", "turn_limit"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.curriculum_episodes is not None and args.curriculum_episodes < 0:
        parser.error("--curriculum-episodes must be non-negative")
    for name in ("epsilon", "epsilon_decay", "epsilon_min", "guided_exploration", "discount_factor",
                 "mountain_density", "forest_density", "hills_density", "water_density",
                 "learning_rate_decay", "learning_rate_min", "demonstrations", "resource_density"):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 1:
            parser.error(f"--{name.replace('_', '-')} must lie between 0 and 1")
    if args.learning_rate is not None and not 0 < args.learning_rate <= 1:
        parser.error("--learning-rate must lie above 0 and at most 1")
    if args.shaping is not None and not 0 <= args.shaping <= 10:
        parser.error("--shaping must lie between 0 and 10")
    for name in ("untried_penalty", "demonstration_margin"):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 100:
            parser.error(f"--{name.replace('_', '-')} must lie between 0 and 100")
    return args


def config_from_args(args, saved=None):
    """Restore omitted scenario flags and reject accidental changes on resume."""
    saved = saved or {}
    values = asdict(game.GameConfig())
    values.update(saved)
    explicit = {}
    for name in ("width", "height", "unit_count", "city_position", "city_strength", "wall_hp", "walls",
                 "buildings", "ranged_strength", "mountain_density", "forest_density", "land_types", "hills_density",
                 "water_density", "river_count", "unit_strengths", "unit_types", "difficulty",
                 "city_population", "build_queue", "city_count", "raid_interval", "capital_strength",
                 "raider_strength"):
        value = getattr(args, name)
        if value is not None:
            explicit[name] = tuple(value) if name in TUPLE_SETTINGS else value
    if args.no_city_healing:
        explicit["city_heals"] = False
    if args.economy:
        explicit["economy"] = True
    if args.player_capital:
        explicit["player_capital"] = True
    if args.difficulty is not None and args.city_strength is None:
        explicit["city_strength"] = DIFFICULTY_STRENGTHS[args.difficulty]
    elif not saved and args.city_strength is None:
        values["city_strength"] = DIFFICULTY_STRENGTHS[values["difficulty"]]
    for name, value in explicit.items():
        old = values.get(name)
        if name in TUPLE_SETTINGS and old is not None:
            old = tuple(old)
        if saved and old != value:
            raise ValueError(f"--{name.replace('_', '-')} differs from the checkpoint; use --fresh for a new scenario")
        values[name] = value
    values["city_position"] = None if values["city_position"] is None else tuple(values["city_position"])
    values["unit_strengths"] = tuple(values["unit_strengths"])
    values["unit_types"] = tuple(values["unit_types"])
    values["land_types"] = tuple(values["land_types"])
    values["buildings"] = tuple(values["buildings"])
    values["build_queue"] = tuple(values["build_queue"])
    return game.GameConfig(**values)


def empire_config_from_args(args, saved=None):
    """The empire counterpart of config_from_args."""
    saved = saved or {}
    values = asdict(empire.EmpireConfig())
    values.update(saved)
    explicit = {}
    for name in ("width", "height", "rivals", "map_type", "turn_limit", "land_types", "hills_density",
                 "forest_density", "mountain_density", "water_density", "river_count", "resource_density",
                 "rival_bonus", "speed", "difficulty"):
        value = getattr(args, name)
        if value is not None:
            explicit[name] = tuple(value) if name == "land_types" else value
    if args.difficulty is not None and args.rival_bonus is None:
        explicit["rival_bonus"] = empire.DIFFICULTY_BONUS[args.difficulty]
    for name, value in explicit.items():
        old = values.get(name)
        if name == "land_types" and old is not None:
            old = tuple(old)
        if saved and old != value:
            raise ValueError(f"--{name.replace('_', '-')} differs from the checkpoint; use --fresh for a new scenario")
        values[name] = value
    values["land_types"] = tuple(values["land_types"])
    return empire.EmpireConfig(**values)


def make_env(config, seed=None, render=False, observation="tactical"):
    """The siege Game or the EmpireGame, chosen by the scenario's configuration."""
    if isinstance(config, empire.EmpireConfig):
        return empire.EmpireGame(config=config, seed=seed, render=render)
    return game.Game(config=config, seed=seed, ml_ai=True, render=render, observation=observation)


def difficulty_level(config):
    """What the training log records as an episode's difficulty: the city's strength, or the rival's bonus."""
    return config.rival_bonus if isinstance(config, empire.EmpireConfig) else config.city_strength


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
    if state.get('mode', 'siege') not in MODES:
        raise ValueError('Checkpoint mode is invalid')
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
    for name in ('guided_exploration', 'demonstrations'):
        # Checkpoints from before demonstrations existed have no share.
        rate = state.get(name, 0)
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 <= rate <= 1:
            raise ValueError(f'Checkpoint {name.replace("_", " ")} must lie between 0 and 1')
    if not isinstance(state['no_guidance'], bool):
        raise ValueError('Checkpoint no_guidance must be a boolean')
    rows = state['metrics']
    if not isinstance(rows, list) or len(rows) != agent.episodes_trained:
        raise ValueError('Checkpoint training history does not match episode count')
    for index, row in enumerate(rows):
        # Rows recorded before demonstrations existed lack that field.
        if not isinstance(row, dict) or set(row) not in (set(METRIC_FIELDS), set(METRIC_FIELDS) - {'demonstration'}):
            raise ValueError('Checkpoint contains an invalid training history row')
        demonstration = row.get('demonstration', False)
        if not isinstance(demonstration, bool):
            raise ValueError('Checkpoint history demonstration must be a boolean')
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
        if demonstration and row['guidance_actions'] != row['steps']:
            raise ValueError('Checkpoint history demonstrations must be played entirely by the heuristic')
        if row['outcome'] not in ('win', 'loss', 'timeout', 'quit') or not 0 <= row['epsilon'] <= 1:
            raise ValueError('Checkpoint history contains invalid outcomes or exploration rates')
    best = state.get('best_validation_score')
    if best is not None and (not isinstance(best, list) or len(best) != 2
                             or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in best)
                             or not 0 <= best[0] <= 1):
        raise ValueError('Checkpoint best validation score is invalid')


def build_agent(args, rng):
    """Load or create the learner and settle the run's scenario and settings.

    A checkpoint keeps its mode and learner; a new model uses --mode (default
    siege) and --agent (default tactical for sieges, empire for empires). Sets
    args.mode, args.config, args.agent, and args.observation for later
    environments.
    """
    loaded = args.model_path.exists() and not args.fresh
    agent = load_agent(args.model_path, rng=rng) if loaded else None
    if loaded and agent.observation_schema is not None:
        _validate_training_state(agent)
        # Episodes recorded before demonstrations existed were all the learner's own.
        for row in agent.training_state.get("metrics", []):
            row.setdefault("demonstration", False)
    saved = agent.training_state if loaded else {}
    if loaded:
        mode = saved.get("mode", "empire" if agent.agent_kind == "empire" else "siege")
    else:
        mode = args.mode or ("empire" if args.agent == "empire" else "siege")
    if args.mode is not None and args.mode != mode:
        raise ValueError(f"--mode {args.mode} differs from the checkpoint's {mode} mode; use --fresh")
    args.mode = mode
    kind = agent.agent_kind if loaded else args.agent or ("empire" if mode == "empire" else "tactical")
    if args.agent is not None and args.agent != kind:
        raise ValueError(f"--agent {args.agent} differs from the checkpoint's {kind} learner; use --fresh")
    if (mode == "empire") != (kind == "empire"):
        raise ValueError("the empire learner plays only --mode empire, and --mode empire needs the empire learner")
    wrong = [name for name in (SIEGE_SETTINGS if mode == "empire" else EMPIRE_SETTINGS)
             if getattr(args, name) is not None]
    if mode == "empire":
        wrong += [flag for flag in ("economy", "no_city_healing", "player_capital") if getattr(args, flag)]
    if wrong:
        raise ValueError(f"--{wrong[0].replace('_', '-')} does not apply to --mode {mode}")
    if kind != "joint" and args.state_precision is not None:
        raise ValueError("--state-precision applies only to the joint learner; use --fresh --agent joint")
    if kind != "tactical" and args.shaping is not None:
        raise ValueError("--shaping applies only to the tactical learner")
    empire_only = [flag for flag in ("untried_penalty", "demonstration_margin", "margin_on")
                   if getattr(args, flag) is not None]
    if kind != "empire" and empire_only:
        raise ValueError(f"--{empire_only[0].replace('_', '-')} applies only to the empire learner")
    annealing = [flag for flag in ("learning_rate_decay", "learning_rate_min") if getattr(args, flag) is not None]
    if kind == "joint" and annealing:
        raise ValueError(f"--{annealing[0].replace('_', '-')} applies only to the tactical and empire learners")
    args.agent = kind
    args.observation = AGENT_TYPES[kind].observation_mode
    try:
        config = (empire_config_from_args(args, saved.get("config")) if mode == "empire"
                  else config_from_args(args, saved.get("config")))
    except (TypeError, KeyError) as error:
        raise ValueError(f'Invalid checkpoint scenario: {error}') from error
    _restore_setting(args, saved, "seed", 7)
    _restore_setting(args, saved, "max_steps", config.turn_limit if mode == "empire" else 80)
    difficulty = args.difficulty or saved.get('config', {}).get('difficulty', 'normal')
    _restore_setting(args, saved, "curriculum_episodes", 1000 if difficulty == 'curriculum' else 0)
    _restore_setting(args, saved, "guided_exploration", 0.35)
    _restore_setting(args, saved, "validation_episodes", 20)
    _restore_setting(args, saved, "validation_seed", 200007)
    if saved.get("no_guidance") and not args.no_guidance:
        args.no_guidance = True
    elif saved and args.no_guidance != saved.get("no_guidance", False):
        raise ValueError("--no-guidance differs from the checkpoint; use --fresh")
    # A checkpoint saved before demonstrations existed continues without them.
    _restore_setting(args, saved, "demonstrations",
                     0.0 if saved or args.no_guidance else DEMONSTRATION_DEFAULTS[kind])
    if args.no_guidance and args.demonstrations:
        raise ValueError("--demonstrations are played by the heuristic; they cannot be combined with --no-guidance")
    env = make_env(config, seed=args.seed, observation=args.observation)
    try:
        observation = env.reset(seed=args.seed)
        schema = env.observation_schema
        if mode == "empire":
            size = action_count = command_counts = None
        else:
            size, action_count = len(observation), env.action_count
            command_counts = [len(directions) for directions in env.unit_directions]
    finally:
        env.close()
    if loaded:
        if agent.action_count != action_count or agent.observation_schema != schema or agent.observation_size != size:
            raise ValueError("Checkpoint uses a different action/observation format; use --fresh to retrain")
        if args.state_precision is not None and args.state_precision != agent.state_precision:
            raise ValueError("State precision cannot change on resume; use --fresh")
        adjustable = ("learning_rate", "discount_factor", "epsilon", "epsilon_decay", "epsilon_min", "initial_q")
        extra = {"tactical": ("shaping", "learning_rate_decay", "learning_rate_min"),
                 "empire": ("learning_rate_decay", "learning_rate_min", "untried_penalty",
                            "demonstration_margin", "margin_on")}.get(kind, ())
        for name in adjustable + extra:
            if getattr(args, name) is not None:
                setattr(agent, name, getattr(args, name))
    else:
        rng.seed(args.seed)
        options = {name: default if getattr(args, name) is None else getattr(args, name)
                   for name, default in LEARNING_DEFAULTS[kind].items()}
        if kind == "tactical":
            agent = TacticalQAgent(command_counts, rng=rng, observation_schema=schema, **options)
        elif kind == "empire":
            agent = EmpireQAgent(rng=rng, observation_schema=schema, **options)
        else:
            agent = QLearningAgent(action_count, rng=rng, observation_schema=schema, observation_size=size, **options)
    args.config = config
    return agent


def curriculum_complete(args, absolute_episode):
    """Whether this episode already uses the final scenario, so learning can settle."""
    return not args.curriculum_episodes or absolute_episode >= args.curriculum_episodes - 1


def episode_config(args, absolute_episode):
    if not args.curriculum_episodes:
        return args.config
    progress = 1.0 if args.curriculum_episodes == 1 else min(1.0, absolute_episode / (args.curriculum_episodes - 1))
    if isinstance(args.config, empire.EmpireConfig):
        start = min(EMPIRE_CURRICULUM_START, args.config.rival_bonus)
        return replace(args.config, rival_bonus=start + (args.config.rival_bonus - start) * progress)
    start_strength = min(18.0, args.config.city_strength)
    return replace(args.config, city_strength=start_strength + (args.config.city_strength - start_strength) * progress)


def is_demonstration(args, absolute_episode):
    """Whether the heuristic plays this training episode while the learner learns from it.

    Q-learning is off-policy, so complete demonstrations show the learner
    coordinated plays that its own exploration rarely finds. The share does
    not decay: once demonstrations stop, the learner's own exploratory play
    drags the values they taught back down. The draw uses a random stream
    derived from the episode's seed, so it consumes none of the agent's or
    environment's randomness, and a resumed run makes the same choices
    without saving anything more. train_agent also pauses demonstrations
    while demonstrations_ahead() is false.
    """
    return (args.demonstrations > 0
            and random.Random(f"demonstration:{args.seed + absolute_episode}").random() < args.demonstrations)


def demonstrations_ahead(recent):
    """Whether recent demonstrations scored more on average than the learner's own recent episodes.

    recent maps True to the rewards of recent demonstrations and False to those
    of the learner's own episodes. Until there are DEMONSTRATION_EVIDENCE of
    each to compare, demonstrations go ahead.
    """
    shown, own = recent[True], recent[False]
    if min(len(shown), len(own)) < DEMONSTRATION_EVIDENCE:
        return True
    return sum(shown) / len(shown) > sum(own) / len(own)


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
                policy="learned", rng=None, anneal=True):
    state = env.reset(seed=seed)
    total_reward = 0.0
    outcome = "timeout"
    counts = {"guided": 0, "random": 0, "policy": 0}
    steps_taken = 0
    policy_rng = rng if rng is not None else agent.rng
    for step in range(max_steps):
        valid = env.valid_actions()
        fallback = None
        if policy == "heuristic":
            action, source = env.guided_action(policy_rng), "guided"
        elif policy == "random":
            # The empire game picks one random legal option per decision.
            action = env.random_action(policy_rng) if hasattr(env, "random_action") else policy_rng.choice(valid)
            source = "random"
        else:
            fallback = env.guided_action(policy_rng) if use_guidance else None
            action = agent.choose_action(state, explore=train, valid_actions=valid,
                                         fallback_action=fallback,
                                         guided_exploration_rate=guided_exploration_rate)
            source = agent.last_action_source
        next_state, reward, done = env.step(action)
        steps_taken = step + 1
        if train:
            # A turn limit cuts the episode short without ending the game, and the
            # observation does not show the turn, so the cut-off state bootstraps.
            # Treating it as terminal would make stalling until the limit look free.
            # A learner taught by the heuristic hears which moves are demonstrations and,
            # in its own episodes, what the heuristic would have chosen.
            extra = ({"demonstration": policy == "heuristic", "teacher": fallback}
                     if getattr(agent, "learns_from_teacher", False) else {})
            agent.learn(state, action, reward, next_state, done,
                        next_valid_actions=[] if done else env.valid_actions(), **extra)
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
        agent.finish_episode(anneal=anneal)
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
            env = make_env(config, seed=seed, render=render, observation=agent.observation_mode)
            rows = []
            try:
                for index in range(count):
                    local_rng = random.Random(seed + index)
                    agent.rng = local_rng
                    result = run_episode(env, agent, index, args.max_steps, train=False,
                                         seed=seed + index, rng=local_rng, policy=policy,
                                         use_guidance=policy == "guided", render_delay_ms=args.render_delay_ms)
                    rows.append({"seed": seed + index, **asdict(result),
                                 **(env.summary() if hasattr(env, "summary") else {})})
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
            if hasattr(env, "summary") and completed:
                # Games also report the averages of their end-of-game summaries: the empire's
                # scores, cities captured in a domination siege, and how often the capital fell.
                for name in env.summary():
                    summaries[policy][f"average_{name}"] = sum(row[name] for row in rows) / completed
            if getattr(config, "player_capital", False):
                summaries[policy]["capital_losses"] = sum(row["capital_lost"] for row in rows)
            if verbose:
                summary = summaries[policy]
                extra = "".join(f", {name[8:].replace('_', ' ')} {value:.1f}" for name, value in summary.items()
                                if name.startswith("average_")
                                and name not in ("average_reward", "average_steps", "average_capital_lost"))
                capital = f" (capital lost {summary['capital_losses']})" if "capital_losses" in summary else ""
                print(f"{policy}: wins {wins}/{completed}, losses {summary['losses']}{capital}, "
                      f"timeouts {summary['timeouts']}, average reward {summary['average_reward']:.2f}{extra}")
            if summaries[policy]["quits"]:
                break
    finally:
        agent.rng = original_rng
        agent.last_action_source = original_source
        agent.observation_size = original_size
    return summaries


def _checkpoint_state(agent, args, rows, best_score=None):
    state = dict(agent.training_state)
    state.update(mode=args.mode, config=asdict(args.config), seed=args.seed, max_steps=args.max_steps,
                 curriculum_episodes=args.curriculum_episodes, guided_exploration=args.guided_exploration,
                 demonstrations=args.demonstrations,
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
    """Train for args.episodes more episodes; return the learner's own rewards and wins.

    Demonstration episodes train the learner too, but they are the heuristic's
    play, so the returned rewards and wins and the log leave them out.
    """
    rows = list(agent.training_state.get("metrics", []))
    rewards = []
    wins = demonstrations = 0
    # The recorded history rebuilds these, so a resumed run gates demonstrations identically.
    recent = {True: collections.deque(maxlen=DEMONSTRATION_WINDOW),
              False: collections.deque(maxlen=DEMONSTRATION_WINDOW)}
    for row in rows:
        recent[bool(row.get("demonstration"))].append(row["reward"])
    start_episode = agent.episodes_trained
    for offset in range(args.episodes):
        episode = start_episode + offset
        config = episode_config(args, episode)
        demonstration = is_demonstration(args, episode) and demonstrations_ahead(recent)
        env = make_env(config, seed=args.seed + episode, render=args.render_training,
                       observation=agent.observation_mode)
        try:
            result = run_episode(env, agent, episode, args.max_steps, train=True,
                                 seed=args.seed + episode, render_delay_ms=args.render_delay_ms,
                                 guided_exploration_rate=args.guided_exploration,
                                 use_guidance=not args.no_guidance,
                                 policy="heuristic" if demonstration else "learned",
                                 anneal=curriculum_complete(args, episode))
        finally:
            env.close()
        recent[demonstration].append(result.reward)
        if demonstration:
            demonstrations += 1
        else:
            rewards.append(result.reward)
            wins += int(result.won)
        rows.append(dict(episode=episode + 1, seed=args.seed + episode, reward=result.reward,
                         outcome=result.outcome, steps=result.steps, epsilon=agent.epsilon,
                         guidance_actions=result.guidance_actions, random_actions=result.random_actions,
                         policy_actions=result.policy_actions, q_table_size=len(agent.q_table),
                         city_strength=difficulty_level(config), demonstration=demonstration))
        _checkpoint_state(agent, args, rows)
        if args.log_every and (offset == 0 or (episode + 1) % args.log_every == 0):
            window = rewards[-args.log_every:]
            average = sum(window) / len(window) if window else 0.0
            played = "demonstration" if demonstration else f"guided {result.guidance_actions}/{result.steps}"
            print(f"episode {episode + 1} | avg reward {average:.2f} | wins {wins}/{len(rewards)} | "
                  f"demonstrations {demonstrations} | epsilon {agent.epsilon:.3f} | "
                  f"{played} | Q states {len(agent.q_table)}")
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
    # Demonstrations show the heuristic's play, so only table growth includes them.
    own = [row for row in rows if not row.get("demonstration")]
    for index, (field, label, color) in enumerate(panels):
        origin_x = (index % 2) * width / 2 + padding
        origin_y = (index // 2) * height / 2 + padding
        plot_w, plot_h = width / 2 - padding - 30, height / 2 - padding - 55
        if field != "q_table_size" and len(own) < len(rows):
            label += " without demonstrations"
        parts.append(f'<text x="{origin_x}" y="{origin_y - 25}">{html.escape(label)}</text>')
        parts.append(f'<path class="axis" fill="none" d="M{origin_x},{origin_y}v{plot_h}h{plot_w}"/>')
        series = rows if field == "q_table_size" else own
        if not series:
            message = "Only demonstrations so far" if rows else "No training episodes"
            parts.append(f'<text x="{origin_x + 15}" y="{origin_y + 35}">{message}</text>')
            continue
        values = []
        for position, row in enumerate(series):
            if field == "win_rate":
                window = series[max(0, position - 49):position + 1]
                value = sum(item["outcome"] == "win" for item in window) / len(window)
            else:
                value = float(row[field])
            values.append(value)
        low, high = (0.0, 1.0) if field == "win_rate" else (min(values), max(values))
        if low == high:
            low, high = low - 0.5, high + 0.5
        first, last = series[0]["episode"], series[-1]["episode"]
        points = " ".join(f"{origin_x + plot_w * (row['episode'] - first) / max(1, last - first):.2f},"
                          f"{origin_y + plot_h * (1 - (value - low) / (high - low)):.2f}"
                          for row, value in zip(series, values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for value, y in ((high, origin_y), (low, origin_y + plot_h)):
            parts.append(f'<text x="{origin_x - 8}" y="{y + 4}" text-anchor="end">{value:.3g}</text>')
        parts.append(f'<text x="{origin_x}" y="{origin_y + plot_h + 24}">{first}</text>')
        parts.append(f'<text x="{origin_x + plot_w}" y="{origin_y + plot_h + 24}" '
                     f'text-anchor="end">{last}</text>')
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
                    ('agent_kind', 'learning_rate', 'discount_factor', 'epsilon', 'epsilon_decay', 'epsilon_min',
                     'state_precision', 'shaping', 'learning_rate_decay', 'learning_rate_min', 'initial_q',
                     'untried_penalty', 'demonstration_margin', 'margin_on', 'episodes_trained',
                     'observation_schema')
                    if hasattr(agent, name)}
        settings.update({name: getattr(args, name) for name in
                         ('mode', 'seed', 'max_steps', 'eval_seed', 'eval_episodes', 'eval_policies',
                          'eval_guidance', 'validation_seed', 'validation_episodes',
                          'curriculum_episodes', 'guided_exploration', 'demonstrations', 'no_guidance')})
        write_metrics(args.metrics_dir, rows, evaluations=evaluations, config=args.config, settings=settings)
        print(f"metrics written to {args.metrics_dir}")
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
