"""Train and evaluate a Q-learning agent in the Civ-like environment."""

import argparse
import random
from pathlib import Path

import numpy as np

import constants
import game
from learning_ai import QLearningAgent


ACTION_COUNT = len(constants.MOVEMENT_THREE_UNITS)
ACTION_INDEX = {tuple(action): index for index, action in enumerate(constants.MOVEMENT_THREE_UNITS)}
DEFAULT_MODEL_PATH = Path("q_learning_model.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Q-learning AI for ML_CIV6.")
    parser.add_argument("--episodes", type=int, default=1000, help="training episodes to run")
    parser.add_argument("--max-steps", type=int, default=80, help="maximum turns per episode")
    parser.add_argument("--eval-episodes", type=int, default=5, help="greedy evaluation episodes after training")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="where to load/save the Q-table")
    parser.add_argument("--fresh", action="store_true", help="ignore any existing saved model")
    parser.add_argument("--no-save", action="store_true", help="do not save the trained Q-table")
    parser.add_argument("--render", action="store_true", help="render evaluation episodes")
    parser.add_argument("--render-training", action="store_true", help="render training episodes")
    parser.add_argument("--render-delay-ms", type=int, default=0, help="delay between rendered turns")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Q-learning update rate")
    parser.add_argument("--discount-factor", type=float, default=0.95, help="future reward discount")
    parser.add_argument("--epsilon", type=float, default=None, help="starting exploration rate")
    parser.add_argument("--epsilon-decay", type=float, default=0.995, help="episode exploration decay")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="minimum exploration rate")
    parser.add_argument("--state-precision", type=int, default=None, help="observation discretization precision")
    parser.add_argument("--initial-q", type=float, default=None, help="initial Q-value for unseen state-actions")
    parser.add_argument("--guided-exploration", type=float, default=0.35, help="chance epsilon exploration uses guidance")
    parser.add_argument("--no-guidance", action="store_true", help="disable heuristic guidance for unseen states")
    parser.add_argument("--log-every", type=int, default=50, help="training progress interval")
    return parser.parse_args()


def build_agent(args, rng):
    if args.model_path.exists() and not args.fresh:
        agent = QLearningAgent.load(args.model_path, rng=rng)
        if agent.action_count != ACTION_COUNT:
            raise ValueError(
                f"Saved model has {agent.action_count} actions, but this game expects {ACTION_COUNT}."
            )
        if args.epsilon is not None:
            agent.epsilon = args.epsilon
    else:
        agent = QLearningAgent(
            ACTION_COUNT,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            epsilon=1.0 if args.epsilon is None else args.epsilon,
            epsilon_decay=args.epsilon_decay,
            epsilon_min=args.epsilon_min,
            state_precision=8 if args.state_precision is None else args.state_precision,
            initial_q=-100.0 if args.initial_q is None else args.initial_q,
            rng=rng,
        )

    agent.learning_rate = args.learning_rate
    agent.discount_factor = args.discount_factor
    agent.epsilon_decay = args.epsilon_decay
    agent.epsilon_min = args.epsilon_min
    if args.state_precision is not None:
        agent.state_precision = args.state_precision
    if args.initial_q is not None:
        agent.initial_q = args.initial_q
    return agent


def choose_guided_action(rng):
    city = next(obj for obj in game.GAME_OBJECTS if obj.__class__ == game.C_City)
    units = [obj for obj in game.GAME_OBJECTS if obj.__class__ == game.C_Unit]
    directions = [choose_guided_direction(unit, city, rng) for unit in units]

    while len(directions) < 3:
        directions.append("SPACE")

    return ACTION_INDEX[tuple(directions[:3])]


def choose_guided_direction(unit, city, rng):
    if not unit.alive:
        return "SPACE"

    distance_to_city = game.hex_distance([unit.x, unit.y], [city.x, city.y])
    if distance_to_city <= 1 and unit.hp / unit.hp_max < 0.35:
        return "SPACE"

    parity = "EVEN" if unit.y % 2 == 0 else "ODD"
    candidates = []

    for direction in constants.MOVEMENT_ONE_UNIT:
        dx, dy = constants.MOVEMENT_DIR[direction][parity]
        next_x = unit.x + dx
        next_y = unit.y + dy

        if direction == "SPACE":
            score = distance_to_city + 0.25
        elif (int(next_x), int(next_y)) not in game.GAME_MAP.grid:
            score = 999.0
        else:
            score = float(game.hex_distance([next_x, next_y], [city.x, city.y]))
            occupant = game.map_check_for_creatures(next_x, next_y, exclude_object=unit)
            if occupant and occupant.__class__ == game.C_Unit:
                score += 5.0
            elif occupant and occupant.__class__ == game.C_City:
                score -= 0.5

        candidates.append((score, direction))

    best_score = min(score for score, _ in candidates)
    best_directions = [direction for score, direction in candidates if score == best_score]
    return rng.choice(best_directions)


def run_episode(
    env,
    agent,
    episode_number,
    max_steps,
    train,
    render_delay_ms=0,
    guided_exploration_rate=0.0,
    use_guidance=True,
):
    env.game_initialize(ep_number=episode_number)
    state = env.get_observation()
    total_reward = 0.0
    won = False
    steps_taken = 0

    for step in range(max_steps):
        fallback_action = choose_guided_action(agent.rng) if use_guidance else None
        action = agent.choose_action(
            state,
            explore=train,
            fallback_action=fallback_action,
            guided_exploration_rate=guided_exploration_rate,
        )
        next_state, reward, done = env.step(action)

        if train:
            agent.learn(state, action, reward, next_state, done)

        total_reward += reward
        state = next_state
        steps_taken = step + 1

        if render_delay_ms > 0 and env.render:
            game.pygame.time.wait(render_delay_ms)

        if done:
            won = True
            break

    if train:
        agent.finish_episode()

    return total_reward, won, steps_taken


def train_agent(agent, args):
    env = game.Game(ml_ai=True, render=args.render_training)
    rewards = []
    wins = 0

    for episode in range(args.episodes):
        reward, won, steps = run_episode(
            env,
            agent,
            episode,
            args.max_steps,
            train=True,
            render_delay_ms=args.render_delay_ms,
            guided_exploration_rate=args.guided_exploration,
            use_guidance=not args.no_guidance,
        )
        rewards.append(reward)
        wins += int(won)

        if args.log_every and (episode == 0 or (episode + 1) % args.log_every == 0):
            window = min(args.log_every, len(rewards))
            average_reward = sum(rewards[-window:]) / window
            print(
                f"episode {episode + 1:4d}/{args.episodes} | "
                f"avg reward {average_reward:7.2f} | "
                f"wins {wins:4d} | "
                f"epsilon {agent.epsilon:.3f} | "
                f"last steps {steps:3d}"
            )

    return rewards, wins


def evaluate_agent(agent, args):
    env = game.Game(ml_ai=True, render=args.render)
    rewards = []
    wins = 0

    for episode in range(args.eval_episodes):
        reward, won, steps = run_episode(
            env,
            agent,
            episode,
            args.max_steps,
            train=False,
            render_delay_ms=args.render_delay_ms,
            guided_exploration_rate=0.0,
            use_guidance=not args.no_guidance,
        )
        rewards.append(reward)
        wins += int(won)
        print(
            f"eval {episode + 1:3d}/{args.eval_episodes} | "
            f"reward {reward:7.2f} | "
            f"won {won} | "
            f"steps {steps:3d}"
        )

    if rewards:
        print(f"evaluation win rate: {wins}/{args.eval_episodes}")
        print(f"evaluation average reward: {sum(rewards) / len(rewards):.2f}")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    agent = build_agent(args, rng)

    if args.episodes > 0:
        train_agent(agent, args)

    if not args.no_save:
        agent.save(args.model_path)
        print(f"saved model to {args.model_path}")

    if args.eval_episodes > 0:
        evaluate_agent(agent, args)


if __name__ == "__main__":
    main()
