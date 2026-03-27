import argparse
import csv
import itertools
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from ..checkers.board import Board
from ..constants import BLACK, WHITE
from .algorithm import minimax
from .rl_algorithm import (
    DRAW_REWARD,
    LOSE_REWARD,
    WIN_REWARD,
    RLAgent,
)
from .training import reset_board


class MinimaxAgent:
    """Thin wrapper to expose minimax with the same interface as RLAgent."""

    def __init__(self, depth: int = 1) -> None:
        self.depth = depth

    def select_entry(self, board: Board, color, train: bool = False):
        max_player = color == BLACK
        _, new_board, move, piece, removed = minimax(deepcopy(board), self.depth, max_player, None)
        if move is None:
            return None, None, None
        return None, None, (new_board, move, piece, removed)


def play_game(agent_black, agent_white, max_turns: int = 120) -> str:
    """Play one game between two agents; return outcome 'black', 'white', or 'draw'."""
    board = reset_board()
    current_color = BLACK
    turns = 0

    while turns < max_turns:
        agent = agent_black if current_color == BLACK else agent_white
        _, _, entry = agent.select_entry(board, current_color, train=False)
        if entry is None:
            return "white" if current_color == BLACK else "black"

        board, _, _, _ = entry
        winner = board.winner()
        if winner:
            return winner.lower()

        current_color = WHITE if current_color == BLACK else BLACK
        turns += 1

    return "draw"


def run_matchup(
    name_a: str,
    agent_a,
    name_b: str,
    agent_b,
    games: int,
    max_turns: int,
    phase: str | None = None,
    game_log: List[Dict[str, object]] | None = None,
) -> Dict[str, Dict[str, int]]:
    """Run a best-of-N series alternating colors; return win/draw counts and optionally log each game."""
    stats = {
        name_a: {"wins": 0, "losses": 0, "draws": 0},
        name_b: {"wins": 0, "losses": 0, "draws": 0},
    }

    for idx in range(games):
        if idx % 2 == 0:
            outcome = play_game(agent_a, agent_b, max_turns)
            black_name, white_name = name_a, name_b
            if outcome == "black":
                stats[name_a]["wins"] += 1
                stats[name_b]["losses"] += 1
            elif outcome == "white":
                stats[name_a]["losses"] += 1
                stats[name_b]["wins"] += 1
            else:
                stats[name_a]["draws"] += 1
                stats[name_b]["draws"] += 1
        else:
            outcome = play_game(agent_b, agent_a, max_turns)
            black_name, white_name = name_b, name_a
            if outcome == "black":
                stats[name_b]["wins"] += 1
                stats[name_a]["losses"] += 1
            elif outcome == "white":
                stats[name_b]["losses"] += 1
                stats[name_a]["wins"] += 1
            else:
                stats[name_a]["draws"] += 1
                stats[name_b]["draws"] += 1

        if game_log is not None:
            game_log.append(
                {
                    "phase": phase or "",
                    "pair": f"{name_a}_vs_{name_b}",
                    "game_index": idx,
                    "black": black_name,
                    "white": white_name,
                    "winner": outcome,
                }
            )

    return stats


def score_series(series_stats: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """Convert win/draw counts into point scores (win=1, draw=0.5)."""
    scores: Dict[str, float] = {}
    for name, stat in series_stats.items():
        wins = stat["wins"]
        draws = stat["draws"]
        scores[name] = wins + 0.5 * draws
    return scores


def play_episode_train_agent(
    train_agent: RLAgent,
    opponent_agent,
    train_color,
    max_turns: int = 120,
) -> str:
    """Train one agent against a fixed opponent for a single episode."""
    board = reset_board()
    turns = 0
    current_color = BLACK

    while turns < max_turns:
        if current_color == train_color:
            state, action_vec, entry = train_agent.select_entry(board, train_color, train=True)
            if entry is None:
                outcome = "white" if train_color == BLACK else "black"
                base_state = state if state is not None else train_agent.encode_board(board, train_color)
                base_action = action_vec if action_vec is not None else train_agent.encode_action((0, 0, 0, 0))
                next_state = train_agent.encode_board(board, train_color)
                train_agent.train_short_memory(base_state, base_action, LOSE_REWARD, next_state, [], True)
                train_agent.remember(base_state, base_action, LOSE_REWARD, next_state, [], True)
                train_agent.n_games += 1
                return outcome

            next_board, move, piece, removed = entry
            shaped_reward = train_agent.compute_reward(next_board, removed, train_color)
            winner = next_board.winner()

            if winner:
                outcome = winner.lower()
                final_reward = WIN_REWARD if outcome == ("black" if train_color == BLACK else "white") else LOSE_REWARD
                next_state = train_agent.encode_board(next_board, train_color)
                train_agent.train_short_memory(state, action_vec, final_reward, next_state, [], True)
                train_agent.remember(state, action_vec, final_reward, next_state, [], True)
                train_agent.n_games += 1
                return outcome

            _, _, opp_entry = opponent_agent.select_entry(next_board, WHITE if train_color == BLACK else BLACK, train=False)
            if opp_entry is None:
                outcome = "black" if train_color == BLACK else "white"
                next_state = train_agent.encode_board(next_board, train_color)
                train_agent.train_short_memory(state, action_vec, WIN_REWARD, next_state, [], True)
                train_agent.remember(state, action_vec, WIN_REWARD, next_state, [], True)
                train_agent.n_games += 1
                return outcome

            opp_board, _, _, _ = opp_entry
            outcome = opp_board.winner()
            done = outcome is not None
            next_state = train_agent.encode_board(opp_board, train_color)
            next_actions = train_agent.available_action_vectors(opp_board, train_color) if not done else []
            final_reward = WIN_REWARD if (done and outcome.lower() == ("black" if train_color == BLACK else "white")) else LOSE_REWARD if done else shaped_reward

            train_agent.train_short_memory(state, action_vec, final_reward, next_state, next_actions, done)
            train_agent.remember(state, action_vec, final_reward, next_state, next_actions, done)
            board = opp_board
        else:
            _, _, opp_entry = opponent_agent.select_entry(board, current_color, train=False)
            if opp_entry is None:
                return "white" if current_color == BLACK else "black"
            board, _, _, _ = opp_entry
            winner = board.winner()
            if winner:
                return winner.lower()

        current_color = WHITE if current_color == BLACK else BLACK
        turns += 1

    train_agent.n_games += 1
    return "draw"


def evaluate_head_to_head(
    name_a: str,
    agent_a,
    name_b: str,
    agent_b,
    games: int,
    max_turns: int,
    phase: str | None = None,
    game_log: List[Dict[str, object]] | None = None,
) -> Tuple[float, float]:
    stats = run_matchup(name_a, agent_a, name_b, agent_b, games, max_turns, phase=phase, game_log=game_log)
    scores = score_series(stats)
    total = games
    winrate_a = scores[name_a] / total
    winrate_b = scores[name_b] / total
    return winrate_a, winrate_b


def alternating_training(
    name_a: str,
    agent_a: RLAgent,
    name_b: str,
    agent_b: RLAgent,
    rounds: int,
    episodes_per_round: int,
    eval_games: int,
    max_turns: int,
    game_log: List[Dict[str, object]],
    training_log: List[Dict[str, object]],
) -> None:
    for r in range(1, rounds + 1):
        win_a, win_b = evaluate_head_to_head(
            name_a,
            agent_a,
            name_b,
            agent_b,
            eval_games,
            max_turns,
            phase="alternating_eval",
            game_log=game_log,
        )
        if win_a == win_b:
            weaker = random.choice([(name_a, agent_a), (name_b, agent_b)])
        else:
            weaker = (name_a, agent_a) if win_a < win_b else (name_b, agent_b)

        weaker_name, weaker_agent = weaker
        opponent_agent = agent_b if weaker_agent is agent_a else agent_a

        for ep in range(episodes_per_round):
            train_color = BLACK if ep % 2 == 0 else WHITE
            play_episode_train_agent(weaker_agent, opponent_agent, train_color=train_color, max_turns=max_turns)
            weaker_agent.update_epsilon(current_winrate=win_a if weaker_agent is agent_a else win_b)

        weaker_agent.train_long_memory()
        training_log.append(
            {
                "round": r,
                "trained": weaker_name,
                "pre_winrate_a": win_a,
                "pre_winrate_b": win_b,
                "episodes": episodes_per_round,
            }
        )
        print(f"Round {r}: trained {weaker_name}; pre-train winrates A={win_a:.3f}, B={win_b:.3f}")


def load_agents(model_dir: Path) -> Dict[str, RLAgent]:
    agents: Dict[str, RLAgent] = {}
    for path in sorted(model_dir.glob("*.pth")):
        agent = RLAgent(model_path=path)
        agent.load()
        agents[path.name] = agent
    return agents


def main(
    games_per_pair: int,
    minimax_depth: int,
    eval_vs_minimax: int,
    training_rounds: int,
    training_episodes: int,
    training_eval_games: int,
    max_turns: int,
) -> None:
    model_dir = Path(__file__).resolve().parent.parent / "model"
    failures_dir = model_dir / "failures"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = model_dir / "tournament_runs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)

    agents = load_agents(model_dir)
    if len(agents) < 2:
        raise SystemExit("Need at least two models to run tournament.")

    minimax_agent = MinimaxAgent(depth=minimax_depth)
    minimax_name = f"minimax_d{minimax_depth}"

    scores: Dict[str, float] = {name: 0.0 for name in agents}
    series_logs: List[Dict[str, object]] = []
    game_logs: List[Dict[str, object]] = []
    training_logs: List[Dict[str, object]] = []

    # models vs models
    for name_a, name_b in itertools.combinations(agents.keys(), 2):
        stats = run_matchup(
            name_a,
            agents[name_a],
            name_b,
            agents[name_b],
            games_per_pair,
            max_turns,
            phase="round_robin",
            game_log=game_logs,
        )
        series_scores = score_series(stats)
        scores[name_a] += series_scores[name_a]
        scores[name_b] += series_scores[name_b]
        series_logs.append(
            {
                "phase": "round_robin",
                "name_a": name_a,
                "name_b": name_b,
                "games": games_per_pair,
                "wins_a": stats[name_a]["wins"],
                "draws": stats[name_a]["draws"],
                "wins_b": stats[name_b]["wins"],
                "points_a": series_scores[name_a],
                "points_b": series_scores[name_b],
            }
        )
        print(f"Match {name_a} vs {name_b}: {stats}")

    # models vs minimax
    for name, agent in agents.items():
        stats = run_matchup(
            name,
            agent,
            minimax_name,
            minimax_agent,
            games_per_pair,
            max_turns,
            phase="round_robin_minimax",
            game_log=game_logs,
        )
        series_scores = score_series(stats)
        scores[name] += series_scores[name]
        series_logs.append(
            {
                "phase": "round_robin_minimax",
                "name_a": name,
                "name_b": minimax_name,
                "games": games_per_pair,
                "wins_a": stats[name]["wins"],
                "draws": stats[name]["draws"],
                "wins_b": stats[minimax_name]["wins"] if minimax_name in stats else stats[name]["losses"],
                "points_a": series_scores[name],
                "points_b": series_scores[minimax_name] if minimax_name in series_scores else games_per_pair - series_scores[name],
            }
        )
        print(f"Match {name} vs {minimax_name}: {stats}")

    ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_two = ranking[:2]
    top_names = {top_two[0][0], top_two[1][0]}
    print("Ranking:")
    for name, score in ranking:
        print(f"  {name}: {score:.2f} points")

    for name, agent in agents.items():
        if name not in top_names:
            src = model_dir / name
            dest = failures_dir / name
            if dest.exists():
                dest.unlink()
            src.rename(dest)
            print(f"Moved {name} to {dest}")

    survivor_names = list(top_names)
    survivor_agents = {name: agents[name] for name in survivor_names}

    # evaluate top two vs minimax
    for name in survivor_names:
        stats = run_matchup(
            name,
            survivor_agents[name],
            minimax_name,
            minimax_agent,
            eval_vs_minimax,
            max_turns,
            phase="post_selection_minimax",
            game_log=game_logs,
        )
        series_scores = score_series(stats)
        series_logs.append(
            {
                "phase": "post_selection_minimax",
                "name_a": name,
                "name_b": minimax_name,
                "games": eval_vs_minimax,
                "wins_a": stats[name]["wins"],
                "draws": stats[name]["draws"],
                "wins_b": stats[minimax_name]["wins"] if minimax_name in stats else stats[name]["losses"],
                "points_a": series_scores[name],
                "points_b": series_scores[minimax_name] if minimax_name in series_scores else eval_vs_minimax - series_scores[name],
            }
        )
        print(f"Post-selection eval {name} vs {minimax_name}: {stats}")

    # alternating training between top two
    if len(survivor_names) == 2:
        name_a, name_b = survivor_names
        alternating_training(
            name_a,
            survivor_agents[name_a],
            name_b,
            survivor_agents[name_b],
            rounds=training_rounds,
            episodes_per_round=training_episodes,
            eval_games=training_eval_games,
            max_turns=max_turns,
            game_log=game_logs,
            training_log=training_logs,
        )

    # final eval
    for name in survivor_names:
        stats = run_matchup(
            name,
            survivor_agents[name],
            minimax_name,
            minimax_agent,
            eval_vs_minimax,
            max_turns,
            phase="final_minimax",
            game_log=game_logs,
        )
        series_scores = score_series(stats)
        series_logs.append(
            {
                "phase": "final_minimax",
                "name_a": name,
                "name_b": minimax_name,
                "games": eval_vs_minimax,
                "wins_a": stats[name]["wins"],
                "draws": stats[name]["draws"],
                "wins_b": stats[minimax_name]["wins"] if minimax_name in stats else stats[name]["losses"],
                "points_a": series_scores[name],
                "points_b": series_scores[minimax_name] if minimax_name in series_scores else eval_vs_minimax - series_scores[name],
            }
        )
        print(f"Final eval {name} vs {minimax_name}: {stats}")
        survivor_agents[name].save()

    # persist logs for offline plotting
    series_path = log_dir / "series_results.csv"
    games_path = log_dir / "game_results.csv"
    training_path = log_dir / "training_rounds.csv"
    ranking_path = log_dir / "ranking.csv"

    with series_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "name_a",
                "name_b",
                "games",
                "wins_a",
                "draws",
                "wins_b",
                "points_a",
                "points_b",
            ],
        )
        writer.writeheader()
        writer.writerows(series_logs)

    with games_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["phase", "pair", "game_index", "black", "white", "winner"],
        )
        writer.writeheader()
        writer.writerows(game_logs)

    with training_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["round", "trained", "pre_winrate_a", "pre_winrate_b", "episodes"],
        )
        writer.writeheader()
        writer.writerows(training_logs)

    with ranking_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "score"])
        writer.writeheader()
        for name, score in ranking:
            writer.writerow({"name": name, "score": score})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Round-robin tournament and alternating training for saved models.")
    parser.add_argument("--games-per-pair", type=int, default=25, help="Games per pairing (per model vs model and vs minimax).")
    parser.add_argument("--minimax-depth", type=int, default=1, help="Depth for minimax baseline.")
    parser.add_argument("--eval-vs-minimax", type=int, default=55, help="Games for post-selection minimax evaluations.")
    parser.add_argument("--training-rounds", type=int, default=20, help="Alternating training rounds between top two.")
    parser.add_argument("--training-episodes", type=int, default=30, help="Episodes per training round for the weaker agent.")
    parser.add_argument("--training-eval-games", type=int, default=12, help="Games to estimate who is weaker each round.")
    parser.add_argument("--max-turns", type=int, default=120, help="Max plies per game to cap runtime.")
    args = parser.parse_args()

    main(
        games_per_pair=args.games_per_pair,
        minimax_depth=args.minimax_depth,
        eval_vs_minimax=args.eval_vs_minimax,
        training_rounds=args.training_rounds,
        training_episodes=args.training_episodes,
        training_eval_games=args.training_eval_games,
        max_turns=args.max_turns,
    )
