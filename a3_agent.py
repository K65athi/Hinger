"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: B6 (100393447, 100361094, and 100387788)
@date:   10/10/2025

"""

from __future__ import annotations
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math
import time
import random

try:
    from a1_state import State
except ImportError:
    raise SystemExit("All of the files should be in the same directory.")

Move = Optional[Tuple[int, int]]  # None (pass) or (row, col)

def _terminal(state: State) -> bool:
    """True if there are no legal moves from the given state."""
    for _ in state.moves():
        return False
    return True

def _diff_move(parent: State, child: State) -> Tuple[int, int]:
    """
    Infer the 'chosen' cell from parent->child.
    Prefer the cell with the largest decrease. If tie, prefer most central
    (this avoids top-left bias). If exactly one cell changed by 1, take that.
    """
    rows, cols = len(parent.grid), len(parent.grid[0])
    diffs: List[Tuple[int,int,int]] = []  # (delta, r, c), delta = parent - child (positive if decreased)
    for r in range(rows):
        for c in range(cols):
            d = parent.grid[r][c] - child.grid[r][c]
            if d > 0:
                diffs.append((d, r, c))
    if not diffs:
        raise ValueError("No decreased cell found between states.")

    if len(diffs) == 1 and diffs[0][0] == 1:
        return (diffs[0][1], diffs[0][2])

    cr, cc = (rows - 1) / 2.0, (cols - 1) / 2.0
    diffs.sort(key=lambda t: (-t[0], (t[1]-cr)**2 + (t[2]-cc)**2))
    _, r, c = diffs[0]
    return (r, c)

def _child_iter(state: State):
    """
    Yield (move, child_state). Deterministically shuffle child order
    based on state to avoid positional bias on ties.
    """
    # If your State exposes richer APIs, prefer them:
    if hasattr(state, "children"):  # yields (move, child)
        children = list(state.children())
        rng = random.Random(hash(str(state)) & 0xffffffff)
        rng.shuffle(children)
        for move, child in children:
            yield move, child
        return

    if hasattr(state, "moves_with_coords"):  # yields (move, child)
        children = list(state.moves_with_coords())
        rng = random.Random(hash(str(state)) & 0xffffffff)
        rng.shuffle(children)
        for move, child in children:
            yield move, child
        return

    # Fallback: only have .moves() -> child states
    children = list(state.moves())
    rng = random.Random(hash(str(state)) & 0xffffffff)
    rng.shuffle(children)
    for child in children:
        move = _diff_move(state, child)
        yield move, child

def _static_eval(state: State) -> float:
    """
    Higher is better for the current player.
    """
    active = sum(1 for row in state.grid for cell in row if cell > 0)
    score = 50 - active  # fewer active cells may be closer to finishing (tune to your game)

    # Bonus for number of regions
    try:
        score += 0.05 * state.numRegions()
    except Exception:
        pass

    # Bonus for 'hingers'
    try:
        hingers = state.numHingers()
        score += 2 * hingers
    except Exception:
        pass

    # Mild parity signal + tiny deterministic jitter to break ties
    score += 0.01 * (1 if active % 2 else -1)
    score += (hash(str(state)) & 0xff) * 1e-5
    return score

@dataclass
class _Result:
    value: float
    move: Move

class Agent:
    def __init__(self, size: Tuple[int, int], name: str = "B6"):
        self.name = name
        self.size = size
        self.modes = ["minimax", "alphabeta"]
        self.max_depth_default = 6
        self._memo_win: Dict[str, bool] = {}

    def __str__(self) -> str:
        return f"Agent(name={self.name}, size={self.size}, modes={self.modes})"

    def move(self, state: State, mode: str = "alphabeta") -> Move:
        if _terminal(state):
            return None
        if mode not in self.modes:
            mode = "alphabeta"

        depth = self.max_depth_default
        if mode == "minimax":
            result = self._minimax(state, depth, True)
        else:
            result = self._alphabeta(state, depth, -math.inf, math.inf, True)

        print(f"[{self.name}] using {mode}, chose move {result.move}")
        return result.move

    def _minimax(self, state: State, depth: int, maximizing: bool) -> _Result:
        if _terminal(state):
            return _Result(-1_000_000 if maximizing else 1_000_000, None)
        if depth == 0:
            return _Result(_static_eval(state), None)

        best_move: Move = None
        if maximizing:
            best_value = -math.inf
            for move, child in _child_iter(state):
                value = self._minimax(child, depth - 1, False).value
                if value > best_value:
                    best_value, best_move = value, move
            return _Result(best_value, best_move)
        else:
            best_value = math.inf
            for move, child in _child_iter(state):
                value = self._minimax(child, depth - 1, True).value
                if value < best_value:
                    best_value, best_move = value, move
            return _Result(best_value, best_move)

    def _alphabeta(self, state: State, depth: int, alpha: float, beta: float, maximizing: bool) -> _Result:
        if _terminal(state):
            return _Result(-1_000_000 if maximizing else 1_000_000, None)
        if depth == 0:
            return _Result(_static_eval(state), None)

        best_move: Move = None
        if maximizing:
            value = -math.inf
            for move, child in _child_iter(state):
                result = self._alphabeta(child, depth - 1, alpha, beta, False)
                if result.value > value:
                    value, best_move = result.value, move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return _Result(value, best_move)
        else:
            value = math.inf
            for move, child in _child_iter(state):
                result = self._alphabeta(child, depth - 1, alpha, beta, True)
                if result.value < value:
                    value, best_move = result.value, move
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return _Result(value, best_move)

    def win(self, state: State) -> bool:
        key = str(state)
        if key in self._memo_win:
            return self._memo_win[key]

        # 1) Instant win if a hinger move exists
        rows, cols = len(state.grid), len(state.grid[0])
        before = state.numRegions()
        for i in range(rows):
            for j in range(cols):
                if state.grid[i][j] == 1:
                    new_grid = [row[:] for row in state.grid]
                    new_grid[i][j] = 0
                    after = State(new_grid).numRegions()
                    if after > before:
                        self._memo_win[key] = True
                        return True

        # 2) Minimax-style: winning if any child is losing for opponent
        for child in state.moves():
            if not self.win(child):
                self._memo_win[key] = True
                return True

        self._memo_win[key] = False
        return False

#
# Tester
#
def _legal_moves_positions(state: State) -> List[Tuple[int,int]]:
    poss = set()
    for child in state.moves():
        try:
            poss.add(_diff_move(state, child))
        except Exception:
            pass
    return sorted(poss)

def tester():
    boardA = [
        [1, 1, 0, 0, 2],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1]
    ]
    sA = State(boardA)
    agent = Agent(size=(4, 5), name="B6")
    print(agent)
    print("State A:")
    print(sA)

    for mode in ["minimax", "alphabeta"]:
        t0 = time.perf_counter()
        mv = agent.move(sA, mode)
        dt = time.perf_counter() - t0
        print(f"{mode} suggests move: {mv} | time: {dt:.6f}s | legal? {mv in _legal_moves_positions(sA)}")

    print("win([[1]]) =", agent.win(State([[1]])))
    print("win([[0]]) =", agent.win(State([[0]])))

if __name__ == "__main__":
    tester()
