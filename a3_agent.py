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

# Import the State class from a1_state.py
try:
    from a1_state import State
except ImportError:
    raise SystemExit("All of the files should be in the same directory.")

Move = Optional[Tuple[int, int]]  # A move is either None (pass) or a tuple (row, col)

"""
    Helper functions and classes for the Hinger agent.

"""

def _terminal(state: State) -> bool:
    """
    if there are no possible moves from the given state, return True (all cells empty).
    """
    for _ in state.moves():
        return False
    return True

def _diff_move(parent: State, child: State) -> Tuple[int, int]:
        """
        Finds the changed cell between parent and child states.
        returns the (row, col) of the cell that was changed.

        """
        for i in range(len(parent.grid)):
             for j in range(len(parent.grid[0])):
                 if child.grid[i][j] == parent.grid[i][j] - 1:
                     return (i, j)
        for i in range(len(parent.grid)):
             for j in range(len(parent.grid[0])):
                 if child.grid[i][j] < parent.grid[i][j]:
                     return (i, j)
        raise ValueError("Error")
        

def _static_eval(state: State) -> float:
    """
    Evaluates how favorable the given state is for the current player.
    A higher score indicates a more favorable position.
    more active cells = better

    """

    active = sum(1 for row in state.grid for cell in row if cell > 0)
    score = float(active)

    # Bonus for number of regions
    try:
        score += 0.05 * state.numRegions()
    except Exception:
        pass
    
    # Small bonus/penalty for odd/even number of active cells
    score += 0.01 * (1 if active % 2 else -1)

    return score

# structure to store result of minimax
@dataclass
class _Result:
    value: float
    move: Move


"""
    Agent Class

"""

class Agent:
    """
    An Ai agent that can play the Hinger game using Minimax and AlphaBeta pruning.
    """
    def __init__(self, size: Tuple[int, int], name: str = "B6"):
        self.name = name
        self.size = size
        self.modes = ["minimax", "alphabeta"]
        self.max_depth_default = 6
        self._memo_win: Dict[str, bool] = {}

    def __str__(self) -> str:
        """ Display agent information """
        return f"Agent(name={self.name}, size={self.size}, modes={self.modes})"


    def move(self, state: State, mode: str = "alphabeta") -> Move:
        """ returns the next move for the given state """
        if _terminal(state):
            return None
    
        if mode not in self.modes:
            mode = "alphabeta"

        depth = self.max_depth_default
        if mode == "minimax":
            result = self._minimax(state, depth, True)
        else:
            result = self._alphabeta(state, depth, -math.inf, math.inf, True)
        return result.move
      

    def _minimax(self, state: State, depth: int, maximizing: bool) -> _Result:
        """ Minimax search algorithm implementation """
        if _terminal(state):
            return _Result(-1_000_000 if maximizing else 1_000_000, None)
    
        if depth == 0:
            return _Result(_static_eval(state), None)
    
        best_move: Move = None
        if maximizing:
            best_value = -math.inf
            for child in state.moves():
                move = _diff_move(state, child)
                value = self._minimax(child, depth - 1, False).value
                if value > best_value:
                    best_value, best_move = value, move
            return _Result(best_value, best_move)
        else:
            best_value = math.inf
            for child in state.moves():
                move = _diff_move(state, child)
                value = self._minimax(child, depth - 1, True).value
                if value < best_value:
                    best_value, best_move = value, move
            return _Result(best_value, best_move)


    def _alphabeta(self, state: State, depth: int, alpha: float, beta: float, maximizing: bool) -> _Result:
        """ Minimax with Alpha-Beta pruning search algorithm implementation """
        if _terminal(state):
            return _Result(-1_000_000 if maximizing else 1_000_000, None)
    
        if depth == 0:
            return _Result(_static_eval(state), None)

        best_move: Move = None
        if maximizing:
            value = -math.inf
            for child in state.moves():
                move = _diff_move(state, child)
                result = self._alphabeta(child, depth - 1, alpha, beta, False)
                if result.value > value:
                    value, best_move = result.value, move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break # pruning
            return _Result(value, best_move)
        else:
            value = math.inf
            for child in state.moves():
                move = _diff_move(state, child)
                result = self._alphabeta(child, depth - 1, alpha, beta, True)
                if result.value < value:
                    value, best_move = result.value, move
                beta = min(beta, value)
                if alpha >= beta:
                    break # pruning
            return _Result(value, best_move)
    

    def win(self, state: State) -> bool:
        """
        Determines if the current state is a winning state for the current player.
        Uses memoization to store previously computed results.   

        """

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

        # 2) Otherwise, minimax-style: if any child is losing for the opponent, current is winning
        for child in state.moves():
            if not self.win(child):
                self._memo_win[key] = True
                return True

        self._memo_win[key] = False
        return False

#
# Tester Function
#

def tester():
    """ Tester function for the Agent """
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
        move = agent.move(sA, mode)
        dt = time.perf_counter() - t0
        print(f"{mode} suggests move: {move} | time: {dt:.6f}s")

    """ Win condition tester """
    print("win([[1]]) =", agent.win(State([[1]])))  # True (hinger present)
    print("win([[0]]) =", agent.win(State([[0]])))  # False (no moves)

if __name__ == "__main__":
    tester()