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
    """
        True if there are no legal moves from the given state.
        Checks if the game is finished. Returns True if there are no legal moves left.

    """
    for _ in state.moves():
        return False
    return True

def _diff_move(parent: State, child: State) -> Tuple[int, int]:
    """
        Find which cell on the board changed from parent to child state.

        - Looks for a cell that decreased in value.
        - if multiple changes, picks the one with largest decrease
        - if tied pick the one closest to center
    """
    rows, cols = len(parent.grid), len(parent.grid[0])
    diffs: List[Tuple[int,int,int]] = []  # list of (decrease, row, col)

    for r in range(rows):
        for c in range(cols):
            d = parent.grid[r][c] - child.grid[r][c]
            if d > 0:
                diffs.append((d, r, c))

    # if nothing changed, error
    if not diffs:
        raise ValueError("No decreased cell found between states.")

    # if only one change by 1, return it
    if len(diffs) == 1 and diffs[0][0] == 1:
        return (diffs[0][1], diffs[0][2])

    # otherwise pick largest decrease, tiebreak by closeness to center
    cr, cc = (rows - 1) / 2.0, (cols - 1) / 2.0
    diffs.sort(key=lambda t: (-t[0], (t[1]-cr)**2 + (t[2]-cc)**2))
    _, r, c = diffs[0]
    return (r, c)

def _child_iter(state: State):
    """
        Creates a list of all possible next moves.

        Each move returns a tuple (move, child_state).

        The order is shuffled a little, based on the board
        So that the computer doesn't always explore in the same order.
    
    """
    # Some states may already have buildin functtion
    # if it exists, use it, if not use method below

    # state has a function called children()
    if hasattr(state, "children"):  # checks if function exixts
        children = list(state.children())
        # shuffle the order of moves, so AI doesn't always explore in same order
        rng = random.Random(hash(str(state)) & 0xffffffff)
        rng.shuffle(children)
        for move, child in children: 
            yield move, child
        return

    # state has a function called moves_with_coords()
    if hasattr(state, "moves_with_coords"):  # checks if function exists
        children = list(state.moves_with_coords())
        rng = random.Random(hash(str(state)) & 0xffffffff)
        rng.shuffle(children)
        for move, child in children:
            yield move, child
        return

    # Backup plan
    children = list(state.moves())
    rng = random.Random(hash(str(state)) & 0xffffffff)
    rng.shuffle(children)
    for child in children:
        move = _diff_move(state, child) #finds which cell has changed
        yield move, child # returns moves and the new state

def _static_eval(state: State) -> float:
    """
        Gives a score to a board position.
        Higher is better for the current player.
    """
    # counts the number of active cells
    active = sum(1 for row in state.grid for cell in row if cell > 0)

    score = 50 - active  # fewer active cells may be closer to finishing (tune to your game)

    # Bonus from A1 , if it exists
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

    # adds a small random factor to avoid ties
    score += 0.01 * (1 if active % 2 else -1)
    score += (hash(str(state)) & 0xff) * 1e-5
    return score

@dataclass
class _Result:
    value: float 
    move: Move # which move led to this value

class Agent:
    def __init__(self, size: Tuple[int, int], name: str = "B6"):
        self.name = name
        self.size = size
        self.modes = ["minimax", "alphabeta"] # two search methods
        self.max_depth_default = 6 # default search depth
        self._memo_win: Dict[str, bool] = {} # memoization for win()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, size={self.size}, modes={self.modes})"

    def move(self, state: State, mode: str = "alphabeta") -> Move:
        """
            Decides the next best move from the given board
            use alphabeta by default, because it's faster than minimax
        """
        if _terminal(state):
            return None # no legal moves
        
        if mode not in self.modes:
            mode = "alphabeta" # default to alphabeta if invalid mode

        # how many moves ahead to search
        depth = self.max_depth_default 

        # depending on mode, call the appropriate search function
        # minimax, checks all possible moves
        # alphabeta, prunes branches that don't need to be explored
        if mode == "minimax":
            result = self._minimax(state, depth, True)
        else:
            result = self._alphabeta(state, depth, -math.inf, math.inf, True)

        print(f"[{self.name}] using {mode}, chose move {result.move}")
        return result.move

    def _minimax(self, state: State, depth: int, maximizing: bool) -> _Result:
        """
            Minimax search algorithm.
            - The "max" player tries to biggest score.
            - The "min" player tries to smallest score.
        """
        # if no moves left, one player loses
        if _terminal(state):
            return _Result(-1_000_000 if maximizing else 1_000_000, None)
        
        # if we reached the depth limit, estimate the score
        if depth == 0:
            return _Result(_static_eval(state), None)

        # stores the best move found
        best_move: Move = None

        # checks if this level is maximizing or minimizing
        if maximizing:
            # start with worst possible value
            best_value = -math.inf
            # go through all possible child states
            for move, child in _child_iter(state):
                # recursively call minimax on the child state
                value = self._minimax(child, depth - 1, False).value
                # if this move gives a better value, update it
                if value > best_value:
                    best_value, best_move = value, move
            return _Result(best_value, best_move)
        else:
            # this is for minimizing player
            # starts with the highest possible value
            best_value = math.inf

            # goes through all possible child states
            for move, child in _child_iter(state):
                # call minimax for the next turn
                value = self._minimax(child, depth - 1, True).value
                if value < best_value:
                    best_value, best_move = value, move
            return _Result(best_value, best_move)

    def _alphabeta(self, state: State, depth: int, alpha: float, beta: float, maximizing: bool) -> _Result:
        """
            AlphaBeta pruning search algorithm.
            It skips bracnches that don't need to be explored, and prunes them
        """
        if _terminal(state):
            # maximizing player, this is bad (-1_000_000)
            # minmizing player, this is good (+1_000_000)
            return _Result(-1_000_000 if maximizing else 1_000_000, None)
        
        # if we reached search limit, stop and return
        if depth == 0:
            return _Result(_static_eval(state), None)

        # start exploring moves
        best_move: Move = None

        # maximizing player
        if maximizing:
            value = -math.inf # lowest possible value
            # looks every possible state
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
        """
            Checks if its a winning position
            A state is winning if:
             - Theres a Hinger cell
             - if any moves leads to a losing position for the opponent
            reasults are memoized, to avoid repeat      
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
