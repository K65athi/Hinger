"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes the functions and testers for the algorithm pathways

@author: Group B6 (100387788, 100361094, 100393447)
@date: 30/09/2025 (finished)
"""

import time
import random
from collections import deque
from typing import List
import heapq
# from heapq import heappush, heappop
import itertools

from a1_state import State

###########################################################################

def grids_equal(grid1, grid2):
    # Check if two grids are identical
    return all(row1 == row2 for row1, row2 in zip(grid1, grid2))

def grid_to_tuple(grid):
    # Convert a grid into a tuple for hashing
    return tuple(tuple(row) for row in grid)

###########################################################################

def path_bfs(start: State, end: State):
    """
    Performs a breadth-first search (BFS) to find the shortest sequence of states 
        transforming `start` into `end`.
    """
    start_tuple = grid_to_tuple(start.grid)
    end_tuple   = grid_to_tuple(end.grid)
    
    queue = deque([(start, [start])])
    
    # Track visited states to prevent repeats
    visited = {start_tuple}
    
    # BFS loop
    while queue:
        current_state, path = queue.popleft()
        
        # Goal check
        if grids_equal(current_state.grid, end.grid):
            return path
        
        # Explore possible moves from current state
        for next_state in current_state.moves():
            next_tuple = grid_to_tuple(next_state.grid)

            if next_tuple not in visited:
                visited.add(next_tuple)
                queue.append((next_state, path+[next_state]))
    
    # No path found
    return None

def test_path_bfs():
    """
    Tester for BFS
    """
    # Starting and goal board states
    grid_start = [[2, 0, 0, 1, 0],
                  [0, 3, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [1, 0, 0, 2, 0]]

    grid_goal  = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]

    # Create State objects for BFS input
    start_state = State(grid_start)
    goal_state  = State(grid_goal)
    
    # Find path
    path = path_bfs(start_state, goal_state)
    
    # Display result
    if path is None:
        print("No valid path found")
    else:
        print(f"Path found in {len(path) - 1} moves")
        for step, state in enumerate(path):
            print(f"\nStep {step}:")
            print(state)

###########################################################################

def path_dfs(start: State, end: State):
    """
    Performs a depth-first search (DFS) to find the shortest sequence of states 
        transforming `start` into `end`.
    """
    # Track visisted states to prevent repeats
    visited = set()
    
    # Recursive DFS function
    def dfs(current: State, path: list):
        current_tuple = grid_to_tuple(current.grid)
        visited.add(current_tuple)

        # Goal check
        if grids_equal(current.grid, end.grid):
            return path

        # Explore possible moves from current state
        for next_state in current.moves():
            next_tuple = grid_to_tuple(next_state.grid)
            
            # Explore only unvisited states to avoid ifinite loop
            if next_tuple not in visited:
                result = dfs(next_state, path + [next_state])
                if result is not None:
                    return result

        return None # If no path found
    
    # Start DFS recursion from state start
    return dfs(start, [start])
    

# Test Harness for DFS

def test_path_dfs():
    """
    Tester for DFS
    """
    # Example start and end grid for the hinger game
    grid_start = [[2, 0, 0, 1, 0],
                  [0, 3, 0, 0, 1],
                  [0, 0, 1, 0, 2],
                  [0, 2, 0, 0, 1]]

    grid_goal  = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]
    
    # Create State objects for DFS input
    start_state = State(grid_start)
    goal_state  = State(grid_goal)
    
    # Find path
    path = path_dfs(start_state, goal_state)
    
    # Display result
    if path is None:
        print("No valid path found")
    else:
        print(f"Path found in {len(path) - 1} moves")
        for step, state in enumerate(path):
            print(f"\nStep {step}:")
            print(state)

###########################################################################

def path_iddfs(start: State, end: State):
    """
    Performs an IDDFS to find the shortest sequence of states 
        transforming `start` into `end`.
    """

    # Depth-Limited Search
    def dls(current: State, end: State, limit: int, path: List[State], visited: set):
        if grids_equal(current.grid, end.grid):
            return path

        if limit == 0:
            return None

        current_tuple = grid_to_tuple(current.grid)
        visited.add(current_tuple)

        for next_state in current.moves():
            next_tuple = grid_to_tuple(next_state.grid)
            if next_tuple not in visited:
                result = dls(next_state, end, limit - 1, path + [next_state], visited)
                if result:
                    return result
        return None

    # Deepen search until path is found or max depth of 50 is reached
    max_depth = 50
    for depth in range(max_depth):
        visited = set()
        result = dls(start, end, depth, [start], visited)
        if result is not None:
            return result
        
    # No valid path found
    return None
    
def test_path_iddfs():
    grid_start = [[2, 0, 0, 0, 0],
                  [0, 3, 2, 0, 0],
                  [1, 0, 1, 0, 3],
                  [1, 0, 0, 2, 0]]

    grid_goal  = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]
    
    start_state = State(grid_start)
    goal_state  = State(grid_goal)
    
    path = path_iddfs(start_state, goal_state)
    
    if path is None:
        print("No valid path found")
    else:
        print(f"Path found in {len(path) - 1} moves")
        for step, state in enumerate(path):
            print(f"\nStep {step}:")
            print(state)

###########################################################################

# The amount of non-zero cells in the grid is called the heuristic
#    This maintains the algorithm's consistency as only one counter is
#    added or removed when playing
def path_astar(start: State, end: State):
    """
    Finds valid path using A*
    """
    def heuristic(state: State):
        return sum(1 for row in state.grid for cell in row if cell != 0) # Insane one-liner, I love you Python

    open_set = []
    counter = itertools.count()
    
    start_tuple = grid_to_tuple(start.grid)
    goal_tuple  = grid_to_tuple(end.grid)

    heapq.heappush(open_set, (heuristic(start), 0, next(counter), start_tuple, start, [start]))
    visited = {start_tuple: 0}

    while open_set:
        i, j, k, current_tuple, current, path = heapq.heappop(open_set)

        if visited.get(current_tuple, float("inf")) < j:
            continue

        if grids_equal(current.grid, end.grid):
            return path

        for next_state in current.moves():
            next_tuple = grid_to_tuple(next_state.grid)
            next_j = j + 1
            next_i = next_j + heuristic(next_state)

            if next_tuple not in visited or next_j < visited[next_tuple]:
                visited[next_tuple] = next_j
                heapq.heappush(open_set, (next_i, next_j, next(counter), next_tuple, next_state, path + [next_state]))
                
    return None

def test_path_astar():
    grid_start = [[2, 0, 1, 0, 0],
                  [2, 1, 0, 0, 1],
                  [0, 0, 3, 0, 0],
                  [2, 2, 0, 0, 0]]

    grid_goal =  [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]
    
    start_state = State(grid_start)
    goal_state  = State(grid_goal)
    
    path = path_astar(start_state, goal_state)
    
    # Display result
    if path is None:
        print("No valid path found")
    else:
        print(f"Path found in {len(path) - 1} moves")
        for step, state in enumerate(path):
            print(f"\nStep {step}:")
            print(state)

###########################################################################

def compare():
    """
    Compares performance of the different algorithms: BFS, DFS, IDDFS, and A*
    """

    # List of test grids (each entry is a tuple of start and goal)
    test_grids = [([[0, 1, 2], # Start 1
                    [0, 1, 1],
                    [1, 0, 3]],
                   
                   [[1, 2, 3], # Goal 1
                    [1, 2, 1],
                    [0, 0, 2]]),


                  ([[0, 1, 2], # Start 1
                    [0, 1, 1],
                    [1, 0, 3]],
                   
                   [[0, 2, 1], # Goal 2
                    [1, 3, 1],
                    [2, 1, 1]]),

                  ([[0, 1, 2], # Start 3
                    [0, 1, 1],
                    [1, 0, 3]],
                   
                   [[2, 0, 1], # Goal 3
                    [1, 1, 0],
                    [0, 0, 0]])
                  ]

    # Map algorithms to their functions
    search_algorithms = {"BFS"   : path_bfs,
                         "DFS"   : path_dfs,
                         "IDDFS" : path_iddfs,
                         "A*"    : path_astar}

    # Time taken and success for each algorithm
    times     = {name: [] for name in search_algorithms.keys()}
    completed = {name: [] for name in search_algorithms.keys()}

    for i, (start_grid, goal_grid) in enumerate(test_grids, start=1):
        start_state = State(start_grid)
        goal_state  = State(goal_grid)

        print(f"\nTest Case {i}:")
        print(f"Start Grid:\n{start_state}")
        print(f"Goal Grid:\n{goal_state}")

        # Iterate through each algorithm
        for name, func in search_algorithms.items():
            start_time = time.time()
            path = func(start_state, goal_state)
            completion_time = time.time() - start_time

            # If elapsed time exceeds 30 seconds, kill it
            if completion_time > 30:
                print(f"{name} exceeded 30 seconds and was terminated")
                times[name].append("30+")
                completed[name].append(False)
            else:
                times[name].append(completion_time)
                completed[name].append(path is not None and grids_equal(path[-1].grid, goal_state.grid))
                print(f"{name} took {completion_time:.4f} seconds")

    # Compute averages of completions
    average_times = {}
    for name, lst in times.items():
        adjusted = [min(t, 30) for t in lst]
        average_times[name] = sum(adjusted) / len(adjusted)

    print("\nAverage Times:")
    for name, avg in average_times.items():
        print(f"{name}: {avg:.4f} seconds")

###########################################################################

def test_all():
    """
    Run each algorithm's tester
    """
    print("Tests:")
    print("BFS")
    test_path_bfs()

    print("DFS")
    test_path_dfs()

    print("IDDFS")
    test_path_iddfs()

    print("A*")
    test_path_astar()

###########################################################################

# Main
test_all()
compare()

