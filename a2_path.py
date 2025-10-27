
def path_BFS(start, end):
    return None
"""
## Breadth-First Seearch template

from collections import deque

# Define the BFS function
def bfs(tree, start):
    visited = []  # List to keep track of visited nodes
    queue = deque([start])  # Initialize the queue with the starting node

    while queue:  # While there are still nodes to process
        node = queue.popleft()  # Dequeue a node from the front of the queue

        if node not in visited:  # Check if the node has been visited
            visited.append(node)  # Mark the node as visited
            print(node, end=" ")  # Output the visited node

            # Enqueue all unvisited neighbors (children) of the current node
            for neighbor in tree[node]:
                if neighbor not in visited:
                    queue.append(neighbor)  # Add unvisited neighbors to the queue
"""

def path_DFS(start, end):
    return None
"""
## Depth-First Seearch template

# Recursive DFS function
def dfs_recursive(tree, node, visited=None):
    if visited is None:
        visited = set()  # Initialize the visited set
    visited.add(node)    # Mark the node as visited
    print(node)          # Print the current node (for illustration)
    for child in tree[node]:  # Recursively visit children
        if child not in visited:
            dfs_recursive(tree, child, visited)

# Run DFS starting from node 'A'
dfs_recursive(tree, 'A')
"""

def path_IDDFS(start, end):
    return None

# Comments explaining "heuristic function" used
def path_astar(start, end):
    return None

# For testing
def tester():
    return True

# Evaulate and compare effectiveness of different algorithms
def compare():
    return None

# Optional, I think
def min_safe(start, end):
    return None

