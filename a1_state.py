# a1_state.py

class State:
    def __init__(self, grid):
        """
        Creates a new State object with the given grid.
        each cell in the grid is an integer representing the number of counters in that cell.
        0 indicates an inactive cell.
        1 or more indicates an active cell.
        """
        self.grid = grid
        
    def __str__(self):
        """
        Return a readable string version of the grid for printing.
        """
        return '\n'.join(' '.join(str(cell) for cell in row) for row in self.grid)

    def moves(self):
        """
        Yield all possible next states by removing one counter from each active cell.
        """
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.grid[i][j] > 0:  # active cell
                    new_grid = [row[:] for row in self.grid]  # copy the grid
                    new_grid[i][j] -= 1
                    yield State(new_grid)

    def numRegions(self):
        """
        Count the number of active regions on the board.
        """
        visited = set()
        regions = 0
        rows, cols = len(self.grid), len(self.grid[0])

        def dfs(r, c):
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and
                        self.grid[nr][nc] > 0 and (nr, nc) not in visited):
                        visited.add((nr, nc))
                        dfs(nr, nc)

        for i in range(rows):
            for j in range(cols):
                if self.grid[i][j] > 0 and (i, j) not in visited:
                    regions += 1
                    visited.add((i, j))
                    dfs(i, j)

        return regions

    def numHingers(self):
        """
        Count how many hinger cells exist in the current state.
        """
        count = 0
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.grid[i][j] == 1:
                    before = self.numRegions()
                    new_grid = [row[:] for row in self.grid]
                    new_grid[i][j] = 0
                    after = State(new_grid).numRegions()
                    if after > before:
                        count += 1
        return count

def tester():
    boardA = [
        [1, 1, 0, 0, 2],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1]
    ]
    sA = State(boardA)
    print("State A:")
    print(sA)
    print("Number of regions:", sA.numRegions())
    print("Number of hingers:", sA.numHingers())
    print("\nGenerated moves (showing first 2):")
    for i, move in enumerate(sA.moves()):
        if i >= 2:
            break
        print(move)
        print()
        

if __name__ == "__main__":
    tester()
