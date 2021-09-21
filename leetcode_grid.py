from collections import deque
from typing import List


def enclave():
    def numEnclaves(grid: List[List[int]]) -> int:
        r = len(grid)
        c = len(grid[0])

        dp = [[0] * (c+1) for _ in range(r+1)]
        count = 0
        for i in range(r):
            for j in range(c):
                if grid[i][j] == 1:
                    dp[i+1][j+1] = max(grid[i][j], grid[i+1][j], grid[i][j+1])

        print(dp)

    grid = [[0, 0, 0, 0], [1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
    numEnclaves(grid)

def closed_island():
    def closedIsland( grid: List[List[int]]) -> int:

        seen = set()
        def bfs(i, j):
            queue = deque([(i, j)])
            ans =1

            while queue:
                i, j = queue.popleft()
                seen.add((i,j))
                for r,c in (i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1):
                    if r < 0 or r >= len(grid) or c < 0 or c >= len(grid[0]):
                        ans = 0
                    elif (r,c) not in seen and not grid[r][c]:
                        queue.append((r,c))
                        seen.add((r, c))
            return ans

        return sum(bfs(i, j) for i, row in enumerate(grid) for j, cell in enumerate(row) if not cell and (i, j) not in seen)

    grid = [[1, 1, 1, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1, 1, 0], [1, 0, 1, 0, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 0]]
    print(closedIsland(grid))

def bfs_all_direction():
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        max_row = len(grid) - 1
        max_col = len(grid[0]) - 1
        directions = [
            (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # Helper function to find the neighbors of a given cell.
        def get_neighbours(row, col):
            for row_difference, col_difference in directions:
                new_row = row + row_difference
                new_col = col + col_difference
                if not (0 <= new_row <= max_row and 0 <= new_col <= max_col):
                    continue
                if grid[new_row][new_col] != 0:
                    continue
                yield (new_row, new_col)

        # Check that the first and last cells are open.
        if grid[0][0] != 0 or grid[max_row][max_col] != 0:
            return -1

        # Set up the BFS.
        queue = deque()
        queue.append((0, 0))
        grid[0][0] = 1

        # Carry out the BFS.
        while queue:
            row, col = queue.popleft()
            distance = grid[row][col]
            if (row, col) == (max_row, max_col):
                return distance
            for neighbour_row, neighbour_col in get_neighbours(row, col):
                grid[neighbour_row][neighbour_col] = distance + 1
                queue.append((neighbour_row, neighbour_col))

        # There was no path.
        return -1


def general_matrix():

    def helper(mat, a, b):
        """
        I use the bound to "shrink" the search space because once we hit a zero, there is no point in iterating past that point.
        O(m^2 x n^2)
        """
        m = len(mat)
        n = len(mat[0])

        count = 0
        bound = n

        for i in range(a,m):
            for j in range(b, bound):
                if mat[i][j]:
                    count+=1
                else:
                    bound=j
        return count

    def original_solver(mat):
        m = len(mat)
        n = len(mat[0])

        count = 0
        for i in range(m):
            for j in range(n):
                count+=helper(mat,i,j)

        return count


closed_island()
