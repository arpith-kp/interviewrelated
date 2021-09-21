import collections
from typing import List


def longest1():
    def longestLine(M: List[List[int]]) -> int:
        row = collections.defaultdict(int)
        col = collections.defaultdict(int)
        ad = collections.defaultdict(int)  # Ascending diagonal
        dd = collections.defaultdict(int)  # Descending diagonal
        mx = 0
        for i in range(len(M)):
            for j in range(len(M[0])):
                if not M[i][j]:
                    row[i] = col[j] = ad[j + i] = dd[j - i] = 0
                else:
                    row[i] += 1
                    col[j] += 1
                    ad[j + i] += 1
                    dd[j - i] += 1
                    mx = max(mx, row[i], col[j], ad[j + i], dd[j - i])
        return mx

    mat = [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 1]]
    longestLine(mat)


def max_connected_island():
    def maxAreaOfIsland(grid: List[List[int]]) -> int:

        def explore(grid, r, c) -> int:
            if r >= len(grid) or r < 0 or c >= len(grid[0]) or col < 0:
                return 0
            if grid[r][c] == 0:
                return 0
            grid[r][c] = 0
            return 1 + explore(grid, r - 1, c) + explore(grid, r, c - 1) + explore(grid, r + 1, c) + explore(grid, r, c + 1)

        row = len(grid)
        col = len(grid[0])
        max_area = 0
        for i in range(row):
            for j in range(col):
                if grid[i][j] == 1:
                    max_area = max(max_area, explore(grid, i, j))
        return max_area

    print(maxAreaOfIsland(grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]))

def robot_clean():
    def cleanRoom(robot):
        """
        :type robot: Robot
        :rtype: None
        """
        dfs(robot, 0, 0, 0, 1, set())

    def dfs(robot, x, y, direction_x, direction_y, visited):
        robot.clean()
        visited.add((x, y))

        for k in range(4):
            neighbor_x = x + direction_x
            neighbor_y = y + direction_y
            if (neighbor_x, neighbor_y) not in visited and robot.move():
                dfs(robot, neighbor_x, neighbor_y, direction_x, direction_y, visited)
                robot.turnLeft()
                robot.turnLeft()
                robot.move()
                robot.turnLeft()
                robot.turnLeft()
            robot.turnLeft()
            direction_x, direction_y = -direction_y, direction_x

    room = [[1, 1, 1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1, 0, 1, 1], [1, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1]]
    cleanRoom(room)


class Solution:
    def cleanRoom(self, robot):
        def dfs( robot, x, y, direction_x, direction_y, visited):
            robot.clean()
            visited.add((x, y))

            for k in range(4):
                neighbor_x = x + direction_x
                neighbor_y = y + direction_y
                if (neighbor_x, neighbor_y) not in visited and robot.move():
                    dfs(robot, neighbor_x, neighbor_y, direction_x, direction_y, visited)
                    robot.turnLeft()
                    robot.turnLeft()
                    robot.move()
                    robot.turnLeft()
                    robot.turnLeft()
                robot.turnLeft()
                direction_x, direction_y = -direction_y, direction_x
        dfs(robot, 0, 0, 0, 1, set())


def max_min_mat():
    seen = set()
    res=[]

    def maximumMinimumPath(grid: List[List[int]]) -> int:

        def dfs(grid, i, j):
            queue = collections.deque([(i,j),])
            temp=[]

            while queue:
                m,n = queue.popleft()
                temp.append(grid[m][n])
                seen.add((m, n))
                for r, c in (m + 1, n), (m - 1, n), (m, n + 1), (m, n - 1):
                    if r>=len(grid[0]) or r<0 or c>=len(grid[0]) or c<0:
                        continue
                    elif r == len(grid)-1 and c==len(grid[0])-1:
                        if res and  sum(temp) > sum(res[-1]):
                            res.append(temp)
                        elif not res:
                            res.append(temp)
                    elif (r,c) not in seen:
                        seen.add((r,c))
                        queue.append((r,c))


        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if (i,j) not in seen:
                    dfs(grid, i, j)
        max_sum=0
        idx = 0
        for j, i in enumerate(res):
            if sum(i)>max_sum:
               idx = j
               max_sum = sum(i)
        print("Res ", res)
        return min(res[idx])

    grid = [[5, 4, 5], [1, 2, 6], [7, 4, 6]]
    print(maximumMinimumPath(grid))

def shortest_path():
    def shortestPathBinaryMatrix( grid: List[List[int]]) -> int:
        max_row = len(grid) - 1
        max_col = len(grid[0]) - 1
        directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]

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
        queue = collections.deque()
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

    grid = [[0, 0, 0], [1, 1, 0], [1, 1, 0]]
    print(shortestPathBinaryMatrix(grid))

shortest_path()
