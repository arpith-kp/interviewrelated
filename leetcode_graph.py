
from collections import *
from typing import List

from graphviz import Digraph

def courseSchedule():

    """
    https://youtu.be/a4hXpeHZ_-c?list=PLujIAthk_iiO7r03Rl4pUnjFpdHjdjDwy&t=887

    while iterating first mark node as -1 if it's current, within same iteration if you find another node
    with -1 then there is loop.

    if no loop mark to 1.

    -1: visiting
    1: visited
    0: unvisited
    :return:
    """

    def build_course(intervals):
        graph = defaultdict(list)
        for interval in intervals:
            x, y = interval
            graph[y].append(x)
        return graph

    def drawGraph(intervals):
        dot = Digraph()
        vertices = set()
        edges = []
        for interval in intervals:
            x, y = interval
            edges.append(str(x)+str(y))
            vertices.add(x)
            vertices.add(y)

        for ver in vertices:
            dot.node(str(ver), str(ver))
        print(" E ", edges)
        dot.edges(edges)
        #
        dot.render('/Users/z003fwy/workspace/UtilityProject/src/c.jpg', view=True)

    def dfs(visited,  key, vertices, graph):
        if key in visited and visited[key] == -1:
            return False
        if key in visited and visited[key] == 1:
            return True
        visited[key] = -1

        for key in vertices:
            if key in graph and not dfs(visited, key, graph[key], graph):
                return False
        visited[key] = 1
        return True

    def course(n, intervals):
        g = build_course(intervals)
        visited = dict()
        for k, v in g.items():
            val = dfs(visited, k, v, g)
            if val is not None and not val:
                return False
            visited[k] = 1
        return True

    c = [[0, 1], [1, 2], [3, 2], [4,3], [2,4]]
    # c = [("a", "b"), ("a", "e"), ("b", "c"), ("b", "d"), ("d", "e")]
    # print(drawGraph(c))
    return course(5, c)


def allPathsFromSource():

    def allPathsSourceTarget( graph: List[List[int]])-> List[List[int]]:
        def dfs(formed, a=1):
            print(f" "*a, f"Begin {formed}, val {formed[-1]}")
            if formed[-1] == n - 1:
                sol.append(formed)
                print(f" " * a, f"solution: {sol}")
                return
            for child in graph[formed[-1]]:
                s = formed + [child]
                dfs(s, a*4)
                print(f" " * a, f"End {s}")
        sol, n = [], len(graph)
        dfs([0])
        return sol

    graph = [[1, 2], [3], [3], []]
    print(allPathsSourceTarget(graph))

def canVisitroom():


    def canVisitAllRooms(rooms: List[List[int]]) -> bool:
        visited = set()

        def dfs(room: int) -> None:
            if room not in visited:
                visited.add(room)
                for key in rooms[room]:
                    dfs(key)

        dfs(0)
        return len(visited) == len(rooms)

    visited = []
    rooms = [[1,3],[3,0,1],[2],[0]]

    return canVisitAllRooms(rooms)

print(canVisitroom())
