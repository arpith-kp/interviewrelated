'''
DFS
'''
import collections
from typing import List


def word_numbers(input):
    "iterate on current element and each time keep append to your previos result"
    digit_map = {
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz',
    }

    input = str(input)
    ret = ['']
    for char in input:
        letters = digit_map.get(char, '')
        res = []
        for letter in letters:
            for prefix in ret:
                res.append(prefix + letter)
        ret = res

    return ret


def island():
    """
    Outer two for loops will iterate and find results.

    checker function will keep resetting values marking visited node to 0 and traverse in all direction
    :return:
    """

    grid = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"]
    ]

    def dfs():
        rows = len(grid)
        cols = len(grid[0])
        count = 0
        for i in range(0, rows):
            for j in range(0, cols):
                if grid[i][j] == '1':
                    check_valid(i, j, grid)
                    count = count + 1
        return count

    def check_valid(i, j, grid=None):
        rows = len(grid)
        cols = len(grid[0])

        if not 0 <= i < rows or not 0 <= j < cols or grid[i][j] != '1':
            return

        grid[i][j] = '0'

        check_valid(i + 1, j, grid)
        check_valid(i - 1, j, grid)
        check_valid(i, j + 1, grid)
        check_valid(i, j - 1, grid)

    return dfs()


def wordsearch():
    board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]

    def exist(board: List[List[str]], word: str) -> bool:
        idx = 0
        result = []
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] is not None and board[i][j] in word:
                    check_valid(i, j, board, word, result, idx)
        return ''.join(result) == word

    def check_valid(i, j, board, word, result, idx):

        if (not 0 <= i < len(board) or not 0 <= j < len(board[0])
            or board[i][j] is None or idx >= len(word) or board[i][j] != word[idx]
            or ''.join(result) == word):
            return False
        print(f"word index at {idx} Char at {i}, {j}: {board[i][j]}, result is {result}")

        result.append(board[i][j])
        board[i][j] = None

        check_valid(i + 1, j, board, word, result, idx + 1)
        check_valid(i - 1, j, board, word, result, idx + 1)
        check_valid(i, j + 1, board, word, result, idx + 1)
        check_valid(i, j - 1, board, word, result, idx + 1)

        return True

    exist(board, 'SFCS')


def splitword():
    def numSplits(s: str) -> int:
        """
        Input: s = "aacaba"
        Output: 2
        Explanation: There are 5 ways to split "aacaba" and 2 of them are good.
        ("a", "acaba") Left string and right string contains 1 and 3 different letters respectively.
        ("aa", "caba") Left string and right string contains 1 and 3 different letters respectively.
        ("aac", "aba") Left string and right string contains 2 and 2 different letters respectively (good split).
        ("aaca", "ba") Left string and right string contains 2 and 2 different letters respectively (good split).
        ("aacab", "a") Left string and right string contains 3 and 1 different letters respectively.
        :param s:
        :return:
        """
        left_count = collections.Counter()
        right_count = collections.Counter(s)
        res = 0
        for c in s:
            left_count[c] += 1
            right_count[c] -= 1
            if right_count[c] == 0:
                del right_count[c]

            if len(left_count) == len(right_count):
                res += 1

        return res


class Node:

    def __init__(self, start, end):
        self.right = self.left = None
        self.start = start
        self.end = end

    def insert(self, node) -> bool:
        if node.start >= self.end:
            if not self.right:
                self.right = node
                return True
            else:
                self.right.insert(node)
        elif node.end <= self.start:
            if not self.left:
                self.left = node
                return True
            else:
                self.left.insert(node)
        else:
            return False


class MyCalendar(object):
    def __init__(self):
        self.root = None

    def book(self, start, end):
        if self.root is None:
            self.root = Node(start, end)
            return True
        return self.root.insert(Node(start, end))

class TimeMap(object):

    def __init__(self):
        self.dic = collections.defaultdict(list)

    def set(self, key, value, timestamp):
        self.dic[key].append([timestamp, value])

    def get(self, key, timestamp):
        arr = self.dic[key]
        n = len(arr)

        left = 0
        right = n

        while left < right:
            mid = (left + right) / 2
            if arr[mid][0] <= timestamp:
                left = mid + 1
            elif arr[mid][0] > timestamp:
                right = mid

        return "" if right == 0 else arr[right - 1][1]


class SnapshotArray:

    def __init__(self, length):

        self.id = -1
        self.items = []
        self.dict_map = {}

    def set(self, index, val):

        self.dict_map[index] = val

    def snap(self):

        self.items.append(self.dict_map)
        self.dict_map = self.dict_map.copy()
        self.id += 1
        return self.id

    def get(self, index, snap_id):

        try:
            d = self.items[snap_id]
            return d[index]
        except KeyError:
            return 0


sanpshotArr = SnapshotArray(3)
print(sanpshotArr.set(0, 5),sanpshotArr.snap(),sanpshotArr.set(1, 6),sanpshotArr.get(0, 0))
