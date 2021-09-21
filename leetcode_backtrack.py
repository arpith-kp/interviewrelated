import collections
from copy import copy
from typing import List


def permutation():
    def permutation(input, level=1):
        """
        remove first element from input, do permutation for the result add back original number and add original to input list
        :param input:
        :param idx:
        :param level:
        :return:
        """
        res = list()
        print(f" " * level, f" result {res}, input {input}")
        if len(input) == 1:
            return [input[:]]

        for i in range(len(input)):
            n = input.pop(0)
            perms = permutation(input, level * 2)
            for p in perms:
                p.append(n)

            res.extend(perms)
            input.append(n)
            print(f" " * level, f" result {res}, input {input}")
        return res

    print("Final ", permutation([1, 2, 3]))


def powerset():
    """
    two option to include or not include
    :return:
    """

    def ps(input: List[int]):
        res = []
        subset = []

        def dfs(i):
            if i >= len(input):
                res.append(subset[:])
                return

            subset.append(input[i])
            dfs(i + 1)

            subset.pop()
            dfs(i + 1)

        dfs(0)
        return res

    print("Final ", ps(([1, 2, 3])))


def wordsearch():
    def exists(board, word):
        rows, cols = len(board), len(board[0])
        path = set()

        def dfs(r, c, i):
            if i == len(word):
                return True
            if (r < 0 or c < 0 or
                r >= rows or c >= rows or
                word[i] != board[r][c] or
                (r, c) in path):
                return False
            path.add((r, c))
            res = (dfs(r + 1, c, i + 1) or
                   dfs(r + 1, c, i + 1) or
                   dfs(r + 1, c, i + 1) or
                   dfs(r + 1, c, i + 1)
                   )
            path.remove((r, c))
            return res

        for i in range(rows):
            for j in range(cols):
                if dfs(i, j, 0):
                    return True

        return False

    board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
    word = 'ASA'
    print(exists(board, word))


def combinationsum():
    def combi(input: List[int], target, level=1):
        res = []

        def dfs(i, cur, total, level=1):
            if total == target:
                res.append(cur.copy())
                return
            if total > target or i >= len(input):
                return False

            cur.append(input[i])
            print(f" " * level, f" With number {input[i]} result: {cur}, final {res}")
            dfs(i, cur, total + input[i], level=level * 2)

            cur.pop()
            print(f" " * level, f" Without number {input[i]} result: {cur} final {res}")
            dfs(i + 1, cur, total, level=level * 2)

        dfs(0, [], 0)

        return res

    print(combi([2, 3, 6, 7], 7))


class Solution:
    def getMaximumGold(self, grid: List[List[int]]) -> int:

        # Given a row and a column, what are all the neighbours?
        def options(row, col):
            if row > 0:
                yield (row - 1, col)
            if col > 0:
                yield (row, col - 1)
            if row < len(grid) - 1:
                yield (row + 1, col)
            if col < len(grid[0]) - 1:
                yield (row, col + 1)

            # Keep track of current gold we have, and best we've seen.

        self.current_gold = 0
        self.maximum_gold = 0

        def backtrack(row, col):

            # If there is no gold in this cell, we're not allowed to continue.
            if grid[row][col] == 0:
                return

            # Keep track of this so we can put it back when we backtrack.
            gold_at_square = grid[row][col]

            # Add the gold to the current amount we're holding.
            self.current_gold += gold_at_square

            # Check if we currently have the max we've seen.
            self.maximum_gold = max(self.maximum_gold, self.current_gold)

            # Take the gold out of the current square.
            grid[row][col] = 0

            # Consider all possible ways we could go from here.
            for neigh_row, neigh_col in options(row, col):
                # Recursively call backtrack to explore this way.
                backtrack(neigh_row, neigh_col)

            # Once we're done on this path, backtrack by putting gold back.
            self.current_gold -= gold_at_square
            grid[row][col] = gold_at_square

        # Start the search from each valid starting location.
        for row in range(len(grid)):
            for col in range(len(grid[0])):
                backtrack(row, col)

            # Return the maximum we saw.
        return self.maximum_gold


"""
All backtrack methods should follow three rules

1. Choice (values to applied to columns or to iterate over)
2. Constraint ()
3. Goal
"""


def some_func_that_uses_backtrack(inputs):
    # initialize global variable which will be used to calculate final result

    def common_backtrack_function(cur_val):
        # 3. Check if goal is reached or exceeded boundary conditions

        # 2. Validate constraints to see if current values matches

        # if this succeds update your global result value
        pass

    # 1. Choice
    for val in inputs:
        common_backtrack_function(val)


def partition_palindrome():
    def partition(s):
        res = []
        dfs(s, [], res)
        return res

    def dfs(s, path, res):
        if not s:
            res.append(path)
        for i in range(1, len(s) + 1):
            if isPalindrome(s[:i]):
                dfs(s[i:], path + [s[:i]], res)

    def isPalindrome(s):
        l, r = 0, len(s) - 1
        while l < r:
            if s[l] != s[r]:
                return False
            l += 1
            r -= 1
        return True

    print(partition("racecar"))


def restoreipaddress():
    def restoreIpAddresses(s):
        res = []
        dfs(s, 0, "", res)
        return res

    def dfs(s, idx, path, res):
        if idx > 4:
            return
        if idx == 4 and not s:
            res.append(path[:-1])
            return
        for i in range(1, len(s) + 1):
            if s[:i] == '0' or (s[0] != '0' and 0 < int(s[:i]) < 256):
                dfs(s[i:], idx + 1, path + s[:i] + ".", res)

    print(restoreIpAddresses('25525511135'))


def letter_permutation():
    def letterCasePermutation(S):

        def backtrack(s='', i=0):
            if len(s) == len(S):
                res.append(s)
            else:
                if S[i].isalpha():
                    backtrack(s + S[i].upper(), i + 1)
                    backtrack(s + S[i].lower(), i + 1)
                    return
                if S[i].isdigit():
                    backtrack(s + S[i], i + 1)

        res = []
        backtrack()
        return res

    s = "a1b2"
    print(letterCasePermutation(s))


def permutation_dfs():
    def permute(nums: List[int]) -> List[List[int]]:

        def backtrack(visited=set(), subset=[]):
            if len(subset) == len(nums):
                final_res.append(copy(subset))
            else:
                for j in range(len(nums)):
                    if j not in visited:
                        visited.add(j)
                        backtrack(visited, subset + [nums[j]])
                        visited.remove(j)

        final_res = []
        backtrack()
        return final_res

    def permute_small(nums2: List[int]) -> List[List[int]]:

        def backtrack(nums2, idx, temp=None, used={}):
            if temp is None:
                temp = []
            if len(temp) == len(nums2):
                res.append(temp[:])
                return
            for i in range(len(nums2)):
                if used.get(i): continue
                temp.append(nums[i])
                used[i] = True
                backtrack(nums2, idx + 1, temp, used)
                temp.pop()
                used[i] = False

        res = []
        backtrack(nums2, 0)
        return res

    nums = [1, 2, 3]
    print(permute_small(nums))
    print(permute(nums))


def increasing_subsequence():
    def findSubsequences(nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        res = []
        subsets(nums, 0, [], res)
        return res

    def subsets(nums, index, temp, res):
        if len(nums) >= index and len(temp) >= 2:
            res.append(temp[:])
        used = {}
        for i in range(index, len(nums)):
            if len(temp) > 0 and temp[-1] > nums[i]: continue
            if nums[i] in used: continue
            used[nums[i]] = True
            temp.append(nums[i])
            subsets(nums, i + 1, temp, res)
            temp.pop()

    nums = [4, 6]
    print(findSubsequences(nums))


def studentRec():
    """
    Brute force solution: Generic one
    :return:
    """

    class Solution:

        def __init__(self):
            self.count = 0
            self.M = 1000000007

        def checkRecord(self, n) -> int:
            self.count = 0
            self.gen("", n)
            return self.count

        def gen(self, s, n):
            if n == 0 and self.checkRecordValid(s):
                self.count = (self.count + 1) % self.M
            elif n > 0:
                self.gen(s + "A", n - 1)
                self.gen(s + "L", n - 1)
                self.gen(s + "P", n - 1)

        def checkRecordValid(self, s):

            c = collections.Counter(s)
            try:
                lates = s.index('LLL')
            except ValueError:
                lates = -1
            return len(s) > 0 and lates < 0 and c['A'] < 2

    print(Solution().checkRecord(2))


def combination3():
    def combinationSum(candidates, target):
        def backtrack(tmp, start, end, target):
            if target == 0:
                ans.append(tmp[:])
            elif target > 0:
                for i in range(start, end):
                    tmp.append(candidates[i])
                    backtrack(tmp, i, end, target - candidates[i])
                    tmp.pop()

        ans = []
        candidates.sort(reverse=True)
        backtrack([], 0, len(candidates), target)
        return ans

    candidates = [2, 3, 6, 7]
    target = 7
    print(combinationSum(candidates, target))


def subsets3():
    def subsets(nums: List[int]) -> List[List[int]]:
        def backtrack(start, end, temp=[]):
            ans.append(temp[:])
            if start < len(nums):
                for i in range(start, end):
                    temp.append(nums[i])
                    backtrack(i + 1, end, temp)
                    temp.pop()

        ans = []
        backtrack(0, len(nums))
        return ans

    nums = [1, 2, 3]
    a = [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
    b = [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]
    a_1 = set([tuple(ls) for ls in a])
    b_1 = set([tuple(ls) for ls in b])
    print(a_1.difference(b_1))

    print(subsets(nums))


def permutation3():
    def permute(nums: List[int]) -> List[List[int]]:

        def backtrack(start, end, temp=[]):
            if len(temp) == len(nums):
                ans.append(temp[:])
            else:
                for i in range(start, end):
                    if nums[i] not in temp:
                        temp.append(nums[i])
                        backtrack(start, end, temp)
                        temp.pop()

        ans = []
        backtrack(0, len(nums))
        return ans

    nums = [1, 2, 3]
    res = permute(nums)
    a1 = set(map(tuple, res))
    b1 = set(map(tuple, [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]))
    print(a1.difference(b1))


def palindrome_sub_3():
    def partition(s: str) -> List[List[str]]:

        def backtrack(start, end, tmp=[]):
            if start == end:
                ans.append(tmp[:])
            for i in range(start, end):
                cur = s[start:i + 1]
                if cur == cur[::-1]:
                    tmp.append(cur)
                    backtrack(i + 1, end, tmp)
                    tmp.pop()

        ans = []
        backtrack(0, len(s))
        return ans

    x = partition("aab")
    print(x)
    return x


def maxUniqueSplit():
    def maxUniqueSplits(s: str) -> int:
        def backtrack(start, end, tmp=[]):
            if start == end:
                if len(ans) == 0 or len(tmp) > len(ans[-1]):
                    ans.append(tmp[:])
            for i in range(start, end):
                cur = s[start:i + 1]
                if cur not in tmp:
                    tmp.append(cur)
                    backtrack(i + 1, end, tmp)
                    tmp.pop()

        ans = []
        backtrack(0, len(s))
        print("Re", ans)
        res = len(ans[-1])
        return res

    print(maxUniqueSplits('ababccc'))


def kequalSubset():
    def canPartitionKSubsets(nums: List[int], k: int) -> bool:
        nums_sum = sum(nums)
        if nums_sum % k != 0:
            return False
        subset_sum = nums_sum / k

        ks = [0] * k
        nums.sort(reverse=True)
        visited = [False] * len(nums)

        def can_partition(rest_k, cur_sum=0, next_index=0):
            if rest_k == 1:
                return True

            if cur_sum == subset_sum:
                return can_partition(rest_k - 1)

            for i in range(next_index, len(nums)):
                if not visited[i] and cur_sum + nums[i] <= subset_sum:
                    visited[i] = True
                    if can_partition(rest_k, cur_sum=cur_sum + nums[i], next_index=i + 1):
                        return True
                    visited[i] = False
            return False

        return can_partition(k)


def word_ladder():
    def findLadders(beginWord, endWord, wordList):
        tree, words, n = collections.defaultdict(set), set(wordList), len(beginWord)
        if endWord not in wordList: return []
        found, bq, eq, nq, rev = False, {beginWord}, {endWord}, set(), False
        while bq and not found:
            words -= set(bq)
            for x in bq:
                for y in [x[:i] + c + x[i + 1:] for i in range(n) for c in 'qwertyuiopasdfghjklzxcvbnm']:
                    if y in words:
                        if y in eq:
                            found = True
                        else:
                            nq.add(y)
                        tree[y].add(x) if rev else tree[x].add(y)
            bq, nq = nq, set()
            if len(bq) > len(eq):
                bq, eq, rev = eq, bq, not rev

        def bt(x):
            return [[x]] if x == endWord else [[x] + rest for y in tree[x] for rest in bt(y)]

        return bt(beginWord)

    beginWord = "hit"
    endWord = "cog"
    wordList = ["hot", "dot", "dog", "lot", "log"]
    print(findLadders(beginWord, endWord, wordList))


def backtrack_summary():
    palindrome_sub_3()
    permutation3()
    combination3()
    subsets3()


def word_search_2():
    def exist(board: List[List[str]], word: str) -> bool:
        m = len(board)
        n = len(board[0])

        def dfs(board, i, j, word, res=[]):
            if ''.join(res) == word:
                return True
            if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
                return False

            if board[i][j] != '#':
                res.append(board[i][j])
                c = board[i][j]
                board[i][j] = '#'
                value = dfs(board, i + 1, j, word, res) or\
                        dfs(board, i, j + 1, word, res) or\
                        dfs(board, i - 1, j, word, res) or \
                        dfs(board, i, j - 1, word, res)

                res.pop(-1)
                board[i][j] = c
                return value

            return False

        for i in range(m):
            for j in range(n):
                if dfs(board, i, j, word):
                    return True
        return False

    board = [["a"]]
    word = "a"
    print(exist(board,word))

def path_maze():
    def hasPath(maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        start_row = start[0]
        start_col = start[1]

        end_row = destination[0]
        end_col = destination[1]

        max_row = len(maze)
        max_col = len(maze[0])
        queue = collections.deque([(start_row, start_col)])

        while queue:
            cur_row, cur_col = queue.popleft()

            for next_row, next_col  in (cur_row, cur_col+1), (cur_row,cur_col-1), (cur_row+1,cur_col), (cur_row-1, cur_col):
                if next_row <0 or next_row>max_row or next_col<0 or next_col>max_col:
                    continue
                elif (next_row, next_col) != (end_row, end_col):
                    return True
                elif maze[next_row][next_col]!=1:
                    maze[next_row][next_col] = 1
                    queue.append((next_row, next_col))

        return False

    maze = [[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [1, 1, 0, 1, 1], [0, 0, 0, 0, 0]]
    start = [0, 4]
    destination = [4, 4]
    print(hasPath(maze,start, destination))

path_maze()
