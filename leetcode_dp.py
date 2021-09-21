import json
from collections import OrderedDict, defaultdict
from typing import List


def climbingstairs(n):
    def sub_solution(n, count, cache=OrderedDict()):

        # print(f"Check for solution at {n} and {count}")
        if n in cache:
            return cache[n]
        if n == 0:
            count = count + 1
            cache[n] = count
            return count
        if n < 0:
            return 0

        return sub_solution(n - 1, count) + sub_solution(n - 2, count)

    print("old", sub_solution(n, 0))


def coinchange():
    def sub_solution(n, count, cache):

        if n == 0:
            count = count + 1
            # print(f" N: {n} count: {count}")
            return count
        if n < 0:
            return 0
        if n == 2:
            x = 1
        a = sub_solution(n - 1, count, cache)
        if cache[n - 1] != 0:
            if a < cache[n - 1]:
                cache[n - 1] = a
        else:
            cache[n - 1] = a
        print(f" n: {n}  count: {a}")
        b = sub_solution(n - 2, count, cache)
        if cache[n - 2] != 0:
            if b < cache[n - 2]:
                cache[n - 2] = b
        else:
            cache[n - 2] = b
        print(f" n: {n}  count: {b}")
        c = sub_solution(n - 5, count, cache)
        print(f" n: {n}  count: {c}")

        if cache[n - 5] != 0:
            if c < cache[n - 5]:
                cache[n - 5] = c
        else:
            cache[n - 5] = c
        return a + b + c

    c = defaultdict(int)
    print(sub_solution(6, 0, c))
    print("-------", json.dumps(c))


def wordbreak():
    """
    first: i
           i c
           i c e
           i c e c
           i c e c r
    :return:
    """

    def wordBreak(s, words):
        ok = [False] * (len(s) + 1)
        ok[0] = True
        for i in range(1, len(s) + 1):
            for j in range(i):
                if ok[j] and s[j:i] in words:
                    ok[i] = True
        print("DDDDDDDD ", ok)
        return ok[len(s)]

    s = "icecream"
    wordDict = ["ice", "cream"]

    return wordBreak(s, wordDict)


def lenghtCommonSubsequence():
    def longestCommonSubsequence(s1: str, s2: str) -> int:
        m = len(s1)
        n = len(s2)
        memo = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        for row in range(1, m + 1):
            for col in range(1, n + 1):
                if s1[row - 1] == s2[col - 1]:
                    memo[row][col] = 1 + memo[row - 1][col - 1]
                else:
                    memo[row][col] = max(memo[row][col - 1], memo[row - 1][col])

        return memo[m][n]

    def min_edit_distance(s1: str, s2: str) -> int:
        m = len(s1)
        n = len(s2)
        memo = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        for row in range(1, m + 1):
            for col in range(1, n + 1):
                if s1[row - 1] == s2[col - 1]:
                    memo[row][col] = 1 + memo[row - 1][col - 1]
                else:
                    memo[row][col] = max(memo[row][col - 1], memo[row - 1][col])

        return (m - memo[m][n]) + (n - memo[m][n])


def regularMatch():
    """
     s=.b p=ab
     chec if first char is ., if true

     1. s[i] == s[j] or s[i] == . thse both leads to .
     2. s[i] == * , p[j+1] == s[i-1] if no matches or one match
    :return:
    """

    def matchs(s, p):
        if len(s) == 0:
            return len(p) == 0
        else:
            first_match = (s[0] == p[0] or p[0] == '.')
            if len(p) >= 2 and p[1] == "*":
                return matchs(s, p[2:]) or (first_match and matchs(s[1:], p))
            else:
                if first_match:
                    return matchs(s[1:], p[1:])
                else:
                    return False
    def isMatch(s, p):
        # The DP table and the string s and p use the same indexes i and j, but
        # table[i][j] means the match status between p[:i] and s[:j], i.e.
        # table[0][0] means the match status of two empty strings, and
        # table[1][1] means the match status of p[0] and s[0]. Therefore, when
        # refering to the i-th and the j-th characters of p and s for updating
        # table[i][j], we use p[i - 1] and s[j - 1].

        # Initialize the table with False. The first row is satisfied.
        table = [[False] * (len(s) + 1) for _ in range(len(p) + 1)]

        # Update the corner case of matching two empty strings.
        table[0][0] = True

        # Update the corner case of when s is an empty string but p is not.
        # Since each '*' can eliminate the charter before it, the table is
        # vertically updated by the one before previous. [test_symbol_0]
        for i in range(2, len(p) + 1):
            table[i][0] = table[i - 2][0] and p[i - 1] == '*'

        for i in range(1, len(p) + 1):
            for j in range(1, len(s) + 1):
                if p[i - 1] != "*":
                    # Update the table by referring the diagonal element.
                    table[i][j] = table[i - 1][j - 1] and \
                                  (p[i - 1] == s[j - 1] or p[i - 1] == '.')
                else:
                    # Eliminations (referring to the vertical element)
                    # Either refer to the one before previous or the previous.
                    # I.e. * eliminate the previous or count the previous.
                    # [test_symbol_1]
                    table[i][j] = table[i - 2][j] or table[i - 1][j]

                    # Propagations (referring to the horizontal element)
                    # If p's previous one is equal to the current s, with
                    # helps of *, the status can be propagated from the left.
                    # [test_symbol_2]
                    if p[i - 2] == s[j - 1] or p[i - 2] == '.':
                        table[i][j] |= table[i][j - 1]

        return table[-1][-1]

def maxProfitStock():
    """

    :return:
    """

    def maxp(prices, fee):
        sell = 0
        buy = -prices[0]

        for i in range(1, len(prices)):
            sell = max(sell, buy + prices[i] - fee)
            buy = max(buy, sell - prices[i])
        return sell


def climbingstep():
    def climb(target):
        dp = [0] * (target + 1)

        dp[1] = 1
        dp[0] = 1

        for i in range(2, target + 1):
            print(f" Storing at {i} results from {dp[i - 1]} {dp[i - 2]}")
            dp[i] = dp[i - 1] + dp[i - 2]

        return dp[target]

    return climb(5)


def uniquePaths():
    """
    final position is always one, handle edge case if i. Draw table of 3,2 on book and write solution
    :return:
    """

    def robot(m, n):
        dp = [[0] * n for _ in range(m)]
        dp[m - 1][n - 1] = 1
        for i in range(m - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                if i != 0:
                    dp[i - 1][j] = dp[i][j] + dp[i - 1][j]
                if j != 0:
                    dp[i][j - 1] = dp[i][j] + dp[i][j - 1]
                print(f" Vertical {(i - 1, j)} -> {dp[i - 1][j]} Horizontal {(i, j - 1)} -> {dp[i][j - 1]} ")
                # a=1
        print("Final result is ", dp[0][0])
        return dp[0][0]

    return robot(3, 7)


def numberDecoding():
    def decode(p, s, leve=1):
        n = len(s)

        if n == p:
            return 1
        if s[p] == '0':
            return 0
        print(f" " * leve, f" Str {s} , char at {s[p]} ")
        res = decode(p + 1, s, leve * 2)
        if p < n - 1 and (s[p] == '1' or s[p] == 2 and s[p + 1] < '7'):
            res += decode(p + 2, s, leve * 2)
        print(f"res: {res}")
        return res

    print(decode(0, '12'))


def subsequence():
    def isSubsequence(s, t):
        t = iter(t)
        return all(c in t for c in s)

    def countVowelStrings(n):
        dp = [0] + [1] * 5
        for i in range(1, n + 1):
            for k in range(1, 6):
                dp[k] += dp[k - 1]
        return dp[5]

    def ss(s, t, i=0, j=0, res=[], level=1):
        if not s:
            return False

        if i >= len(s) or j >= len(t):
            return ''.join(res) == s

        print(f" " * level, f" char at {i}, {j} = {s[i]} , {t[j]} ")
        if s[i] == t[j]:
            res.append(s[i])
            i = i + 1
            j = j + 1
            if not ss(s, t, i, j, res, level * 2):
                return False
            else:
                return True

        if s[i] != t[j]:
            j = j + 1
            if not ss(s, t, i, j, res, level * 2):
                return False
            else:
                return True

        return True

    s = "abc"
    t = "ahbgdc"
    s = "b"
    t = "abc"
    return ss(s, t)


def longestIncreasingSubsequence():
    """
    Draw diagram;
        1. Visualize LIS = longest path in dag + 1
        2. Find subproblem LIS[k] = LIS ending at k
        3. Find relationship among subproblem: to solve problem at LIS[4] = 1 + max{LIS[]...}
        4. Generalize LIS[5] = 1 + max{LIS[k] | k< 5 and A[k] < a[5]} . To satisfy condition any number at current position must be less than last index
        5. Code this
    https://youtu.be/aPQY__2H3tE?t=470
    :return:
    """

    def lengthOfLIS(nums: List[int]) -> int:
        if not nums:
            return 0

        n = len(nums)
        dp = [1] * n

        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], 1 + dp[j])

        return max(dp)

    return lengthOfLIS([3, 1, 8, 2, 5])


def subset_sum():
    def sb(input: List[int]):
        if sum(input) % 2:
            return False

        dp = set()
        dp.add(0)
        target = sum(input) // 2
        for i in range(len(input) - 1, -1, -1):
            nextDp = set()
            for t in dp:
                nextDp.add(t + nextDp)
                nextDp.add(t)
            dp = nextDp
        return True if target in dp else False


def countTeams():
    def teams(nums: List[int]):
        level = 1
        n = len(nums)
        up = [0] * n
        down = [0] * n

        teams = 0
        for i in range(n - 1, -1, -1):
            level = level * 2
            for j in range(i + 1, n):
                print(" " * level, f" I is: {i} J: is {j} Compare {nums[i]}, {nums[j]} total is {teams}")
                if nums[i] > nums[j]:
                    up[i] += 1
                    teams += up[j]
                elif nums[i] < nums[j]:
                    down[i] += 1
                    teams += down[j]

        return teams

    print(teams([2, 5, 3, 4, 1]))


def palindrome_substring():
    """

    :return:
    """

    def pal(s: str):
        count = 0
        result = []
        level = 2
        for idx, i in enumerate(s):
            print(f" " * level, f" Index {idx}")
            l = idx
            r = idx

            print(f" " * level, f" Odd result {result}")
            # print(" "* level, f" left is {l} right is {r}")
            while l >= 0 and r < len(s) and s[l] == s[r]:
                # print(" " * level, f" lc: {s[l]} rc: {s[r]}, count {count}")
                result.append(s[l])
                l = l - 1
                r = r + 1
                count += 1
                print(" " * level, f" odd leng res {result}")

            l = idx
            r = idx + 1
            while l >= 0 and r < len(s) and s[l] == s[r]:
                print(" " * level, f" left is {l} right is {r}", f" lc: {s[l]} rc: {s[r]}, count {count}",
                      f" even leng res {result}")
                result.append(s[l] + s[r])
                count += 1
                l = l - 1
                r = r + 1

            level *= 2
        print(f" Count is {count}", f" leng res {result}")

    pal('aaa')


def mincostticket():
    """
    # a = cost[one day pass] + cost of next day
# b = cost[ week pass ] + cost of next day after week
# c = cost[ month pass ] + cost of next day after month
    :return:
    """

    def mincostTickets(days: List[int], costs: List[int]) -> int:
        # index of ticket
        _1day_pass, _7day_pass, _30day_pass = 0, 1, 2

        # Predefined constant to represent not-traverling day
        NOT_Traveling_Day = -1

        # DP Table, record for minimum cost of ticket to travel
        dp_cost = [NOT_Traveling_Day for _ in range(366)]

        # base case:
        # no cost before travel
        dp_cost[0] = 0

        for day in days:
            # initialized to 0 for traverling days
            dp_cost[day] = 0

        # Solve min cost by Dynamic Programming
        for day_i in range(1, 366):

            if dp_cost[day_i] == NOT_Traveling_Day:

                # today is not traveling day
                # no extra cost
                dp_cost[day_i] = dp_cost[day_i - 1]


            else:

                # today is traveling day
                # compute optimal cost by DP

                dp_cost[day_i] = min(dp_cost[day_i - 1] + costs[_1day_pass],
                                     dp_cost[max(day_i - 7, 0)] + costs[_7day_pass],
                                     dp_cost[max(day_i - 30, 0)] + costs[_30day_pass])

        # Cost on last day of this year is the answer
        return dp_cost[365]

    days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 31]
    costs = [2, 7, 15]
    print(mincostTickets(days, costs))


def numTiles():
    res = set()

    def check_valid(chars: str):
        return not chars in res

    def numTilePossibilities2(tiles: str, idx=0, level=1, temp_res=""):

        # if len(tiles) == len(temp_res):
        #     return True
        print(" " * level, f" path is {temp_res}  tiles {tiles}")
        for idx, ch in enumerate(tiles):
            temp_res += ch
            if check_valid(temp_res):
                res.add(temp_res)
                # print(" "* level, f" found valid subsequence {res}, index {idx}")
                numTilePossibilities2(tiles[:idx] + tiles[idx + 1:], idx + 1, level * 2, temp_res)
            temp_res = temp_res[:-1]
        return len(res)

    def numTilePossibilities(tiles, level=1):
        """
        :type tiles: str
        :rtype: int
        """
        res = set()

        def dfs(path, t, level=1):
            print(" " * level, f" path is {path} tiles is {t}")
            if path:
                res.add(path)
                print(" " * level, f" found valid subsequence {res}")
            for i in range(len(t)):
                dfs(path + t[i], t[:i] + t[i + 1:], level * 2)

        dfs('', tiles)
        return len(res)

    # numTilePossibilities("AAABBC")
    print('------------------------------------------------------')
    print("----", numTilePossibilities2("AAABBC"))
    print(f"REsult is {len(res)}")


"""
Notes:

Whenever you want to get final result in number use tabulation.


Longest increasing subsequence:

"""


class Solution:
    res = set()

    def numTilePossibilities(self, tiles: str, level=1, temp_res="") -> int:
        def ts(tiles: str, level=1, temp_res=""):
            for idx, ch in enumerate(tiles):
                temp_res += ch
                if not temp_res in self.res:
                    self.res.add(temp_res)
                    # print(" "* level, f" found valid subsequence {res}, index {idx}")
                    ts(tiles[:idx] + tiles[idx + 1:], level * 2, temp_res)
                temp_res = temp_res[:-1]

        ts(tiles)
        return len(self.res)


def climbing_stairs(y):
    def climb(y: int):

        dp = [0] * (y + 1)
        dp[0] = 1
        dp[1] = 1

        for i in range(2, len(dp)):
            dp[i] = dp[i - 2] + dp[i - 1]

        return dp[y]

    def climb_cost(cost: List[int]):
        x = len(cost)
        dp = [0] * (x)
        dp[0] = cost[0]
        dp[1] = cost[1]

        for i in range(2, x):
            dp[i] = cost[i] + min(dp[i - 2], dp[i - 1])

        # when you reach last step consider either -1 or -2 postion
        return min(dp[x - 1], dp[x - 2])

    print(climb_cost(y))


def min_path_sum():
    def minPathSum(grid: List[List[int]]) -> int:

        row = len(grid)
        col = len(grid[0])
        dp = [[0] * col for _ in range(row)]
        dp[0][0] = grid[0][0]
        for i in range(1, row):
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        for j in range(1, col):
            dp[0][j] = dp[0][j - 1] + grid[0][j]

        for i in range(1, row):
            for j in range(1, col):
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]

        return dp[row - 1][col - 1]

    print(minPathSum(grid=[[1, 3, 1], [1, 5, 1], [4, 2, 1]]))


def generate_parenthesis(n):
    def backtrack(res, open, close, max_len, final_res: List):
        if len(res) == max_len * 2:
            final_res.append(res)
            return

        if open < max_len:
            backtrack(res + "(", open + 1, close, max_len, final_res)
        if close < open:
            backtrack(res + ')', open, close + 1, max_len, final_res)

    def generateParenthesis(n, res=""):
        final_res = []
        backtrack(res, 0, 0, n, final_res=final_res)
        return final_res

    print(generateParenthesis(n))


def count_submatrix():
    def numSubmat(mat: List[List[int]]):

        row = len(mat)
        col = len(mat[0])
        count = 0
        res = []
        for i in range(row):
            for j in range(col):

                if mat[i][j] == 1:
                    res.append(1)
                if mat[i][j] == 0:
                    print(f" [{i}] [{j}]  {res}")
                    if len(res) > 1:
                        count += 1
                        count = count + len(res)
                        print(res)
                        res = []
        res = []
        for i in range(row):
            for j in range(col):

                if mat[j][i] == 1:
                    res.append(1)
                if mat[j][i] == 0:
                    print(f" [{i}] [{j}] , {res}")
                    if len(res) > 1:
                        count += 1
                        count = count + len(res)
                        print(" REvers ", res)
                    res = []
        return count

    mat = [[1, 0, 1],
           [1, 1, 0],
           [1, 1, 0]]
    print(numSubmat(mat))


def subarray_with_Same_diff():
    """kandanes """

    def numberOfArithmeticSlices(nums: List[int]) -> int:
        curr = 0
        sum = 0
        for i in range(2, len(nums)):
            if nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]:
                curr += 1
                sum += curr
            else:
                curr = 0
        return sum

    def maxScore(cardPoints: List[int], k: int) -> int:
        size = len(cardPoints) - k
        minSubArraySum = float('inf')
        j = curr = 0

        for i, v in enumerate(cardPoints):
            curr += v
            if i - j + 1 > size:
                curr -= cardPoints[j]
                j += 1
            if i - j + 1 == size:
                minSubArraySum = min(minSubArraySum, curr)

        return sum(cardPoints) - minSubArraySum

    cardPoints = [1, 2, 3, 4, 5, 6, 1]
    k = 3
    print(maxScore(cardPoints, k))

    nums = [3, -1, -5, -9]
    # nums = [1,2,3,8,9,10]
    print(numberOfArithmeticSlices(nums))


def string_difference():
    """You are given an array of words where each word consists of lowercase English letters.

wordA is a predecessor of wordB if and only if we can insert exactly one letter anywhere in wordA without changing the order of the other characters to make it equal to wordB.

For example, "abc" is a predecessor of "abac", while "cba" is not a predecessor of "bcad"."""

    def longestStrChain(words: List[str]) -> int:
        d = dict()
        for word in words:
            d[word] = 1
        longest = 1
        for word in sorted(words, key=len):
            for i in range(len(word)):
                prev = word[:i] + word[i + 1:]
                if prev in d:
                    d[word] = max(d[word], d[prev] + 1)
            longest = max(longest, d[word])
        return longest

    def longestStrChain2(words: List[str]) -> int:
        res = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                word_a = words[i]
                word_b = words[j]
                if len(word_b) > len(word_a):
                    diff = "".join(set(word_b) - set(word_a))
                    if len(diff) == 1:
                        for k in range(len(word_a) + 1):
                            if word_a[:k] + diff + word_a[k:] == word_b and word_a == ''.join(set(word_b) - set(diff)):
                                res.append(word_a)
                                res.append(word_b)
        print("Res ", res)
        return len(res)

    words = ["a", "b", "ba", "bca", "bda", "bdca"]
    print(longestStrChain(words))


def max_abs_sum():
    def maxAbsoluteSum(nums: List[int]) -> int:
        max_so_far, pos_ending_here, neg_ending_here = 0, 0, 0
        for num in nums:
            pos_ending_here = max(0, pos_ending_here + num)
            neg_ending_here = min(0, neg_ending_here + num)
            max_so_far = max(max_so_far, pos_ending_here, -neg_ending_here)
        return max_so_far

    nums = [1, -3, 2, 3, -4]
    print(maxAbsoluteSum(nums))


def partition_equal():
    def canPartition(nums: List[int]) -> bool:

        total = sum(nums)
        if total % 2 == 1: return False

        target = total / 2  # target sum
        s = {0}  # stores the sums of the subsets

        for n in nums:
            sums_with_n = []  # stores the sums of the subsets that contain n
            for i in s:
                if i + n == target:
                    return True
                if i + n < target:
                    sums_with_n.append(i + n)
            s.update(sums_with_n)

        return False

    def canPartitionKSubsets(nums, k):
        sums = [0] * k
        subsum = sum(nums) / k
        nums.sort(reverse=True)
        l = len(nums)

        def walk(i):
            if i == l:
                return len(set(sums)) == 1
            for j in range(k):
                sums[j] += nums[i]
                if sums[j] <= subsum and walk(i + 1):
                    return True
                sums[j] -= nums[i]
                if sums[j] == 0:
                    break
            return False

        return walk(0)

    nums = [4, 3, 2, 3, 5, 2, 1]

    print(canPartitionKSubsets(nums, 4))


def lis_pattern_1():
    def lis():
        def lis(nums: List[int]) -> int:
            dp = [1] * len(nums)
            for i in range(1, len(nums)):
                for j in range(i):
                    if nums[i] > nums[j]:
                        dp[i] = max(1 + dp[j], dp[i])
            return max(dp)

        nums = [10, 9, 2, 5, 3, 7, 101, 18]
        print(lis(nums))

    def triplet():
        def increasingTriplet(nums: List[int]) -> bool:
            dp = [1] * len(nums)

            for i in range(2, len(nums)):
                for j in range(i):
                    for k in range(j):
                        if nums[i] > nums[j] > nums[k]:
                            dp[i] = max(1 + dp[k], dp[i], dp[j])
            return sum(dp) > len(nums)

        nums = [2, 1, 5, 0, 4, 6]
        print(increasingTriplet(nums))

    def divsubset():
        """
        Variant solution: DP with tracking index. https://youtu.be/Wv6DlL0Sawg
        :return:
        """

        def largestDivisibleSubset(nums: List[int]) -> List[int]:
            dp = [1] * len(nums)
            nums.sort()
            res = []
            for i in range(1, len(nums)):
                for j in range(i):
                    if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
                        dp[i] = max(1 + dp[j], dp[i])
            max_items = max(dp)

            for i in range(len(nums) - 1, -1, -1):
                if dp[i] == max_items and (len(res) == 0 or res[-1] % nums[i] == 0):
                    res.append(nums[i])
                    max_items -= 1
            return res

        nums = [1, 2, 3, 9, 18]
        print(largestDivisibleSubset(nums))

    def longchain():
        def findLongestChain(pairs: List[List[int]]) -> int:
            dp = [1] * len(pairs)
            pairs.sort()
            for i in range(1, len(pairs)):
                for j in range(i):
                    a, b = pairs[i]
                    c, d = pairs[j]
                    if d < a:
                        dp[i] = max(dp[j] + 1, dp[i])
            return max(dp)

        def findLongestChainFaster(pairs: List[List[int]]) -> int:
            pairs.sort()
            count = 0
            prev = float('-inf')
            for i, j in pairs:
                if i > prev:
                    count += 1
                    prev = j
            return count

        pairs = [[1, 2], [7, 8], [4, 5]]
        print(findLongestChainFaster(pairs))

    longchain()


def distinct_ways():
    def climb():
        def climb_stairs(n, ways):
            dp = [0] * (n + 1)
            dp[0] = 1
            dp[1] = 1

            for stair in range(2, n + 1):
                for step in range(ways + 1):
                    dp[stair] += dp[stair - step]

            return dp[n]

        def climb2(y: int):

            dp = [0] * (y + 1)
            dp[0] = 1
            dp[1] = 1

            for i in range(2, len(dp)):
                dp[i] = dp[i - 2] + dp[i - 1]

            return dp[y]

        print(climb_stairs(5, 2))
        print(climb2(5))

    def unique_way_grid(m, n):
        # m = len(grid)
        # n = len(grid[0])

        dp = [[1 for _ in range(n)] for _ in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

        return dp[m - 1][n - 1]

    print(unique_way_grid(1, 1))


def grid():
    def maximalSquare(matrix: List[List[int]]) -> int:
        """
           0  1
           _____
        0 |1 1|
        1 |1 1|
          -----
        value at (1,1) = min((0,1)(0,0)(1,0))+1 = 2 if all of neighbors are 1 then items of current position increase it by 1

           0  1
           _____
        0 |1 0|
        1 |1 1|
          -----
        value at (1,1) = min((0,1)(0,0)(1,0))+1 = 1 if any of neighbors are 0 then items of current position will remain as is.
        https://leetcode.com/problems/maximal-square/discuss/600149/Python-Thinking-Process-Diagrams-DP-Approach

        :return:
        """
        r = len(matrix)
        c = len(matrix[0])

        dp = [[0] * (c + 1) for _ in range(r + 1)]
        max_side = 0
        ans = 0
        for i in range(r):
            for j in range(c):
                if matrix[i][j] == 1:
                    dp[i + 1][j + 1] = int(min(dp[i][j], dp[i + 1][j], dp[i][j + 1])) + 1
                    ans += dp[i + 1][j + 1]
                    print(f" {i} , {j} -> ", dp[i+1][j+1], " MIn ", dp[i][j], dp[i + 1][j], dp[i][j + 1])

                    max_side = max(max_side, dp[i + 1][j + 1])
        return ans

    matrix = [
        [0, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 1]
    ]
    print(maximalSquare(matrix))


def palindrome_sub():
    def countSubstrings( s: str) -> int:

        dp = [[0] * len(s) for _ in range(len(s))]

        for idx in range(len(s)-1):
            dp[idx][idx] = 1
        res = 0

        for i in range(len(s)-1, -1, -1):
            for j in range(i, len(s)):
                if s[i]==s[j] and ((j-i+1)<3 or dp[i+1][j-1]):
                    dp[i][j] = 1
                    res+=dp[i][j]
        return res

    print(countSubstrings('aaa'))

def dp_rob():
    def rob(nums: List[int]) -> int:

        dp = [0] * len(nums)

        dp[0]= nums[0]
        dp[1]= nums[1]

        if len(nums) <=2:
            return max(nums)

        for i in range(2, len(nums)):
            dp[i] = max(dp[i-2]+nums[i], dp[i-1])

        return dp[len(nums)-1]

    nums = [2,7,9,3,1]
    print(rob(nums))

dp_rob()
