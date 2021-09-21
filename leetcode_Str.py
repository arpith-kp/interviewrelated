import bisect
import collections
from typing import List


def partition_label(S):
    """
    https://leetcode.com/problems/partition-labels/
    Input: s = "ababcbacadefegdehijhklij"
Output: [9,7,8]
Explanation:
The partition is "ababcbaca", "defegde", "hijhklij".
    :param S:
    :return:
    """
    def partition_labels(S):

        rightmost = {c: i for i, c in enumerate(S)}
        left, right = 0, 0

        result = []
        for i, letter in enumerate(S):

            right = max(right, rightmost[letter])

            if i == right:
                result += [right - left + 1]
                left = i + 1

        return result

    print(partition_label(S))


def repeated_sub_array():
    def findLength(A: List[int], B: List[int]) -> int:
        n = len(A)
        m = len(B)
        dp = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
        max_val = 0
        for i in range(1, n+1):
            for j in range(1, m+1):
                if A[i-1] == B[j-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                    max_val = max(max_val, dp[i][j])
        return max_val
    nums1 = [0,0,0,0,0]
    nums2 = [0,0,0,0,0]
    print(findLength(nums1, nums2))

def autosuggest():
    """
    Time complexity : O(nlog(n)) + O(mlog(n))Where n is the length of products and m is the length of the search word. Here we treat string comparison in sorting as O(1)O(1). O(nlog(n))O(nlog(n)) comes from the sorting and O(mlog(n))O(mlog(n)) comes from running binary search on products m times.
    :return:
    """
    def suggestedProducts(products: List[str], searchWord: str) -> List[List[str]]:
        products.sort()
        cur, ans = '', []
        for char in searchWord:
            cur += char
            i = bisect.bisect_left(products, cur)
            ans.append([product for product in products[i : i + 3] if product.startswith(cur)])
        return ans

    products = ["chill", "mobile", "mouse", "moneypot", "monitor", "mousepad"]
    searchWord = "mouse"
    print(suggestedProducts(products, searchWord))

def pairs():
    def findPairs( nums: List[int], k: int) -> int:
        nums = sorted(nums)
        res = []
        seen = set()
        for idx, i in enumerate(nums):
            if i in seen: continue
            seen.add(i)
            if i+k in nums[idx+1:]:
                res.append((i,i+k))
        return len(res)

    nums = [1,3,1,5,4]
    k = 0
    print(findPairs(nums,k))

def divisible_subset():
    def largestDivisibleSubset(nums: List[int]) -> List[int]:
        sets = set()
        for i in range(2, len(nums)+1):
            if nums[i-2] % nums[i-1] == 0 or nums[i-1] % nums[i-2] == 0:
                sets.add(nums[i-2])
                sets.add(nums[i-1])
        return list(sets)

    nums = [1,2,3]
    print(largestDivisibleSubset(nums))

def minWindowSub():
    def min_window( s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        # Struggled with this problem for a long while.
        # Idea: Two pointers: moving end forward to find a valid window,
        #                     moving start forward to find a smaller window
        #                     counter and hash_map to determine if the window is valid or not

        # Count the frequencies for chars in t
        hash_map = dict()
        for c in t:
            if c in hash_map:
                hash_map[c] += 1
            else:
                hash_map[c] = 1

        start, end = 0, 0

        # If the minimal length doesn't change, it means there's no valid window
        min_window_length = len(s) + 1

        # Start point of the minimal window
        min_window_start = 0

        # Works as a counter of how many chars still need to be included in a window
        num_of_chars_to_be_included = len(t)

        while end < len(s):
            # If the current char is desired
            if s[end] in hash_map:
                # Then we decreased the counter, if this char is a "must-have" now, in a sense of critical value
                if hash_map[s[end]] > 0:
                    num_of_chars_to_be_included -= 1
                # And we decrease the hash_map value
                hash_map[s[end]] -= 1

            # If the current window has all the desired chars
            while num_of_chars_to_be_included == 0:
                # See if this window is smaller
                if end - start + 1 < min_window_length:
                    min_window_length = end - start + 1
                    min_window_start = start

                # if s[start] is desired, we need to update the hash_map value and the counter
                if s[start] in hash_map:
                    hash_map[s[start]] += 1
                    # Still, update the counter only if the current char is "critical"
                    if hash_map[s[start]] > 0:
                        num_of_chars_to_be_included += 1

                # Move start forward to find a smaller window
                start += 1

            # Move end forward to find another valid window
            end += 1

        if min_window_length == len(s) + 1:
            return ""
        else:
            return s[min_window_start:min_window_start + min_window_length]

    s = "ADOBECODEBANC"
    t = "ABC"
    print(min_window(s,t))

def word_transformation():
    def ladderLength( beginWord, endWord, wordList):
        wordList = set(wordList)
        queue = collections.deque([[beginWord, 1]])
        while queue:
            word, length = queue.popleft()
            if word == endWord:
                return length
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    next_word = word[:i] + c + word[i + 1:]
                    if next_word in wordList:
                        wordList.remove(next_word)
                        queue.append([next_word, length + 1])
        return 0

    beginWord = "hit"
    endWord = "cog"
    wordList = ["hot", "dot", "dog", "lot", "log", "cog"]
    print(ladderLength(beginWord, endWord, wordList))

def substring_k_distinct():
    def lengthOfLongestSubstringKDistinct(s, k):
        d = collections.defaultdict(int)
        i, res = 0, 0
        for j in range(len(s)):
            d[s[j]] += 1
            # while have more than k, we keep removing from i
            while len(d) > k:
                d[s[i]] -= 1
                if d[s[i]] == 0:
                    del d[s[i]]
                i += 1
            res = max(res, j - i + 1)
        return res

    print(lengthOfLongestSubstringKDistinct('abccce', 3))

def distinct_dp():
    def numDistinct(s: str, t: str) -> int:
        len_s, len_t = len(s) + 1, len(t) + 1
        dp = [[0] * len_t for _ in range(len_s)]

        for si in range(len_s):
            dp[si][0] = 1

        for si in range(1, len_s):
            for ti in range(1, len_t):
                dp[si][ti] += dp[si - 1][ti]  # Don't use s[si - 1]
                dp[si][ti] += dp[si - 1][ti - 1] if s[si - 1] == t[ti - 1] else 0  # Do use s[si-1]

        return dp[-1][-1]

    print(numDistinct('abdecaeb', 'ab'))

def wordbreak():
    def wordBreak( s, words):
        ok = [True]
        for i in range(1, len(s) + 1):
            ok += any(ok[j] and s[j:i] in words for j in range(i)),
        return ok[-1]

    s = "leetcodes"
    wordDict = ["leet", "code"]
    wordBreak(s, wordDict)

wordbreak()
