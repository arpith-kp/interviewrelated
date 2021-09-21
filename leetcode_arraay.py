import collections
from collections import Counter
from typing import List


def substring():

    def lengthOfLongestSubstring(s: str) -> int:
        start = maxLength = 0
        usedChar = {}

        for i in range(len(s)):
            if s[i] in usedChar and start <= usedChar[s[i]]:
                start = usedChar[s[i]] + 1
            else:
                maxLength = max(maxLength, i - start + 1)

            usedChar[s[i]] = i

        return maxLength

    print(lengthOfLongestSubstring('aab'))

def next_perm():
    def nextPermutation(nums: List[int]) -> None:
        i = j = len(nums) - 1
        while i > 0 and nums[i - 1] >= nums[i]:
            i -= 1
        if i == 0:  # nums are in descending order
            nums.reverse()
            return
        k = i - 1  # find the last "ascending" position
        while nums[j] <= nums[k]:
            j -= 1
        nums[k], nums[j] = nums[j], nums[k]
        l, r = k + 1, len(nums) - 1  # reverse the second part
        while l < r:
            nums[l], nums[r] = nums[r], nums[l]
            l += 1
            r -= 1

    nums = [1, 2, 3]
    nextPermutation(nums)

def min_window_sub():
    def minwindow(s,t):
        need, missing = collections.Counter(t), len(t)
        i = I = J = 0
        for j, c in enumerate(s, 1):
            missing -= need[c] > 0
            need[c] -= 1
            if not missing:
                while i < j and need[s[i]] < 0:
                    need[s[i]] += 1
                    i += 1
                if not J or j - i <= J - I:
                    I, J = i, j
        return s[I:J]


    def minWindow2(s,t):
        start =0
        end = 0
        cnt = collections.Counter(t)
        ln_t = float('infinity')
        head = 0
        counter = len(cnt)
        while end < len(s):

            c = s[end]
            if c in cnt:
                cnt[c] = cnt[c] -1
                if cnt[c] == 0:
                    counter-=1
            end+=1

            while counter ==0:
                d = s[start]
                if d in cnt:
                    cnt[c] = cnt[c] + 1
                    if cnt[c] > 0:
                        counter += 1
                if end -start < ln_t:
                    ln_t = end - start
                    head = start
                start+=1
        return s[head:head+ln_t]

    s = "DEFABC"
    t = "ABC"
    minWindow2(s, t)

def lenght_of_sub():
    def lengthOfLongestSubstringTwoDistinct(s: 'str') -> 'int':
        n = len(s)
        if n < 3:
            return n

        # sliding window left and right pointers
        left, right = 0, 0
        # hashmap character -> its rightmost position
        # in the sliding window
        hashmap = collections.defaultdict()

        max_len = 2

        while right < n:
            # when the slidewindow contains less than 3 characters
            hashmap[s[right]] = right
            right += 1

            # slidewindow contains 3 characters
            if len(hashmap) == 3:
                # delete the leftmost character
                del_idx = min(hashmap.values())
                del hashmap[s[del_idx]]
                # move left pointer of the slidewindow
                left = del_idx + 1

            max_len = max(max_len, right - left)

        return max_len

    print(lengthOfLongestSubstringTwoDistinct('aabc'))


def backspace():

    def backspace(s,t):
        def build(S):
            ans = []
            for c in S:
                if c != '#':
                    ans.append(c)
                elif ans:
                    ans.pop()
            return "".join(ans)
        return build(s) == build(t)

    backspace('ab##', 'c#d#')

def max_product_array():
    def maxProduct(nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max_prod, min_prod, ans = nums[0], nums[0], nums[0]
        for i in range(1, len(nums)):
            x = max(nums[i], max_prod * nums[i], min_prod * nums[i])
            y = min(nums[i], max_prod * nums[i], min_prod * nums[i])
            max_prod, min_prod = x, y
            ans = max(max_prod, ans)
        return ans

    nums = [2, 3, -2, 4]
    print(maxProduct(nums))
