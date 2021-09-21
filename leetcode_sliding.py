def sliding_without_duplicate():
    def sliding(S, K):
        res, i = 0, 0
        cur = set()
        for j in range(len(S)):
            while S[j] in cur:
                cur.remove(S[i])
                i += 1
            cur.add(S[j])
            res += j - i + 1 >= K
        return res

    s = "havefunonleetcode"
    k = 5
    print(sliding(s, k))


sliding_without_duplicate()
