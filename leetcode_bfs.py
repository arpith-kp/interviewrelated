from collections import deque, defaultdict, Counter
from typing import List


def min_stop_flight():
    def findCheapestPrice( n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:

        tracker = deque()
        mapper = defaultdict(list)

        for flight in flights:
            src, dst, cost = flight
            mapper[src].append((dst, cost))

        tracker.append(mapper[src])
        max_val = 0
        temp = 0
        count_stop = k +1

        while len(tracker) > 0 and count_stop >=0:
            count_stop-=1
            temp+=mapper[src][0][1]
            src = mapper[src][0][0]
            if src == dst and count_stop==0:
                max_val=max(temp, max_val)
                break
            tracker.popleft()
            tracker.append(mapper[src])
        return max_val

    n = 3
    flights = [[0,1,100],[1,2,100],[0,2,500]]
    src = 0
    dst = 2
    k = 0
    print(findCheapestPrice(n,flights,src,dst,k))

def letter_change():
    def minMutation( start: str, end: str, bank: List[str]) -> int:
        diff = set(start) - set(end)
        lookup = Counter(diff)
        res = []
        reached_end = 0

        while len(lookup)>=0 and reached_end<len(bank):
            word = bank[reached_end]
            char = set(start) - set(word)



min_stop_flight()
