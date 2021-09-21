from collections import defaultdict
from queue import PriorityQueue
from typing import Optional, List
from collections import Counter
import heapq

def frequent():
    """
    to iterate backwards insert items with -1
    :return:
    """
    def topKfrequent(input: Optional[List], top):
        c = Counter(input)
        print(c.most_common(top))
        count = defaultdict(int)
        for i in input:
            count[i] = count[i] +1
        for k, v in count.items():
            pq.put((v*-1, k))
        res = []
        for i in range(top):
            n = pq.get()[0]
            res.append(n*-1)
        return res
    pq = PriorityQueue()
    print(topKfrequent([1,2,3,4,4,3,3,6], 2))


def findStreamMedian():
    """
    Maintain two heaps max and min, at insertion insert to max first, pop from it and reinsert to min to maintain
    same number of elements in both when possible . At any point max should hold more for easy avg, cos we  just
    get number from max if unequal. https://youtu.be/Rr3vJNEZMzk?list=PLujIAthk_iiO7r03Rl4pUnjFpdHjdjDwy&t=1085
    :return:
    """

    def addNum(num):
        heapq.heappush(max_heap_list, num)
        n = heapq.heappop(max_heap_list)
        heapq.heappush(min_heap_list, -n)
        if len(max_heap_list) < len(min_heap_list):
            heapq.heappush(max_heap_list, -1 * heapq.heappop(min_heap_list))
        print(f" Number is {num}: max heap {max_heap_list} , min heap {min_heap_list}")

    def findrunningmedian(input):
        addNum(input)
        if len(max_heap_list) == len(min_heap_list):
            high = max_heap_list[0]
            low = -1 * min_heap_list[0]
            print(f" High {high} and Low is {low} Final: {float(( high + low) / 2.0)}")
            return float(( high + low) / 2.0)
        else:
            s = max_heap_list[0]
            print(f" From Max {s}")
            return s

    def numgen():
        for i in range(1, 7):
            findrunningmedian(i)

    max_heap_list = []
    min_heap_list = []
    numgen()

def rearrage():
    def reorganizeString(S):
        res, c = [], Counter(S)
        pq = [(-value, key) for key, value in c.items()]
        heapq.heapify(pq)
        p_a, p_b = 0, ''
        while pq:
            a, b = heapq.heappop(pq)
            res += [b]
            if p_a < 0:
                heapq.heappush(pq, (p_a, p_b))
            a += 1
            p_a, p_b = a, b
        res = ''.join(res)
        if len(res) != len(S): return ""
        return res

    print(reorganizeString("aabcdeendes"))

def max_events():
    def maxEvents( events):
        events = sorted(events)
        total_days = max(event[1] for event in events)
        min_heap = []
        day, cnt, event_id = 1, 0, 0
        while day <= total_days:
            # if no events are available to attend today, let time flies to the next available event.
            if event_id < len(events) and not min_heap:
                day = events[event_id][0]

            # all events starting from today are newly available. add them to the heap.
            while event_id < len(events) and events[event_id][0] <= day:
                heapq.heappush(min_heap, events[event_id][1])
                event_id += 1

            # if the event at heap top already ended, then discard it.
            while min_heap and min_heap[0] < day:
                heapq.heappop(min_heap)

            # attend the event that will end the earliest
            if min_heap:
                heapq.heappop(min_heap)
                cnt += 1
            elif event_id >= len(events):
                break  # no more events to attend. so stop early to save time.
            day += 1

        return cnt
    d =[[1,4],[4,4],[2,2],[3,4],[1,1]]
    maxEvents(d)

max_events()
