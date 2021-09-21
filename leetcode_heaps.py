from bisect import bisect, insort
from heapq import heappush, heappop, heappushpop

from typing import List


def fixed_list():
    class MedianFinder:

        def __init__(self):
            self.small_heap= []
            self.large_heap = []

        def addNum(self, num):
            heappush(self.small_heap, -heappushpop(self.large_heap, num))
            if len(self.large_heap) < len(self.small_heap):
                heappush(self.large_heap, -heappop(self.small_heap))

        def findMedian(self):
            if len(self.large_heap) > len(self.small_heap):
                return float(self.large_heap[0])
            return (self.large_heap[0] - self.small_heap[0]) / 2.0


    m = MedianFinder()
    c = [[1], [2], [], [3], []]

    for a in c:
        if len(a)>=1:
            m.addNum(a[0])
        else:
            print(m.findMedian())

def sliding_window():
    def medianSlidingWindow(nums: List[int], k: int) -> List[float]:
        medians, window = [], []

        for i in range(len(nums)):

            # Find position where outgoing element should be removed from
            if i >= k:
                # window.remove(nums[i-k])        # this works too
                window.pop(bisect(window, nums[i - k]) - 1)

            # Maintain the sorted invariant while inserting incoming element
            insort(window, nums[i])

            # Find the medians
            if i >= k - 1:
                medians.append(float((window[int(k / 2)]
                                      if k & 1 > 0
                                      else (window[int(k / 2) - 1] + window[int(k / 2)]) * 0.5)))

        return medians

    def move(h1, h2):
        x, i = heappop(h1)
        heappush(h2, (-x, i))

    def get_med(h1, h2, k):
        return h2[0][0] * 1. if k & 1 else (h2[0][0] - h1[0][0]) / 2.

    nums = [1,3,-1,-3,5,3,6,7]
    print(medianSlidingWindow(nums,3))


def meeting_rooms():
    def minMeetingRooms( intervals: List[List[int]]) -> int:

        # If there is no meeting to schedule then no room needs to be allocated.
        if not intervals:
            return 0

        # The heap initialization
        free_rooms = []

        # Sort the meetings in increasing order of their start time.
        intervals.sort(key=lambda x: x[0])

        # Add the first meeting. We have to give a new room to the first meeting.
        heappush(free_rooms, intervals[0][1])

        # For all the remaining meeting rooms
        for i in intervals[1:]:

            # If the room due to free up the earliest is free, assign that room to this meeting.
            if free_rooms[0] <= i[0]:
                heappop(free_rooms)

            # If a new room is to be assigned, then also we add to the heap,
            # If an old room is allocated, then also we have to add to the heap with updated end time.
            heappush(free_rooms, i[1])

        # The size of the heap tells us the minimum rooms required for all the meetings.
        return len(free_rooms)

    intervals = [[0, 30], [5, 10], [15, 20]]
    print(minMeetingRooms(intervals))

