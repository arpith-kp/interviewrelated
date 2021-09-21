from collections import deque


class MovingAverage:

    def __init__(self, size: int):
        """
        Initialize your data structure here.
        """
        self.queue = deque()
        self.size = size

    def next(self, val: int) -> float:
        if len(self.queue)>self.size-1:
            self.queue.popleft()
        self.queue.append(val)
        return sum(self.queue) / (len(self.queue))

movingAverage = MovingAverage(1)
print(movingAverage.next(4))
print(movingAverage.next(0))

