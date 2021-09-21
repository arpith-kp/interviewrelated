import collections
import heapq
import itertools
from typing import List


class Twitter(object):

    def __init__(self):
        self.timer = itertools.count(step=-1)
        self.tweets = collections.defaultdict(collections.deque)
        self.followees = collections.defaultdict(set)

    def postTweet(self, userId, tweetId):
        self.tweets[userId].appendleft((next(self.timer), tweetId))

    def getNewsFeed(self, userId):
        tweets = heapq.merge(*(self.tweets[u] for u in self.followees[userId] | {userId}))
        return [t for _, t in itertools.islice(tweets, 10)]

    def follow(self, followerId, followeeId):
        self.followees[followerId].add(followeeId)

    def unfollow(self, followerId, followeeId):
        self.followees[followerId].discard(followeeId)


class Node:
    def __init__(self, userId, tweetId):
        self.userId = userId
        self.tweetId = tweetId
        self.next = None


class Twitter2:

    def __init__(self):
        self.head = None
        self.relations = set()

    def postTweet(self, userId: int, tweetId: int) -> None:
        tweet = Node(userId, tweetId)
        if self.head == None:
            self.head = tweet
        else:
            tweet.next = self.head
            self.head = tweet

    def getNewsFeed(self, userId: int) -> List[int]:
        curr = self.head
        count = 0
        feed = []
        while curr != None and count < 10:
            if curr.userId == userId:
                feed.append(curr.tweetId)
                count += 1
            elif (userId, curr.userId) in self.relations:
                feed.append(curr.tweetId)
                count += 1
            curr = curr.next

        return feed

    def follow(self, followerId: int, followeeId: int) -> None:
        self.relations.add((followerId, followeeId))

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if (followerId, followeeId) in self.relations:
            self.relations.remove((followerId, followeeId))



t = Twitter2()
t.postTweet(1,5)
print(t.getNewsFeed(1))
t.follow(1,2)
t.postTweet(2,6)
print(t.getNewsFeed(1))
t.unfollow(1,2)
