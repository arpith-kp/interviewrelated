import concurrent.futures
import operator
import statistics
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache, cached_property
from itertools import *
from collections import *
from multiprocessing import Pool
from multiprocessing.dummy import current_process
from pprint import pprint
import bisect
from sys import getsizeof

import requests
import tldextract as tldextract
import dotenv

def zipOperator():
    a = [1, 2, 3]
    b = ['a', 'b', 'c']
    for i in zip(a, b):
        print(i)


def mapOp():
    a = [1, 2, 3]
    b = ['a', 'b', 'c']

    for i in map(len, zip(a, b)):
        print(i)


def sumOp():
    a = [1, 2, 3]
    b = [4, 5, 6]

    for i in map(sum, zip(a, b)):
        print(i)


def better_grouper(inputs=None, n=1):
    if inputs is None:
        inputs = [1, 2, 3,4]
    iters = [iter(inputs)] * n
    print(iters)
    for i in zip(*iters):
        print(i)
    # time -f "Memory used (kB): %M\nUser time (seconds): %U"



def add(num, value, lock):
    try:
        lock.acquire()
        print('before add{0}:num={1}'.format(value, num))
        for i in range(0, 2):
            num += value
            print('after add{0}:num={1}'.format(value, num))
            time.sleep(1)
    except Exception as err:
        raise err
    finally:
        lock.release()

def custom_iter():
    class PrintNumber:
        def __init__(self, max):
            self.max = max

        def __iter__(self):
            self.num = 0
            return self

        def __next__(self):
            if (self.num >= self.max):
                raise StopIteration
            self.num += 1
            return self.num

    print_num = PrintNumber(3)
    print_num_iter = iter(print_num)
    iters = [print_num_iter] * 2
    for i in zip_longest(*iters):
        print(i)


def zipLong():
    x = [1, 2, 3, 4, 5]
    y = ['a', 'b', 'c']
    for a in zip_longest(x, y):
        print(a)

def combination():
    bills = [20, 20, 10, 5]
    for a in combinations(bills, 2):
        print(a)

def permutaio():
    bills = [20, 20, 10, 5]
    for a in permutations(bills ,2):
        print(a)

def counterdemo():
    counter = count()
    for a in range(5):
        print(next(counter))


def acc():
    bills = [20, 20, 10, 5]
    for z in accumulate(bills, operator.sub):
        print(z)


def chainitr():
    print(list(chain.from_iterable([[1, 2, 3], [4, 5, 6]])))


def grpby():
    things = [("animal", "bear"), ("animal", "duck"), ("plant", "cactus"), ("vehicle", "speed boat"), ("vehicle", "school bus")]
    for key, group in groupby(things, lambda x: x[0]):
        for thing in group:
            print("A %s is a %s." % (thing[1], key))
        print("")


def collectinCounter():
    things = [("animal", "bear"), ("animal", "bear"), ("plant", "cactus"), ("vehicle", "speed boat"), ("vehicle", "school bus")]
    dic = Counter(things)
    for k, v in dic.items():
        print (f"Key repeated {k} times {v}")

    print("Most common" , dic.most_common(1))

def defaultDictOp():
    dd = defaultdict(list)
    things = [("animal", "bear"), ("animal", "bear"), ("plant", "cactus"), ("vehicle", "speed boat"), ("vehicle", "school bus")]
    for key, grp in groupby(things, lambda k:k[1]):
       dd[key].append(list(grp))

    print("Defualt dict ", dd)

def dqOp():
    names_list = "Mike John Mike Anna Mike John John Mike Mike Britney Smith Anna Smith".split()
    dq = deque()
    for names in names_list:
        dq.appendleft(names)
    print("Deque ", dq)

def namedTuple():
    stu = namedtuple('Student', 'name, age')
    s1 = stu('a', '12')
    print(s1.name)

def binarsearch():
    sorted_fruits = ['apple', 'banana', 'orange', 'plum']
    print(bisect.bisect_left(sorted_fruits, 'banana'))


def gtLtEx():

    @dataclass(order=True)
    class Person:
        name: str
        surname: str
    a, b =  Person('Bob', 'Williams'), Person('John', 'Doe')
    print (a<b)


def datclassEx():
    RANKS = '2 3 4 5 6 7 8 9 10 J Q K A'.split()
    SUITS = '♣ ♢ ♡ ♠'.split()

    """
    Comparison is done on first field (sort_index), to ignore it from init paas false and
    override str method not to include in repr

    https://realpython.com/python-data-classes/

    """
    @dataclass(order=True)
    class PlayingCard:
        sort_index: int = field(init=False, repr=False)
        rank: str
        suit: str

        def __post_init__(self):
            self.sort_index = (RANKS.index(self.rank) * len(SUITS)
                               + SUITS.index(self.suit))

        def __str__(self):
            return f'{self.suit}{self.rank}'

    queen_of_hearts = PlayingCard('Q', '♡')
    ace_of_spades = PlayingCard('A', '♠')

    print("Comparision ", ace_of_spades>queen_of_hearts)


def memoryUitl():
    @dataclass
    class RegularCardData:
        rank: str
        suit: str

    class RegularCard:
        def __init__(self, rank, suit):
            self.rank = rank
            self.suit = suit

    @dataclass
    class RegularCardDataSlot:
        __slots__ = ['rank', 'suit']
        rank: str
        suit: str

    print("Size Dataclass", getsizeof(RegularCardData))
    print("Size Regular", getsizeof(RegularCard))
    print("Size Slots ", getsizeof(RegularCardDataSlot))


def useCache():
    @lru_cache(maxsize=32)
    def fib(n):
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)

    print("cache" ,fib(5))


def pipDependency():
    class Task(object):
        def __init__(self, name, *depends):
            self.__name = name
            self.__depends = set(depends)

        @property
        def name(self):
            return self.__name

        @property
        def depends(self):
            return self.__depends

    # "Batches" are sets of tasks that can be run together
    def get_task_batches(nodes):

        # Build a map of node names to node instances
        name_to_instance = dict((n.name, n) for n in nodes)

        # Build a map of node names to dependency names
        name_to_deps = dict((n.name, set(n.depends)) for n in nodes)

        # This is where we'll store the batches
        batches = []

        # While there are dependencies to solve...
        while name_to_deps:

            # Get all nodes with no dependencies
            ready = {name for name, deps in name_to_deps.iteritems() if not deps}

            # If there aren't any, we have a loop in the graph
            if not ready:
                msg = "Circular dependencies found!\n"
                msg += format_dependencies(name_to_deps)
                raise ValueError(msg)

            # Remove them from the dependency graph
            for name in ready:
                del name_to_deps[name]
            for deps in name_to_deps.itervalues():
                deps.difference_update(ready)

            # Add the batch to the list
            batches.append({name_to_instance[name] for name in ready})

        # Return the list of batches
        return batches

    # Format a dependency graph for printing
    def format_dependencies(name_to_deps):
        msg = []
        for name, deps in name_to_deps.iteritems():
            for parent in deps:
                msg.append("%s -> %s" % (name, parent))
        return "\n".join(msg)

    # Create and format a dependency graph for printing
    def format_nodes(nodes):
        return format_dependencies(dict((n.name, n.depends) for n in nodes))

def window(seq=None, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "

    l = [1,2,3,4,5]
    if seq is None:
        seq = l
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def cacheprop():

    @dataclass
    class CacheProp(object):
        data: list

        @cached_property
        def std_variance(self):
            return statistics.stdev(self.data)

    l = [1.0, 2.0, 3.0, 4.0, 5.0]
    c= CacheProp(l)
    # print(c.std_variance())
    print("Standard Deviation of sample is % s "
          % (statistics.stdev(l)))


def lru():
    class LRUCache:
        def __init__(self, capacity):
            self.capacity = capacity
            self.cache = OrderedDict()

        def get(self, key):
            try:
                value = self.cache.pop(key)
                self.cache[key] = value
                return value
            except KeyError:
                return -1

        def set(self, key, value):
            try:
                self.cache.pop(key)
            except KeyError:
                if len(self.cache) >= self.capacity:
                    self.cache.popitem(last=False)
            self.cache[key] = value

def func(a, b):
    return a + b

def paralleprocess():

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(operator.mul, 2,3)
        print(future.result())

    # with concurrent.futures.Executor() as executor:
    #     futures = {executor.submit(perform, task) for task in get_tasks()}
    #
    #     for fut in concurrent.futures.as_completed(futures):
    #         print(f"The outcome is {fut.result()}")


def threadrelated():
    thread_local = threading.local()

    def get_session():
        if not hasattr(thread_local, 'session'):
            thread_local.session = requests.Session()
        return thread_local.session

    def download_urls(url):
        session = get_session()
        with session.get(url) as response:
            print(f"Read {len(response.content)} from {url}")

    def download_site(sites):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as workers:
            workers.map(download_urls, sites)


def multiprocessrelate():
    session = None

    def set_global_session():
        global session
        if not session:
            session = requests.Session()

    def download_site(url):
        with session.get(url) as response:
            name = current_process().name
            print(f"{name}:Read {len(response.content)} from {url}")

    def download_sites(sites):
        with Pool(initializer=set_global_session) as pool:
            pool.map(download_site, sites)


# zipOperator()
# mapOp()
# sumOp()
# better_grouper(range(10), 2)
# zipLong()
# combination()
# counterdemo()
# acc()
# chainitr()
# grpby()
# collectinCounter()
# defaultDictOp()
# dqOp()
# namedTuple()
# binarsearch()
# gtLtEx()
# datclassEx()
# memoryUitl()
# useCache()
# print(list(window(n=3)))
# cacheprop()
# paralleprocess()

# dotread()
