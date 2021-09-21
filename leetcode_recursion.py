from collections import deque
from random import choices
from typing import Tuple, List


def decode():
    """
    Moving next char in string is done using iterator
    :return:
    """
    def recursiveImplement(s: str, it) -> str:
        ret, times = [], 0
        for c in it:
            if c.isdigit():
                times = 10 * times + int(c)
            elif c == '[':
                ret.append(times * recursiveImplement(s, it))
                times = 0
            elif c == ']':
                break
            else:
                ret.append(c)
        return "".join(ret)

    def decodeString(s: str) -> str:
        it = iter(s)
        return recursiveImplement(s, it)

    def decodeString3(s:str)-> str:
        charT = deque()
        timeT = deque()
        inttracker = ''
        for c in s:
            if c.isdigit():
                inttracker+=c
            elif c == ']':
                x = charT.pop()
                temp = '' + x
                while x != '[':
                    x = charT.pop()
                    if x != '[':
                        temp+= x
                d = temp[::-1] * int(timeT.pop())
                charT.append(d)
            else:
                charT.append(c)
                if inttracker!='':
                    timeT.append(int(inttracker))
                    inttracker = ''
        return ''.join(charT)


    def decodeString2(s:str, idx=0, stored='', level=1):
        res = ''
        print( " " * level, f" Idx at {idx} char {stored}")
        times_to_repeat = 1
        for _ in s:
            if s[0] == ']':
                return stored
            if s[0].isdigit():
                times_to_repeat = s[0]
            if not s[0].isdigit() and s[0] != '[':
                stored += s[0]
            to_print = decodeString(s[1:], idx+1, stored, level*2)
            for _ in range(int(times_to_repeat)):
                res = res + to_print
        print(" " * level, f" Result {res}")
        return res

    print(decodeString3("10[a]")=='aaaaaaaaaa')

def tinyurl():
    """
    : time complexity of one encoding and decoding is O(n + m), where n is length of original string and m is length of encoded strigng, if we assume that probability of collision is small enough.
    :return:
    """
    class Codec:
        def __init__(self):
            self.long_short = {}
            self.short_long = {}
            self.alphabet = "abcdefghijklmnopqrstuvwzyz"

        def encode(self, longUrl):
            while longUrl not in self.long_short:
                code = "".join(choices(self.alphabet, k=6))
                if code not in self.short_long:
                    self.short_long[code] = longUrl
                    self.long_short[longUrl] = code
            return 'http://tinyurl.com/' + self.long_short[longUrl]

        def decode(self, shortUrl):
            return self.short_long[shortUrl[-6:]]


def partition():

    def labels(s:str) -> List[int]:
        final_list = list()
        temp_list = list()
        itr = iter(s)
        for char in itr:
            print(f"Evaluating {char}, templist {temp_list} , final list {final_list}")
            if not char:
                temp_list.append(char)
            elif char not in temp_list:

                next_char = next(itr)
                if next_char not in char or next_char != char:
                    final_list.append(''.join(temp_list))
                    temp_list = list()
                    continue
                temp_list.append(char)
                temp_list.append(next_char)
                print(f" char NOT found in templist : {temp_list}")
            elif char in temp_list:
                print(f"Found in temp {temp_list}")
                temp_list.append(char)
        return [len(x) for x in final_list]

    s = "ababcbacadefegdehijhklij"

    print(labels(s))

decode()
