from collections import defaultdict


class Trie:
    def __init__(self):
        self._end = '_end_'
        self.root = defaultdict(dict)

    def make_trie(self, *words):

        for word in words:
            current_dict = self.root
            for letter in word:
                current_dict = current_dict.setdefault(letter, {})
            current_dict[self._end] = self._end
        return self.root

    def in_trie(self, trie, word):
        current_dict = trie
        for letter in word:
            if letter not in current_dict:
                return False
            current_dict = current_dict[letter]
        return self._end in current_dict


t= Trie()
x = t.make_trie('foo', 'bar', 'baz', 'barz')
print(t.in_trie(x, 'bard'))
