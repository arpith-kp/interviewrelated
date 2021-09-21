def remove_duplicate():
    def removeDuplicateLetters(s):
        rindex = {c: i for i, c in enumerate(s)}
        result = ''
        for i, c in enumerate(s):
            if c not in result:
                while c < result[-1:] and i < rindex[result[-1]]:
                    result = result[:-1]
                result += c
        return result

    print(removeDuplicateLetters("aavdevss"))

remove_duplicate()
