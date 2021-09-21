def merge_intervals():
    def merge(intervals):
        out = []
        for i in sorted(intervals, key=lambda i: i[1]):
            if out and i[0] <= out[-1][1]:
                out[-1][1] = max(out[-1][1], i[1])
            else:
                out += i,
        return out

    interval = [[1, 3], [2, 6], [8, 10], [15, 18]]
    print(merge(interval))

merge_intervals()
