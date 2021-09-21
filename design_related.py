from collections import OrderedDict


class AuthenticationManager:
    def __init__(self, timeToLive: int):
        self.expiry = OrderedDict()
        self.life = timeToLive

    def generate(self, tokenId: str, currentTime: int) -> None:
        self.evict_expired(currentTime)
        self.expiry[tokenId] = self.life + currentTime

    def renew(self, tokenId: str, currentTime: int) -> None:
        self.evict_expired(currentTime)
        if tokenId in self.expiry:
            self.expiry.move_to_end(tokenId)  # necessary to move to the end to keep expiry time in ascending order.
            self.expiry[tokenId] = self.life + currentTime

    def countUnexpiredTokens(self, currentTime: int) -> int:
        self.evict_expired(currentTime)
        return len(self.expiry)

    def evict_expired(self, currentTime: int) -> None:
        while self.expiry and next(iter(self.expiry.values())) <= currentTime:
            self.expiry.popitem(last=False)


class LogSystem(object):
    def __init__(self):
        self.logs = []

    def put(self, tid, timestamp):
        self.logs.append((tid, timestamp))

    def retrieve(self, s, e, gra):
        index = {'Year': 5, 'Month': 8, 'Day': 11,
                 'Hour': 14, 'Minute': 17, 'Second': 20}[gra]
        start = s[:index]
        end = e[:index]

        return sorted(tid for tid, timestamp in self.logs
                      if start <= timestamp[:index] <= end)

logSystem = LogSystem()
logSystem.put(1, "2017:01:01:23:59:59")
logSystem.put(2, "2017:01:01:22:59:59")
logSystem.put(3, "2016:01:01:00:00:00")
print(logSystem.retrieve("2016:01:01:01:01:01", "2017:01:01:23:00:00", "Year"))
print(logSystem.retrieve("2016:01:01:01:01:01", "2017:01:01:23:00:00", "Hour"))
