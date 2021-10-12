from wrappers import RecordEpisodeStatistics, NormalizeReward
from envs import *

if __name__ == "__main__":
    e = CarVecEnv(20, time_limit=20000)
    # e = TaxiVecEnv(20, time_limit=200)
    e = NormalizeReward(e, 0.95)
    e = RecordEpisodeStatistics(e, 0.95)
    # e = NormalizeReward(e, 0.95)
    o = e.reset()
    for t in range(10000):
        o, r, d, info = e.step(e.action_space.sample())
        if d.any(): print(info)
    print(3)
