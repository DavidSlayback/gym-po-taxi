import time

from wrappers import RecordEpisodeStatistics, NormalizeReward
from envs import *
from gym.wrappers.record_video import RecordVideo

if __name__ == "__main__":
    # e = CarVecEnv(20, time_limit=160)
    # e = TaxiEnv()
    e = HansenTaxiVecEnv(8, time_limit=200)
    # e = MultistoryFourRoomsVecEnv(2, time_limit=1000)
    e = RecordVideo(e, video_folder='videos')
    # e = NormalizeReward(e, 0.95)
    # e = RecordEpisodeStatistics(e, 0.95)
    # e = NormalizeReward(e, 0.95)
    o = e.reset()
    # on = e.single_observation_space.n
    # assert (o <= on).all()
    for t in range(1000):
        o, r, d, info = e.step(e.action_space.sample())
        # e.render()
        # time.sleep(0.2)
        # if d.any(): print(info)
        # assert (o <= on).all()
    e.close()
    print(3)
