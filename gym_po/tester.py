from wrappers import RecordEpisodeStatistics, NormalizeReward
from envs import *
from gym.wrappers.record_video import RecordVideo

if __name__ == "__main__":
    # e = CarVecEnv(20, time_limit=160)
    # e = TaxiEnv()
    e = HansenTaxiVecEnv(8, time_limit=200)
    e = RecordVideo(e, video_folder='videos')
    # e = NormalizeReward(e, 0.95)
    # e = RecordEpisodeStatistics(e, 0.95)
    # e = NormalizeReward(e, 0.95)
    o = e.reset()
    on = e.single_observation_space.n
    assert (o <= on).all()
    for t in range(10000):
        o, r, d, info = e.step(e.action_space.sample())
        if d.any(): print(info)
        assert (o <= on).all()
    e.close()
    print(3)
