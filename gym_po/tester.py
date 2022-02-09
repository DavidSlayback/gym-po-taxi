import time

from wrappers import RecordEpisodeStatistics, NormalizeReward
from envs import *
from gym.wrappers.record_video import RecordVideo
from envs.multistory_fourrooms_v3 import MultistoryFourRoomsVecEnv

if __name__ == "__main__":
    # e = DiscreteActionCarVecEnv(7, 20, time_limit=160)
    # e = TaxiVecEnv(2, num_passengers=3, hansen_obs=False, time_limit=10000)
    # e = TaxiEnv()
    # e = HansenTaxiVecEnv(8, time_limit=200)
    e = MultistoryFourRoomsVecEnv(8, time_limit=1000, grid_z=3, obs_n=5, goal_floor=0)
    # o = e.reset()
    # img = e.render()
    e = RecordVideo(e, video_folder='videos')
    # e = NormalizeReward(e, 0.95)
    # e = RecordEpisodeStatistics(e, 0.95)
    # e = NormalizeReward(e, 0.95)
    # o = e.reset()
    # on = e.single_observation_space.n
    # assert (o <= on).all()
    o = e.reset()
    for t in range(10000):
        o, r, d, info = e.step(e.action_space.sample())
        # e.render()
        # time.sleep(0.2)
        # if d.any(): print(info)
        # assert (o <= on).all()
    e.close()
    print(3)
