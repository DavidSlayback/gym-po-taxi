import time

import numpy as np

from envs import *
from gym.wrappers.record_video import RecordVideo
from envs.multistory_fourrooms_v3 import MultistoryFourRoomsVecEnv
from envs.ant_tag import AntTagEnv

if __name__ == "__main__":
    e = AntTagEnv()
    # e = DiscreteActionCarVecEnv(7, 20, time_limit=160)
    # e = TaxiVecEnv(8, num_passengers=2, hansen_obs=True, time_limit=2000, map=EXTENDED_TAXI_MAP)
    # e = MultistoryFourRoomsVecEnv(8, time_limit=1000, grid_z=3, obs_n=0, goal_floor=0)
    # o = e.reset()
    # img = e.render()
    # e.metadata["video.frames_per_second"] = 60
    # e = RecordVideoWithText(e, video_folder='videos', render_idx=np.arange(8))
    # e = RecordVideo(e, video_folder='videos')
    # e = NormalizeReward(e, 0.95)
    # e = RecordEpisodeStatistics(e, 0.95)
    # e = NormalizeReward(e, 0.95)
    # o = e.reset()
    # on = e.single_observation_space.n
    # assert (o <= on).all()
    o = e.reset()
    for t in range(100000):
        o, r, d, info = e.step(e.action_space.sample())
        e.render()
        # e.render(idx=np.arange(8))
        # if (r > 0).any(): print('Reward!')
        # e.render()
        # time.sleep(0.2)
        # if d.any(): print(info)
        # assert (o <= on).all()
    e.close()
    print(3)
