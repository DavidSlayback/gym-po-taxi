import time

import numpy as np

from envs import *
from gym.wrappers.record_video import RecordVideo
from envs.rooms.rooms import RoomsEnv
from envs.rooms.crooms import CRoomsEnv
from envs.rooms.msrooms import generate_layouts_and_img, MultistoryFourRoomsEnv
# from envs.multistory_fourrooms_v3 import MultistoryFourRoomsVecEnv
from envs.ant_tag import AntTagEnv
from envs.ant_heaven_hell import AntHeavenHellEnv
import torch
v = torch.ones(1, device='cuda')

if __name__ == "__main__":
    # e = AntHeavenHellEnv(render_mode='human')
    e = MultistoryFourRoomsEnv(16, 3, obs_type='vector_goal_hansen', time_limit=10000)
    # env = envs.create_gym_env('walker2d', 20)
    # env = to_torch.JaxToTorchWrapper(env, device='cuda')
    # e = Rooms(16, '8b', obs_type='mdp', action_type='yx')
    o = e.reset()
    for _ in range(100):
        o2 = e.step(e.action_space.sample())
        # e.render()

    # e = AntTagEnv()
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
    # for t in range(100000):
    #     o, r, d, trunc, info = e.step(e.action_space.sample())
    #     if np.any(d | trunc): print(d.nonzero()[0], r[d.nonzero()[0]])
        # e.render()
        # e.render(idx=np.arange(8))
        # if (r > 0).any(): print('Reward!')
        # e.render()
        # time.sleep(0.2)
        # if d.any(): print(info)
        # assert (o <= on).all()
    # e.close()
    print(3)
