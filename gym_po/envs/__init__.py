from .car_flag import CarVecEnv, DiscreteActionCarVecEnv
from .extended_taxi import TaxiVecEnv, HansenTaxiVecEnv, ExtendedHansenTaxiVecEnv, EXTENDED_TAXI_MAP, \
    ExtendedTaxiVecEnv, TAXI_MAP
from .rooms import RoomsEnv, CRoomsEnv, MultistoryFourRoomsEnv

from gymnasium.envs.registration import register


register(
    id='pdomains-ant-heaven-hell-v1',
    entry_point='gym_po.envs.ant_heaven_hell:AntHeavenHellEnv',
    max_episode_steps=500,
)

register(
    id='pdomains-ant-tag-v1',
    entry_point='gym_po.envs.ant_tag:AntTagEnv',
    max_episode_steps=500,
)