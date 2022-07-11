# gym-po-taxi
Partially-observable taxi environment, with internal vectorization


## Links to look at for own implementation

[Gymnax](https://github.com/RobertTLange/gymnax/): Classic control, bsuite, MinAtar, FourRooms, MetaMaze, PointRobot, Bandits in JAX.
Supports [Podracer](https://arxiv.org/pdf/2104.06272.pdf) architecture
- Most interesting environments are probably MemoryChain, FourRooms, MetaMaze, PointRobot

[ROOMS and C-ROOMS](https://github.com/aijunbai/hplanning): ROOMS and C-ROOMs for reference
- Velocity-based vs just position
- Fixed layouts ahead of time. Random agent spawn. Fixed or set or random goal
- Discrete action (8 or 4 cardinal directions) vs Continuous (2D)
  - 2 forms of action failure. 0.2 chance of taking random action (cardinal) or flipping signs (continuous). 0.2 standard deviation for Gaussian movement
- What to do for walls?
  - Discrete case is easy. Don't move. 
  - Continuous case could be the same. Alternatively, draw the vector, stop right at wall. 
- Observation?
  - Non-continuous: 
    - Fully observable: grid discrete state. Goal state if random?
    - Partially observable: 4D Hansen (adjacent), 8D Hansen, nxn grid
  - Continuous:
    - Fully observable: (x,y) coordinate, Need (dx, dy) if velocity-based. Goal state if random?
    - Partially observable: 
      - (x,y) w/o velocity, (x,y) downsampled to grid
      - 4/8D Hansen (0/1 walls in range 1M), 4/8D walls (distance of closest wall)

[Pocman/Pacman](https://github.com/bmazoure/ms_pacman_gym): Fully/partially-observable pocman from [POMCP](https://proceedings.neurips.cc/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html)

[Battleship](https://github.com/thomashirtz/gym-battleship): Partially observable battleship

[Rocksample](https://github.com/d3sm0/gym_pomdp): Also has battleship

[Isaacverse](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/): GPU physics control

[Mo-Gym](https://github.com/LucasAlegre/mo-gym): Multi-objective. Fancy fourrooms, reacher with more objectives, 

[gym-sokoban](https://github.com/mpSchrader/gym-sokoban): pixel-based though...

[CARL](https://github.com/automl/CARL): Context-adaptive RL, reconfigure envs (Mario, Brax, control)

[highway-env](https://github.com/eleurent/highway-env): Must infer behaviors of others



Other
- [SpaceRobot](https://github.com/Tsinghua-Space-Robot-Learning-Group/SpaceRobotEnv): Non-actuated base space robot
- [Learn2Race](https://github.com/learn-to-race/l2r/): Needs GPU. Eh...
- [tmrl](https://github.com/trackmania-rl/tmrl/): TrackMania racing, 19-D LIDAR option
- [ShinRL](https://github.com/omron-sinicx/ShinRL/): Future reference, interesting
- 

