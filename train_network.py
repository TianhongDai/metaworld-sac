import torch
import numpy as np
import random
from arguments import get_args
from sac_agent import sac_agent
from utils import env_wrapper
# from metaworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import SawyerPickAndPlaceEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place import SawyerReachPushPickPlaceEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerFaucetOpenEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerSweepEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerBinPickingEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerWindowOpenEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerBasketballEnv
from metaworld.envs.mujoco.sawyer_xyz import SawyerSweepIntoGoalEnv

TASKS = {'sawyer_faucet_open': SawyerFaucetOpenEnv,
        'pick_place': SawyerReachPushPickPlaceEnv, 
        'sweep': SawyerSweepEnv,
        'pick_bin': SawyerBinPickingEnv,
        'open_window': SawyerWindowOpenEnv,
        'basketball': SawyerBasketballEnv,
        'sweep_into_hole': SawyerSweepIntoGoalEnv,
        }

if __name__ == '__main__':
    args = get_args()
    # build the environment
    # env = TASKS[args.env_name]()
    env = TASKS[args.env_name](random_init=args.random_init, obs_type=args.obs_type)
    env = env_wrapper(env, args)
    env.seed(args.seed)
    # create the eval env
    # eval_env = TASKS[args.env_name]()
    eval_env = TASKS[args.env_name](random_init=args.random_init, obs_type=args.obs_type)
    eval_env = env_wrapper(eval_env, args)
    eval_env.seed(args.seed + 100)
    # set seeds
    np.random.seed(args.seed)
    random.seed(args.seed)
    # set the seed of torch
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # create the agent
    sac_trainer = sac_agent(env, eval_env, args)
    sac_trainer.learn()
