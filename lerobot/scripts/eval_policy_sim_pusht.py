#%%
import gymnasium as gym
import gym_pusht  # noqa: F401

# import mani_skill.envs
import time
# from mani_skill.utils.wrappers import CPUGymWrapper
# import matplotlib.pyplot as plt
import torch
# import tqdm
import numpy as np
# from IPython.display import Video

# from mani_skill.trajectory.dataset import ManiSkillTrajectoryDataset
# from mani_skill.utils.io_utils import load_json
# from mani_skill.trajectory.utils import index_dict, dict_to_list_of_dicts
# from mani_skill.utils.visualization.misc import images_to_video
# from mani_skill.utils.wrappers.record import RecordEpisode
# import multiprocessing
from pathlib import Path

import cv2

import time

import einops

import hydra
from hydra import compose, initialize
from omegaconf import OmegaConf

from lerobot.common.utils.utils import _relative_path_between
import yaml

from omegaconf import OmegaConf
import wandb
import zarr
import json
import subprocess
import logging

# make a video writer
import imageio
import os

from typing import Optional

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

eval_policy_sim_logger = logging.getLogger(__name__)

eval_policy_sim_logger.info("finished importing libraries")

#%%
def get_coverage_from_info(info):
    coverage = info.get('coverage', None)
    if coverage is None:
        coverage = info['final_info'][0]['coverage']
    return float(coverage)

def recursive_json_serialize_dict(item):
    """
    Recursively converts numpy arrays and PyTorch tensors in a dictionary to lists
    so they can be serialized to JSON.
    """
    if isinstance(item, dict):
        return {key: recursive_json_serialize_dict(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [recursive_json_serialize_dict(element) for element in item]
    elif isinstance(item, np.ndarray):
        return item.tolist()
    elif torch.is_tensor(item):
        return item.detach().cpu().numpy().tolist()
    # handle int64
    elif isinstance(item, np.int64) or isinstance(item, np.int32) or isinstance(item, torch.Tensor):
        return int(item)
    else:
        return item
    
class RenderHandler:
    def __init__(self, 
                 desired_viewing_size: tuple = (512, 512),
                 open_cv_window: bool = False,
                record_episode: bool = False,
                video_output_path: Optional[str] = None,
                seed_list: Optional[list] = None,
                 ):
        self.open_cv_window = open_cv_window
        self.desired_viewing_size = desired_viewing_size
        if self.open_cv_window:
            cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)

        self.record_episode = record_episode
        if self.record_episode:
            self.traj_counter = 0
            self.seed_list = seed_list
            self.video_output_path = video_output_path
            video_path = Path(self.video_output_path) / f'{self.traj_counter}_traj_{self.seed_list[self.traj_counter]}_envseed.mp4'
            self.video_writer = imageio.get_writer(str(video_path), fps=10, quality=5)

    def render(self, env, action_plan=None, action=None):
        # frame = cv2.cvtColor(obs['pixels'][0], cv2.COLOR_RGB2BGR)
        frame = cv2.cvtColor(env.envs[0].render(), cv2.COLOR_RGB2BGR)
        if frame.shape[:2] != self.desired_viewing_size:
            frame = cv2.resize(frame, self.desired_viewing_size)
        if action_plan is not None:
            # draw the 2D action plan with a color gradient to indicate the time step
            for i in range(action_plan.shape[1]):
                draw_action = (int(action_plan[0][i][0]*(self.desired_viewing_size[0]/512)), int(action_plan[0][i][1]*(self.desired_viewing_size[1]/512)))
                # get the color for the action plan
                color = (0, int(255*(1-(i/action_plan.shape[1]))), int(255*((i/action_plan.shape[1]))))
                # draw the action plan
                cv2.circle(frame, draw_action, 5, color, -1)
        if self.open_cv_window:
            cv2.imshow("frame", frame)
        if self.record_episode:
            if self.video_writer is None:
                self.traj_counter += 1
                # then open a new video writer
                self.video_writer = imageio.get_writer(str(self.video_output_path / f'{self.traj_counter}_traj_{self.seed_list[self.traj_counter]}_envseed.mp4'), fps=10, quality=5)
            # save the frame
            # convert the frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.video_writer.append_data(frame)
        
    def check_user_input(self):
        if self.open_cv_window:
            key = cv2.waitKey(1) & 0xFF
        else:
            key = None
        return key
    
    def write_video(self):
        if self.record_episode:
            self.video_writer.close()
            self.video_writer = None
            
#%%
class DiffusionPolicyLerobotPushTWrapper:
    def __init__(self, 
    cfg,
    # segmentation_id_map: dict, 
    sim_control_freq: int, 
    num_envs: int,
    env_device: str = 'cpu',
    policy = None, 
    ):
        self.running_locally = os.uname().nodename == 'MAGI-SYSTEM'

        # >>>>>>>>>> make diffusion policy using omegaconf
        self.cfg = cfg
        path_to_pretrained_checkpoint_dir = Path(self.cfg.eval_cfg.path_to_pretrained_checkpoint_dir)
        assert path_to_pretrained_checkpoint_dir.exists(), f"Path to pretrained checkpoint dir {path_to_pretrained_checkpoint_dir} does not exist"
        self.policy = DiffusionPolicy.from_pretrained(path_to_pretrained_checkpoint_dir)
        self.policy.reset()
        # <<<<<<<<<< make diffusion policy using omegaconf

        self.env_device = env_device

        # self.policy = policy
        self.num_envs = num_envs
        # self.device = self.policy.device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)
        self.policy_config = self.policy.config

        self.action_dim = self.policy.config.output_features['action'].shape[0]
        self.action_plan_horizon = self.policy.config.horizon
        self.action_plan_horizon += -(self.policy.config.n_obs_steps - 1)

        self.current_action_plan = None
     
        self.elapsed_steps = 0

        self.n_obs_steps = self.policy.config.n_obs_steps
        if self.n_obs_steps > 1:
            # then set up queues for the observations
            # HACK JUST HARDCODING THE HEIGHT AND WIDTH FOR NOW
            self.rgb_queue = torch.zeros(num_envs, self.n_obs_steps, 3, 96, 96, dtype=torch.float32).to(self.device)
            self.depth_queue = torch.zeros(num_envs, self.n_obs_steps, 2, dtype=torch.float32).to(self.device)

        # self.policy_freq = self.policy_config.policy_frequency
        # self.sim_control_freq = sim_control_freq
        self.replan_in_n_steps = self.policy.config.n_action_steps

        self.lerobot_rgb_batch_key = 'observation.image'

    def pusht_obs_to_lerobot_obs(self, obs):
        # TODO: actually implement sequence length handling. For now, we'll just hardcode it in here
        # images should be BxSxCxHxW
        lerobot_batch = {}
        # lerobot_batch['observation.image'] = torch.as_tensor(observations['pixels'], device=self.config.device).float().unsqueeze(0)
        # BXHXWX3
        
        current_rgb = einops.rearrange(torch.from_numpy(obs['pixels']).float().unsqueeze(1), 'b s h w c -> b s c h w').to(self.device)
        # then need to mimic the normalized rgb images that the lerobot dataset uses
        current_rgb = current_rgb / 255.0
        if self.n_obs_steps > 1:
            # later indices are more recent
            if self.elapsed_steps == 0:
                # then need to fill the queue with the current rgb
                self.rgb_queue = torch.cat([current_rgb]*self.n_obs_steps, dim=1)
            else:
                self.rgb_queue = torch.cat([self.rgb_queue[:, 1:], current_rgb], dim=1)
            lerobot_batch[self.lerobot_rgb_batch_key] = self.rgb_queue
        else:
            lerobot_batch[self.lerobot_rgb_batch_key] = current_rgb

        # lerobot_batch['observation.state'] = torch.as_tensor(observations['features'], device=self.config.device).float().unsqueeze(0) # add batch dimension
        current_agent_pos = torch.from_numpy(obs['agent_pos']).unsqueeze(1).float().to(self.device) # bx2 -> bxsx2
        if self.n_obs_steps > 1:
            if self.elapsed_steps == 0:
                self.agent_pos_queue = torch.cat([current_agent_pos]*self.n_obs_steps, dim=1)
            else:
                self.agent_pos_queue = torch.cat([self.agent_pos_queue[:, 1:], current_agent_pos], dim=1)
            lerobot_batch['observation.state'] = self.agent_pos_queue
        else:
            lerobot_batch['observation.state'] = current_agent_pos
        return lerobot_batch
    
    def reset(self):
        self.current_action_plan = None
        if self.n_obs_steps > 1:
            self.rgb_queue = torch.zeros(self.num_envs, self.n_obs_steps, 3, 96, 96, dtype=torch.float32).to(self.device)
            self.depth_queue = torch.zeros(self.num_envs, self.n_obs_steps, 2, dtype=torch.float32).to(self.device)
        self.elapsed_steps = 0
        self.policy.reset()
            
    def act(self, maniskill_obs):
        if self.elapsed_steps % self.replan_in_n_steps == 0:
            lerobot_obs = self.pusht_obs_to_lerobot_obs(maniskill_obs)
            _, action_plan = self.policy.get_action_plan(lerobot_obs) # BxTxA
            if self.n_obs_steps > 1:
                action_plan = action_plan[:, self.n_obs_steps-1:] # remove the first n_obs_steps-1 steps
            assert action_plan.shape == (self.num_envs, self.action_plan_horizon, self.action_dim), f"action_plan should be {(self.num_envs, self.action_plan_horizon, self.action_dim)}, but got {action_plan.shape}"
            self.current_action_plan = action_plan
        assert self.current_action_plan is not None, "current_action_plan is None"
        # get the action to be executed from the current_action_plan
        action_to_execute = self.current_action_plan[:, self.elapsed_steps % self.replan_in_n_steps]
        visualized_plan = self.current_action_plan[:, self.elapsed_steps % self.replan_in_n_steps:]
        self.elapsed_steps += 1
        return action_to_execute.to(self.env_device), visualized_plan
    
# with initialize(version_base=None, config_path="cfgs", job_name="test_app"):
#     omegaconf_cfg = compose(config_name="config_eval", overrides=["eval_cfg=sim", "agent=diffusion", "suite=frankagym", "suite/frankagym_task@_global_=insertion", "use_wandb=false"])
from lerobot.common.envs.factory import make_env
#%%
@hydra.main(config_path="cfgs", config_name="config_eval_pusht")
def main(omegaconf_cfg):
    record_episode = True
    run_at_realtime = False

    if omegaconf_cfg.eval_cfg.universal_unseen_env_seed_start is not None:
        omegaconf_cfg.eval_cfg.output_dir = Path(omegaconf_cfg.eval_cfg.output_dir) / f"seed_start_{omegaconf_cfg.eval_cfg.universal_unseen_env_seed_start}_seed_end_{omegaconf_cfg.eval_cfg.universal_unseen_env_seed_start + omegaconf_cfg.eval_cfg.num_unseen_eval_envs}"
        
    # add date and time to the output directory
    import datetime
    now = datetime.datetime.now()
    omegaconf_cfg.eval_cfg.output_dir = omegaconf_cfg.eval_cfg.output_dir / now.strftime("%Y-%m-%d_%H-%M-%S")
    if not omegaconf_cfg.eval_cfg.output_dir.exists():
        omegaconf_cfg.eval_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    # make the policy

    env = make_env(omegaconf_cfg.env, n_envs=1)
    eval_policy_sim_logger.info("made the environment")
    diffusion_policy = DiffusionPolicyLerobotPushTWrapper(
        cfg=omegaconf_cfg,
        sim_control_freq=omegaconf_cfg.env.fps,
        num_envs=env.num_envs,
        env_device='cpu',
    )
    eval_policy_sim_logger.info("made the policy")

    #%%
    num_trajs = 0
    # save the results to a json
    eval_results = dict(episodes=list())
    #%%
    sim_dt = 1.0 / omegaconf_cfg.env.fps
    sim_dt_bw_step = sim_dt

    max_eval_length = omegaconf_cfg.env.episode_length
    #%%
    # get the wandb training run just to get some metadata
    seed_list = list()
    seed_desc_list = list()

    assert diffusion_policy.cfg.eval_cfg.universal_unseen_env_seed_start is not None, "universal_unseen_env_seed_start must be set in the config" 
    unseen_env_seeds_to_eval = np.arange(diffusion_policy.cfg.eval_cfg.universal_unseen_env_seed_start, diffusion_policy.cfg.eval_cfg.universal_unseen_env_seed_start + diffusion_policy.cfg.eval_cfg.num_unseen_eval_envs)
    seed_list.extend(unseen_env_seeds_to_eval)
    seed_desc_list.extend(['unseen']*len(unseen_env_seeds_to_eval))

    assert len(seed_list) == len(seed_desc_list), f"len(seed_list): {len(seed_list)}, len(seed_desc_list): {len(seed_desc_list)}"

    # create the output directory
    Path(omegaconf_cfg.eval_cfg.output_dir).mkdir(parents=True, exist_ok=True)

    eval_policy_sim_logger.info(f"seed_list: {seed_list}")
    render_handler = RenderHandler(desired_viewing_size=(512, 512), record_episode=record_episode, video_output_path=omegaconf_cfg.eval_cfg.output_dir, seed_list=seed_list)

    # also write the omegaconf config to the output dir as yaml
    # copy the config.json and train_config.json to the output dir
    path_to_pretrained_checkpoint_dir = Path(omegaconf_cfg.eval_cfg.path_to_pretrained_checkpoint_dir)
    assert path_to_pretrained_checkpoint_dir.exists(), f"Path to pretrained checkpoint dir {path_to_pretrained_checkpoint_dir} does not exist"
    import shutil
    # copy the config.json and train_config.json to the output dir
    shutil.copy(path_to_pretrained_checkpoint_dir / 'config.json', omegaconf_cfg.eval_cfg.output_dir)
    shutil.copy(path_to_pretrained_checkpoint_dir / 'train_config.json', omegaconf_cfg.eval_cfg.output_dir)

    eval_results_json_path = Path(omegaconf_cfg.eval_cfg.output_dir) / 'eval_results.json'
    
    #%%
    eval_policy_sim_logger.info("starting evaluation")
    # evaluate the policy
    num_trajs = 0
    key = None
    info = None
    episode_success = list()
    for (seed, seed_desc) in zip(seed_list, seed_desc_list):
        seed = int(seed)
        print(f"evaluating seed: {seed}, seed_desc: {seed_desc}")
        if key is not None or info is not None:
            if key == ord('q'):
                num_trajs += 1
                break
            # elif key == ord('c'):
            elif info['is_success']: # policy succeeded
                num_trajs += 1
                episode_success.append(True)
                
                logged_info_dict = recursive_json_serialize_dict(dict(coverage=get_coverage_from_info(info), is_success=info['is_success']))
                eval_results['episodes'].append(dict(seed=int(seed), seed_desc=seed_desc, info=logged_info_dict))
                obs, info = env.reset(seed=seed)
                diffusion_policy.reset()
                action, action_plan = diffusion_policy.act(obs)
                render_handler.render(env, action_plan=action_plan, action=action)
                render_handler.write_video()
            elif not info['is_success']: # policy failed
                num_trajs += 1
                episode_success.append(False)
                logged_info_dict = recursive_json_serialize_dict(dict(coverage=get_coverage_from_info(info), is_success=info['is_success']))
                eval_results['episodes'].append(dict(seed=int(seed), seed_desc=seed_desc, info=logged_info_dict))
                obs, info = env.reset(seed=seed)
                diffusion_policy.reset()
                action, action_plan = diffusion_policy.act(obs)
                render_handler.render(env, action_plan=action_plan, action=action)
                render_handler.write_video()

            # elif key == ord('r'):
            #     env.reset(seed=seed, options=dict(save_trajectory=False))
            #     diffusion_policy.reset()
            #     action = diffusion_policy.act(obs)
            #     viewer = env.render_human()
            #     continue
            else:
                break

            print(f"saving intermediate results at traj {num_trajs} to {eval_results_json_path}")
            with open(eval_results_json_path, 'w') as f:
                json.dump(eval_results, f)
        else: # first time
            obs, info = env.reset(seed=seed)
            diffusion_policy.reset()
            action, action_plan = diffusion_policy.act(obs)
            render_handler.render(env, action_plan=action_plan, action=action)

        start_time = time.perf_counter()
        # while True:
        for i in range(max_eval_length):
            # action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            action, action_plan = diffusion_policy.act(obs)
            render_handler.render(env, action_plan=action_plan, action=action)

            # frames.append(current_frame)
            
            elapsed_simtime = diffusion_policy.elapsed_steps * sim_dt_bw_step
            elapsed_realtime = time.perf_counter() - start_time
            if run_at_realtime:
                # time_to_sleep = sim_dt_bw_step - elapsed_time
                time_to_sleep = elapsed_simtime - elapsed_realtime
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
            if diffusion_policy.elapsed_steps % 100 == 0:
                print(f"realtime_factor: {elapsed_simtime/elapsed_realtime} | elapsed steps: {diffusion_policy.elapsed_steps} | elapsed rt {elapsed_realtime} | elapsed simt {elapsed_simtime}")
                print(f"success: {info['is_success']} | coverage: {get_coverage_from_info(info)}")
            
            key = render_handler.check_user_input()
            # if render_mode == 'human':
            if key == ord('q'):
                break
            elif info['is_success']:
                print("Policy succeeded, stopping episode")
                break
            # elif viewer.window.key_press('c'): 
            #     key = ord('c')
            #     break
            # elif viewer.window.key_press('r'):
            #     key = ord('r')
            #     break
        
    if info['is_success']: # policy succeeded
        num_trajs += 1
        episode_success.append(True)
        logged_info_dict = recursive_json_serialize_dict(dict(coverage=get_coverage_from_info(info), is_success=info['is_success']))
        eval_results['episodes'].append(dict(seed=int(seed), seed_desc=seed_desc, info=logged_info_dict))
        env.reset(seed=seed)
        diffusion_policy.reset()
        render_handler.write_video()
    elif not info['is_success']: # policy failed
        num_trajs += 1
        episode_success.append(False)
        logged_info_dict = recursive_json_serialize_dict(dict(coverage=get_coverage_from_info(info), is_success=info['is_success']))
        eval_results['episodes'].append(dict(seed=int(seed), seed_desc=seed_desc, info=logged_info_dict))
        env.reset(seed=seed)
        diffusion_policy.reset()
        render_handler.write_video()

    #%%
    assert num_trajs == len(seed_list) == len(seed_desc_list) == len(episode_success), f"num_trajs: {num_trajs}, len(seed_list): {len(seed_list)}, len(seed_desc_list): {len(seed_desc_list)}, len(episode_success): {len(episode_success)}"
    cv2.destroyAllWindows()
    #%%
    env.close()
    del env
    #%%
    for (seed, seed_desc, success) in zip(seed_list, seed_desc_list, episode_success):
        print(f"seed: {seed}, seed_desc: {seed_desc}, success: {success}")

    print(f"saving final results to {eval_results_json_path}")
    with open(eval_results_json_path, 'w') as f:
        json.dump(eval_results, f)

    # #%%
    # # load the results
    # with open(eval_results_json_path, 'r') as f:
    #     eval_results = json.load(f)
    # #%%
    # # plot the results
    # import binomial_cis
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # from collections import Counter
    # #%%
    # success_counter = Counter(eval_results['episode_success'])
    # success_counter
    # #%%
    # success_counter['True'], success_counter['False']
    # #%%
    # success_counter['True'] / (success_counter['True'] + success_counter['False'])
    # #%%
    # # plot the results
    # fig, ax = plt.subplots()
    # ax.bar(success_counter.keys(), success_counter.values())
    # ax.set_title("Success rate")
    # plt.show()
    # #%%

    sys.exit(0)

if __name__ == "__main__":
    main()