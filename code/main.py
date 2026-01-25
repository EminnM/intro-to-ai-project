import numpy as np
import argparse
from pathlib import Path
from typing import Iterable, Dict, Any
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
import highway_env
import os 
from stable_baselines3 import DQN
from gymnasium.wrappers import RecordVideo

DURATION = 200
HIGH_SPEED_RANGE = [25, 35]
CHANGE_LANE_REWARD = -0.25
HIGH_SPEED_REWARD = 10
COLLISION_REWARD = -10

config = {
            "reward_speed_range":HIGH_SPEED_RANGE,
            "lane_change_reward":CHANGE_LANE_REWARD,
            "high_speed_reward":HIGH_SPEED_REWARD ,
            "collision_reward": COLLISION_REWARD,
            "duration": DURATION,
            "normalize_rewards": False,
        }



VIDEO_DIR = Path("videos")
MODEL_DIR = Path("models")

VIDEO_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


class SpeedMonitorCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.episode_speeds = []
            self.current_episode_speeds = []

        def _on_step(self) -> bool:
            info = self.locals["infos"][0]

            if "speed" in info:
                self.current_episode_speeds.append(info["speed"])

            if self.locals["dones"][0]:
                if len(self.current_episode_speeds) > 0:
                    avg_speed = np.mean(self.current_episode_speeds)
                    self.logger.record("rollout/avg_speed", avg_speed)
                    self.episode_speeds.append(avg_speed)

                self.current_episode_speeds = []

            return True

class RewardMonitorCallback(BaseCallback):
    def __init__(self, v:list, lane_penalty:float, crash_penalty:float, verbose=0):
        super().__init__(verbose)
        self.ep_speed = []
        self.ep_lane = []
        self.ep_collision = []
        self.v = v
        self.lane_penalty = lane_penalty
        self.crash_penalty = crash_penalty

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        action = self.locals["actions"][0]
        done = self.locals["dones"][0]

        #  Speed reward
        if "speed" in info:
            speed = info["speed"]
            v_min, v_max = self.v
            speed_reward = np.clip((speed - v_min) / (v_max - v_min), 0, 1)
            self.ep_speed.append(speed_reward)

        # Lane change penalty 
        if action in [0, 2]:  # LEFT or RIGHT
            self.ep_lane.append(self.lane_penalty)
        else:
            self.ep_lane.append(0.0)

        # --- Collision penalty
        if info.get("crashed", False):
            self.ep_collision.append(self.crash_penalty)

        if done:
            self.logger.record("reward/speed", np.sum(self.ep_speed))
            self.logger.record("reward/lane", np.sum(self.ep_lane))
            self.logger.record("reward/collision", np.sum(self.ep_collision))
            self.logger.record(
                "reward/total",
                np.sum(self.ep_speed) + np.sum(self.ep_lane) + np.sum(self.ep_collision),
            )

            self.ep_speed.clear()
            self.ep_lane.clear()
            self.ep_collision.clear()

        return True

class ModelSaveCallback(BaseCallback):
    def __init__(self, save_timesteps: list, path: Path):
        super().__init__()
        self.save_timesteps = save_timesteps
        self.path = path

    def _on_step(self):
        for timestep in self.save_timesteps:
            if self.num_timesteps >= timestep:
                self.save_timesteps.remove(timestep)
                self.model.save(self.path / f"model_{timestep}")
                break
        return True

def record(model_path:Path, steps=100) -> None:
    model = DQN.load(model_path)

    env = gym.make(id="highway-fast-v0", render_mode="rgb_array",config={"duration":200})
    env = RecordVideo(env, video_folder=VIDEO_DIR, fps=30)

    obs, _ = env.reset()

    for _ in range(steps):
        action, _ = model.predict(obs, deterministic=True)

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

def train(total_timesteps: int) -> None:
    env = gym.make("highway-fast-v0", config=config)
    
    model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=1e-3,
              buffer_size=50_000,
              learning_starts=200,
              batch_size=32,
              gamma=0.99,
              exploration_initial_eps = 0.2,
              exploration_fraction = 0.05,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=500,
              verbose=1,
              tensorboard_log="highway_dqn/")

    save_timesteps = [
                int(total_timesteps * 0.01),
                int(total_timesteps * 0.1),
            ]
    callback = [
                SpeedMonitorCallback(), 
                ModelSaveCallback(save_timesteps, MODEL_DIR),
                RewardMonitorCallback(v=HIGH_SPEED_RANGE, lane_penalty=CHANGE_LANE_REWARD, crash_penalty=COLLISION_REWARD)
                ]

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )

    model.save("highway_dqn/model")


def evaluate(model_path, n_episodes):
    model = DQN.load(model_path)
    env = gym.make("highway-fast-v0", render_mode="human", config={"duration":DURATION})

    for _ in range(n_episodes):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Highway-Env Runner"
    )

    parser.add_argument(
        "--mode",
        choices=["train", "eval", "record"],
        required=True,
        help="Operation mode",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=1e5,
        help="Training timesteps (train mode)",
    )

    parser.add_argument(
        "--save-name",
        type=str,
        help="Model save name",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to trained model",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Recording/Evaluation episodes",
    )


    args = parser.parse_args()

    if args.mode == "train":
        train(args.timesteps)

    elif args.mode == "eval":
        if args.model_path is None:
            raise ValueError("--model-path is required for eval mode")
        evaluate(Path(args.model_path), args.episodes)

    elif args.mode == "record":
        if args.model_path is None:
            raise ValueError("--model path is required for record mode")

        record(args.model_path)


if __name__ == "__main__":
    main()

