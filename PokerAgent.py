import os
import torch
import gymnasium as gym

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from stable_baselines3.common.vec_env import DummyVecEnv
from PokerEnv import Poker5EnvFull

SEED = 42
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 10000
N_EVAL_EPISODES = 30
REWARD_TARGET = 500
CHECKPOINT_DIR = "./checkpoints_poker/"
TENSORBOARD_DIR = "./tensorboard_poker/"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

class PokerPartialResetWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.total_episodes = 0

    def reset(self, *, seed=None, options=None):
        obs = self.env.partial_reset()

        return obs, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, truncated, info

    def action_masks(self):
        return self.env.action_masks()

def make_env():
    model1 = MaskablePPO.load("checkpoints_poker/MaskPPO10/ppo_poker_final.zip")
    model2 = MaskablePPO.load("checkpoints_poker/MaskPPO9/ppo_poker_final.zip")
    model3 = MaskablePPO.load("checkpoints_poker/MaskPPO8/ppo_poker_final.zip")
    model4 = MaskablePPO.load("checkpoints_poker/MaskPPO7/ppo_poker_final.zip")
    env = Poker5EnvFull(model_player1=model1, model_player2=model2, model_player3=model3, model_player4=model4)
    env = PokerPartialResetWrapper(env)
    return env

env = DummyVecEnv([make_env])
eval_env = DummyVecEnv([make_env])

callback_on_best = StopTrainingOnRewardThreshold(REWARD_TARGET, verbose=1)
eval_callback = MaskableEvalCallback(
    eval_env,
    callback_on_new_best=callback_on_best,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    best_model_save_path=CHECKPOINT_DIR,
    deterministic=False,
    render=False,
)

obs_space = env.observation_space
policy_type = "MultiInputPolicy" if isinstance(obs_space, gym.spaces.Dict) else "MlpPolicy"

model = MaskablePPO(
    policy=policy_type,
    env=env,
    verbose=1,
    seed=SEED,
    tensorboard_log=TENSORBOARD_DIR,

    learning_rate=3e-4,
    n_steps=2048,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=eval_callback,
    tb_log_name="MaskablePPO_Poker_Run",
)

model.save(os.path.join(CHECKPOINT_DIR, "ppo_poker_final"))

print("✅ Entrenamiento completado. Mejor modelo guardado en:", CHECKPOINT_DIR)