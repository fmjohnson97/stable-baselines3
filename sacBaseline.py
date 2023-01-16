import torch
import gym

from stable_baselines3.sac.sac import SAC

env = gym.make("MountainCarContinuous-v0")#, render_mode="human")

model = SAC("MlpPolicy", env, verbose=1, seed=34) #PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()