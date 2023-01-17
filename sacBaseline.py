import torch
import gym

from stable_baselines3.sac.sac import SAC #doesn't work     #
# from stable_baselines3.ppo.ppo import PPO #works!           # doesn't
# from stable_baselines3.a2c.a2c import A2C #doesn't work   #
# from stable_baselines3.dqn.dqn import DQN #works!         #
# from stable_baselines3.ddpg.ddpg import DDPG #slow af     #

env = gym.make("CarRacing-v0")#, render_mode="human")
model = SAC("MlpPolicy", env, verbose=1, seed=34)
# model=model.load('sacMountainCarTest.pt', env)
model.learn(total_timesteps=500_000)
model.save('carRacingSAC.pt')
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