import torch
import gym

from stable_baselines3.sac.sac import SAC #doesn't work   #
# from stable_baselines3.ppo.ppo import PPO #works!         #
# from stable_baselines3.a2c.a2c import A2C #doesn't work   #
# from stable_baselines3.dqn.dqn import DQN #works!         #only discrete
# from stable_baselines3.ddpg.ddpg import DDPG #slow af     #
# from stable_baselines3.td3.td3 import TD3 #didn't test      #actor loss grows

env = gym.make("CarRacing-v0")#, render_mode="human")
model = SAC("CnnPolicy", env, verbose=1, seed=34)#, batch_size=8, buffer_size=500_000)
# model=model.load('sacMountainCarTest.pt', env)
model.learn(total_timesteps=10_000_000)
model.save('carRacing.pt')
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(4000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()