#!/usr/bin/env python
# coding: utf-8

import numpy as np
import gym
from ddpg import Agent, Buffer

def main():
    action_high = 2
    action_low = -2
    action_high = np.array([action_high])
    action_low = np.array([action_low])
    buffer_size = 100000

    env = gym.make("Pendulum-v0")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = Agent(state_size, action_size, action_high, action_low)
    buffer = Buffer(buffer_size)
    for i_episode in range(500):
        print("episode: %d" % i_episode)
        state = env.reset()
        total_reward_in_episode = 0
        for t_timesteps in range(env.spec.timestep_limit):
            env.render()
            action = agent.choose_action(state, (1+i_episode*0.05))
            next_state, reward, done, info = env.step(action)
            total_reward_in_episode += reward
            transition = [state, action, next_state, reward, done]
            state = next_state
            buffer.store(transition)
            agent.train(buffer.transitions)
            if (done or t_timesteps == env.spec.timestep_limit - 1):
                print("Episode finish---time steps: %d" % t_timesteps)
                print("total reward: %d" % total_reward_in_episode)
                break

if __name__ == '__main__':
    main()
