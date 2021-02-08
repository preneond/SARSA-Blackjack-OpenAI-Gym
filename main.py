from gym.wrappers import Monitor

from abstractagent import AbstractAgent
from dealeragent import DealerAgent
from evaluate import *
import gym
from gym import wrappers
from gym.envs.registration import register
from randomagent import RandomAgent
from sarsaagent import SarsaAgent
from tdagent import TDAgent

def get_env() -> Monitor:
    """
    Creates the environment. Check the OpenAI Gym documentation.

    :rtype: Environment of the blackjack game that follows the OpenAI Gym API.
    """
    environment = gym.make('smu-blackjack-v0')
    return wrappers.Monitor(environment, 'smuproject4', force=True, video_callable=False)


if __name__ == "__main__":
    # Registers the environment so that it can be used
    register(
        id='smu-blackjack-v0',
        entry_point='blackjack:BlackjackEnv'
    )
    # ######################################################
    # IMPORTANT: do not modify the code above this line! ###
    # ######################################################
    import os.path
    import numpy as np

    env = get_env()
    number_of_epochs = 50000



    # FLAG the Agent who want you to evaluate
    is_sarsa = True
    is_td = False
    is_dealer = False
    is_random = False

    # FLAG for using the saved rewards from older evaluation
    use_old = False

    rewards_fname = '_rewards.npy'
    ll_fname = '_ll_strategy.npy'
    if is_sarsa:
        rewards_fname = 'data/sarsa' + rewards_fname
        ll_fname = 'data/sarsa' + ll_fname
    elif is_td:
        rewards_fname = 'data/td' + rewards_fname
        ll_fname = 'data/td' + ll_fname
    else:
        rewards_fname = 'data/random' + rewards_fname
        ll_fname = 'data/random' + ll_fname


    episode_rewards = None
    ll_data = None

    if os.path.isfile(rewards_fname) and os.path.isfile(ll_fname) and use_old:
        episode_rewards = np.load(rewards_fname)
        ll_data = np.load(ll_fname)
    else:
        if is_random:
            agent: AbstractAgent = RandomAgent(env, number_of_epochs)
        elif is_dealer:
            agent: AbstractAgent = DealerAgent(env, number_of_epochs)
        elif is_td:
            agent: AbstractAgent = TDAgent(env, number_of_epochs)
        else:
            agent: AbstractAgent = SarsaAgent(env, number_of_epochs)

        agent.train()
        episode_rewards = env.get_episode_rewards()
        ll_data = agent.ll_data
        np.save(rewards_fname, np.array(episode_rewards))
        np.save(ll_fname, np.array(ll_data))


    evaluate(episode_rewards, ll_data)
