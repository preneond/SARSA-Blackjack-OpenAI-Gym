from abc import ABC, abstractmethod

import numpy as np

from blackjack import BlackjackEnv
from blackjack import BlackjackObservation

class AbstractAgent(ABC):
    """
    Abstract base class for agents in the homework. Provides two fields: env for the environment and number_of_epochs.
    """
    ll_ns = np.zeros((25, 25))
    ll_data = np.zeros((25, 25))

    def __init__(self, env: BlackjackEnv, number_of_epochs: int):
        """
        Initializes the agent.
        :param env: The environment in which the agent plays Blackjack.
        :param number_of_epochs: Number of epochs to train on.
        """
        self.env = env
        self.number_of_epochs = number_of_epochs
        super().__init__()

    @abstractmethod
    def train(self):
        """
        This method should train the agent by repeatedly playing the game.
        :return: None.
        """
        pass

    def store_heatmap_data(self, observation, action):
        ll1, ll2 = observation.player_hand.value(), observation.dealer_hand.value()
        ll_count = self.ll_ns[ll1, ll2]
        self.ll_data[ll1, ll2] = (ll_count * self.ll_data[ll1, ll2] + action) / (ll_count + 1)
        self.ll_ns[ll1, ll2] += 1


    def create_state1(self, observation: BlackjackObservation) -> (int, int):
        return observation.player_hand.value(), observation.dealer_hand.value()

    def create_state2(self, observation: BlackjackObservation):
        p1_val = []
        p2_val = []
        for c1, c2 in zip(observation.player_hand.cards, observation.dealer_hand.cards):
            p1_val.append(c1.value())
            p2_val.append(c2.value())
        return p1_val.sort(), p2_val.sort()

    def create_state3(self, observation: BlackjackObservation):
        return (observation.player_hand.value(), len(observation.player_hand.cards)), \
               (observation.dealer_hand.value(), len(observation.dealer_hand.cards))