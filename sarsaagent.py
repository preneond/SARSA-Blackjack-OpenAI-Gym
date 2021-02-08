from collections import defaultdict

import numpy as np

from abstractagent import AbstractAgent
from blackjack import BlackjackObservation


class SarsaAgent(AbstractAgent):
    """
    Here you will provide your implementation of SARSA method.
    You are supposed to implement train() method. If you want
    to, you can split the code in two phases - training and
    testing, but it is not a requirement.

    For SARSA explanation check AIMA book or Sutton and Burton
    book. You can choose any strategy and/or step-size function
    (learning rate).
    """
    Q_table = defaultdict(lambda: [0.0, 0.0])
    Ns = defaultdict(lambda: 0)
    gamma = 0.99
    c = 20

    def train(self):
        alpha = lambda state: self.c / (self.c + self.Ns[state] - 1)

        for i in range(self.number_of_epochs):
            if (i + 1) % 1000 == 0:
                print('Epoch #{}'.format(i + 1))

            observation = self.env.reset()
            terminal = False
            state = self.create_state3(observation)  # (observation, terminal)
            self.Ns[state] += 1
            while not terminal:
                # self.env.render()
                action = self.make_step(state)
                self.store_heatmap_data(observation,action)
                observation, reward, terminal, _ = self.env.step(action)

                next_state = self.create_state3(observation)
                next_action = self.make_step(next_state)
                self.Ns[next_state] += 1

                Q_current = self.Q_table[state][action]
                Q_next = self.Q_table[next_state][next_action]

                self.Q_table[state][action] += alpha(state) * (reward + self.gamma * Q_next - Q_current)
                state = next_state

    # using epsilon greedy policy
    def make_step(self, state) -> int:
        epsilon = np.divide(self.c, self.c + self.Ns[state] - 1)
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q_table[state])

    def get_q_value(self, observation: BlackjackObservation, action: int) -> float:
        """
        SARSA - State Action Reward State Action
        Q(s,a) ← Q(s,a) + α(R(s) + γ Q(s',a') − Q(s,a)) where
        a is executed in state s leading to state s' and
        a'is the action actually taken in state s'.

        Implement this method so that I can test your code. This method is supposed to return your learned Q value for
        particular observation and action.

        :param observation: The observation as in the game. Contains information about what the player sees - player's
        hand and dealer's hand.
        :param action: Action for Q-value.
        :return: The learned Q-value for the given observation and action.
        """
        state = self.create_state3(observation)
        return self.Q_table[state][action]
