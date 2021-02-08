from collections import defaultdict

from abstractagent import AbstractAgent
from blackjack import BlackjackObservation


class TDAgent(AbstractAgent):
    """
    Implementation of an agent that plays the same strategy as the dealer.
    This means that the agent draws a card when sum of cards in his hand
    is less than 17.

    Your goal is to modify train() method to learn the state utility function.
    I.e. you need to change this agent to a passive reinforcement learning
    agent that learns utility estimates using temporal diffrence method.
    """

    def train(self):
        c = 20
        self.U = defaultdict(lambda: 1)
        Ns = defaultdict(lambda: 0)
        discount_factor = 0.99

        alpha = lambda state: c / (c + Ns[state] - 1)

        for i in range(self.number_of_epochs):
            if (i + 1) % 1000 == 0:
                print('Epoch #{}'.format(i + 1))

            observation = self.env.reset()
            terminal = False
            last_state = self.create_state3(observation)
            Ns[last_state] += 1
            reward = 0
            while not terminal:
                # render method will print you the situation in the terminal
                # self.env.render()
                action = self.make_step(observation, reward, terminal)
                self.store_heatmap_data(observation, action)
                observation, reward, terminal, _ = self.env.step(action)
                state = self.create_state3(observation)

                Ns[state] += 1
                u_update = alpha(last_state) * (reward + discount_factor * self.U[state] - self.U[last_state])
                self.U[last_state] = self.U[last_state] + u_update
                last_state = state

    def make_step(self, observation: BlackjackObservation, reward: float, terminal: bool) -> int:
        return 1 if observation.player_hand.value() < 17 else 0

    def get_u_value(self, observation: BlackjackObservation) -> float:
        """
        Implement this method so that I can test your code. This method is supposed to return your learned U value for
        particular observation.

        :param observation: The observation as in the game. Contains information about what the player sees - player's
        hand and dealer's hand.
        :return: The learned U-value for the given observation.
        """
        state = self.create_state3(observation)

        return self.U[state]
