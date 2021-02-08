# import matplotlib
# matplotlib.use('Agg')
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()


def evaluate(rewards: List[float], ll_data):
    np.set_printoptions(precision=5)

    print("Rewards:")
    print(rewards)
    print("Simple moving average:")
    # if you reuse this code, you should change the parameters
    print(simple_moving_average(rewards, 10))
    print("Exponential moving average")
    print(exponential_moving_average(rewards, 0.02))
    print("Average")
    print(np.sum(rewards) / len(rewards))

    plt1 = exponential_moving_average(rewards, 0.01)
    plt2 = exponential_moving_average(rewards, 0.1)
    plot_series_with_bg(plt1, plt2)

    plot_heatmap(ll_data)
    # plot_heatmap(ll_data[4:23,2:12])
    # plot_series(exponential_moving_average(rewards, 0.01))


# check Wikipedia: https://en.wikipedia.org/wiki/Moving_average
def simple_moving_average(x: List[float], n: int) -> float:
    mean = np.zeros(len(x) - n + 1)
    tmp_sum = np.sum(x[0:n])
    for i in range(len(mean) - 1):
        mean[i] = tmp_sum
        tmp_sum -= x[i]
        tmp_sum += x[i + n]
    mean[len(mean) - 1] = tmp_sum
    return mean / n


# check Wikipedia: https://en.wikipedia.org/wiki/Moving_average
def exponential_moving_average(x: List[float], alpha: float) -> float:
    mean = np.zeros(len(x))
    mean[0] = x[0]
    for i in range(1, len(x)):
        mean[i] = alpha * x[i] + (1.0 - alpha) * mean[i - 1]
    return mean


def heatmap(x: List[float]):
    pass


# you can use this function to get a plot
# you need first to install matplotlib (conda install matplotlib)
# and then uncomment this function and lines 1-3
def plot_series(arr):
    plt.plot(arr)
    plt.xlabel = 'Number of Epochs'
    plt.ylabel = 'Reward'
    plt.show()


def plot_series_with_bg(arr1, arr2):
    plt.plot(arr1, alpha=1.0, color='b')
    plt.plot(arr2, alpha=0.1, color='b')
    plt.ylim(-1.0,1.0)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Reward')
    plt.show()


def plot_heatmap(data):
    ax = sns.heatmap(1-data, cmap='rocket_r')
    ax.set_aspect('equal')
    plt.ylim(4, 22)
    plt.xlim(2, 12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.xlabel('Dealer\'s sum of cards')
    plt.ylabel('Player\'s sum of cards')
    plt.title('Probability that player will stop')
    plt.show()