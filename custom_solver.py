from random import randint, random
import matplotlib.pyplot as plt
import numpy as np
import math
import os

class Machine:

    def __init__(self, sample_rate, probability):
        self.sample_rate = sample_rate
        self.distribution = generate_beta(N=self.sample_rate)
        self.peak = 1
        self.trials = 0
        self.successes = 0
        self.probability = probability
        self.estimate = 0.5

    def get_estimate(self):
        x_of_largest = np.argmax(self.distribution, axis=0)
        self.estimate = x_of_largest / self.sample_rate
        self.peak = self.distribution[x_of_largest]

    def trial(self):
        self.trials += 1
        if random() <= self.probability:
            self.successes += 1
        self.distribution = generate_beta(total=self.trials, success=self.successes, N=self.sample_rate)
        self.get_estimate()


# Bin(n,p) point probabilities for probability x.
def binomial(n, k, x):
    return x**k*(1-x)**(n-k)

# generates a *numerically* normalised beta distribution. note the numerical integration in return statement.
def generate_beta(total=0, success=0, N=100):
    unnormalised = binomial(total, success, np.linspace(0,1,N))
    return unnormalised * N / sum(unnormalised)

# generates the plot grid setup.
def gen_grid(n):
    for i in range(1,1000):
        if i**2 >= n:
            return [(x,y) for x in range(i) for y in range(i)][:n]

def pick_best(machines):
    N = machines[0].sample_rate
    best = [0,0]
    for index, machine in enumerate(machines):
        while True: # this loop is very slow if the distribution is narrow.
            x, y = randint(0,N-1), random() * machine.peak
            if y <= machine.distribution[x]:
                best = [index, x] if x > best[1] else best
                break
    return best[0]

# simulates whole system for a txt file. much faster.
def main(iterations, sample_rate):

    if "data.txt" in os.listdir(os.path.dirname(os.path.realpath(__file__))):
        with open("data.txt") as f:
            data = f.readline().split(",")
        data = [float(item) for item in data]
        n = len(data)
    
    else:
        print("data.txt not found. quitting.")
        return -1

    X = np.linspace(0,1,sample_rate)
    
    machines = [Machine(sample_rate, probability) for probability in data]
    machine_pos = gen_grid(n)
    plot_size = math.ceil(math.sqrt(n))
    figure, ax = plt.subplots(plot_size, plot_size, num="Multi-arm bandit solver")

    # sort the machines by probability for easier viewing.
    machines.sort(reverse=True, key=lambda a: a.probability)

    # iterate the machines.
    for i in range(iterations):
        machines[pick_best(machines)].trial()
    
    # draw results.
    for machine, pos in zip(machines, machine_pos):
        machine.get_estimate()
        ax[pos[0], pos[1]].plot(X, machine.distribution, "b")
        ax[pos[0], pos[1]].set_yticks([])
        ax[pos[0], pos[1]].set_xticks([])
        if n <= 40: # don't draw labels if there's waaaay too many plots
            ax[pos[0], pos[1]].tick_params(axis="x",direction="in", pad=-15)
            ax[pos[0], pos[1]].set_title(f"p = {str(machine.probability)[:6]}, est = {str(machine.estimate)[:6]}", fontsize=10)
    
    plt.show()

if __name__ == "__main__":

    while True:
        try:
            iterations = int(input("How many iterations?")) # total number 
            sample_rate = int(input("Sample rate? Higher is better."))
            break
        except:
            print("Erroneous inputs, try again.")
            continue
        
    main(iterations, sample_rate)