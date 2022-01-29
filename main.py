from random import randint, random
import matplotlib.pyplot as plt
import numpy as np
import math

class Machine:

    def __init__(self, N):
        self.N = N
        self.distribution = generate_beta(N=self.N)
        self.peak = 1
        self.area_proportion = 1 # graph area / plot area = 1 / self.peak.
        self.trials = 0
        self.successes = 0
        self.probability = random()
        self.p = str(self.probability)
        self.estimate = 0.5

    def get_estimate(self):
        x_of_largest = np.argmax(self.distribution, axis=0)
        self.estimate = x_of_largest / self.N
        self.peak = self.distribution[x_of_largest]
        self.area_proportion = 1 / self.peak

    def trial(self):
        self.trials += 1
        if random() <= self.probability:
            self.successes += 1
        self.distribution = generate_beta(total=self.trials, success=self.successes, N=self.N)
        self.get_estimate()


# Bin(n,p) point probabilities for probability x.
def binomial(n, k, x):
    return x**k*(1-x)**(n-k)

# generates a *numerically* normalised beta distribution.
def generate_beta(total=0, success=0, N=100):
    unnormalised = binomial(total, success, np.linspace(0,1,N))
    return unnormalised / sum(unnormalised) / N

# generates the plot grid setup.
def gen_grid(n):
    for i in range(1,10):
        if i**2 >= n:
            return [(x,y) for x in range(i) for y in range(i)][:n]

def pick_best(machines):
    N = machines[0].N
    best = [0,0]
    c = 0
    for index, machine in enumerate(machines):
        while True: # this loop is very slow if the distribution is narrow.
            c += 1
            x, y = randint(0,N-1), (random() * machine.peak)
            if y <= machine.distribution[x]:
                best = [index, x] if x > best[1] else best
                break
    return best[0]

def main(n, iterations, sample_rate, tick_rate, hide_history):

    from IPython.display import clear_output

    intervals = sample_rate # how many points we approximate the distributions on.
    tick_rate = tick_rate
    X = np.linspace(0,1,intervals)
    
    machines = [Machine(intervals) for x in range(n)]
    machine_pos = gen_grid(n)
    plot_size = math.ceil(math.sqrt(n))
    figure, ax = plt.subplots(plot_size, plot_size, num="Multi-arm bandit solver")

    loops = iterations
    c=0
    while c < loops:

        for machine, pos in zip(machines, machine_pos):
            
            if hide_history:
                ax[pos[0], pos[1]].clear()
            ax[pos[0], pos[1]].plot(X, machine.distribution)
            ax[pos[0], pos[1]].set_yticks([])
            ax[pos[0], pos[1]].tick_params(axis="x",direction="in", pad=-15)
            ax[pos[0], pos[1]].set_title(f"p = {machine.p[:6]}, est = {str(machine.estimate)[:6]}")
            machines[pick_best(machines)].trial()
        
        clear_output(wait=True)
        plt.pause(tick_rate)

        c+=1

        figure.suptitle(f"Iteration {c}")

    plt.show()

# much faster- computes all end-results and prints out graph.
def main_not_animated(n, iterations, sample_rate):

    intervals = sample_rate # how many points we approximate the distributions on.
    X = np.linspace(0,1,intervals)
    
    # a single machine: (intervals x Y values, true probability, total plays, # of successes)
    machines = [Machine(intervals) for x in range(n)]
    machine_pos = gen_grid(n)
    plot_size = math.ceil(math.sqrt(n))
    figure, ax = plt.subplots(plot_size, plot_size, num="Multi-arm bandit solver")

    # iterate the machines.
    loops = iterations
    for i in range(loops):
        for machine, pos in zip(machines, machine_pos):
            machines[pick_best(machines)].trial()

    # draw results.
    for machine, pos in zip(machines, machine_pos):
        ax[pos[0], pos[1]].plot(X, machine.distribution)
        ax[pos[0], pos[1]].set_yticks([])
        ax[pos[0], pos[1]].tick_params(axis="x",direction="in", pad=-15)   
        ax[pos[0], pos[1]].set_title(f"p = {machine.p[:6]}, est = {str(machine.estimate)[:6]}")
    
    plt.show()

if __name__ == "__main__":

    n = 9 # number of machines at once
    iterations = 200 # iteration will stop after this many plays.
    tick_rate = 0.05 # time in seconds between ticks.
    sample_rate = 200 # granularity of the functions. higher is more precise.
    hide_history = True # only show one curve at a time for each plot.
    animated = True # by default.

    if input("Multi-arm Bandit Solver.\n=====================\nPress enter to input settings,\nwrite 'default' to use default settings.\n>").lower() == "default":
        main(n, iterations, sample_rate, tick_rate, hide_history)
    
    choices = {n: input("number of machines? (at least 2) "), iterations: input("iterations? "), sample_rate: input("sample rate? (100-200+ recommended) "), animated: input("animated? y/n ")}
    if choices[animated][0].lower() == "y":
        hide_history = (input("show history? y/n ")[0].lower() == "n")
    else:
        animated = False
    n = int(choices[n].strip())
    iterations = int(choices[iterations].strip())
    sample_rate = int(choices[sample_rate].strip())

    if animated:
        main(n, iterations, sample_rate, tick_rate, hide_history)
    else:
        main_not_animated(n, iterations, sample_rate)