import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
from random import randint, random
import math
import json
import os

# loads the .json data in.
class Settings:
    def __init__(self):
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.directory = os.listdir(self.path)
        if "dev_defaults.json" in self.directory:
            self.config_file = self.path + "/dev_defaults.json"
        elif "defaults.json" in self.directory:
            self.config_file = self.path + "/defaults.json"
        
    def load_custom_settings(self):
        if self.config_file != None:
            with open(self.config_file) as file:
                data = file.read()
            s = json.loads(data)
            try:
                self.n = s["number_of_machines"]
                self.iterations = s["iterations"]
                self.plays_per_iteration = s["plays_per_iteration"]
                self.tick_rate = s["tick_rate"]
                self.sample_rate = s["sample_rate"]
                self.hide_history = bool(int(s["hide_history"]))
            except:
                print("erroneous settings. loading defaults...")
                self.load_defaults()
        else:
            self.load_defaults()
    
    def load_defaults(self):
        self.n = 9
        self.iterations = 200
        self.plays_per_iteration = self.n
        self.tick_rate = 0.05
        self.sample_rate = 200
        self.hide_history = True

# globally accessible.
settings = Settings()
settings.load_custom_settings()

# models a single slot machine.
class Machine:

    def __init__(self, N):
        self.N = N
        self.distribution = generate_beta(N=self.N)
        self.peak = 1
        self.trials = 0
        self.successes = 0
        self.probability = random()
        self.p = str(self.probability)
        self.estimate = 0.5

    def get_estimate(self):
        x_of_largest = np.argmax(self.distribution, axis=0)
        self.estimate = x_of_largest / self.N
        self.peak = self.distribution[x_of_largest]

    def trial(self):
        self.trials += 1
        if random() <= self.probability:
            self.successes += 1
        self.distribution = generate_beta(total=self.trials, success=self.successes, N=self.N)
        self.get_estimate()

# saves new settings into defaults.json (or dev_defaults.json)
def save_defaults():
    with open(settings.config_file, "r+") as file:
        config = json.load(file)
        config["number_of_machines"] = settings.n
        config["iterations"] = settings.iterations
        config["plays_per_iteration"] = settings.plays_per_iteration
        config["tick_rate"] = settings.tick_rate
        config["sample_rate"] = settings.sample_rate
        config["hide_history"] = int(settings.hide_history)
    with open(settings.config_file, "w") as file:
        json.dump(config, file, indent=2)

# Bin(n,p) point probabilities for probability x.
def binomial(n, k, x):
    return x**k*(1-x)**(n-k)

# generates a *numerically* normalised beta distribution.
def generate_beta(total=0, success=0, N=100):
    unnormalised = binomial(total, success, np.linspace(0,1,N))
    return unnormalised * N / np.sum(unnormalised)

# generates the plot grid setup.
def gen_grid(n):
    for i in range(1,10):
        if i**2 >= n:
            return [(x,y) for x in range(i) for y in range(i)][:n]

def pick_best(machines):
    best = [0,0]
    for index, machine in enumerate(machines):
        while True: # this loop is very slow if the distribution is narrow.
            x, y = randint(0,settings.sample_rate-1), random() * machine.peak
            if y <= machine.distribution[x]:
                best = [index, x] if x > best[1] else best
                break
    return best[0]

def main():

    X = np.linspace(0, 1, settings.sample_rate)
    
    machines = [Machine(settings.sample_rate) for x in range(settings.n)]
    machine_pos = gen_grid(settings.n)
    plot_size = math.ceil(math.sqrt(settings.n))
    figure, ax = plt.subplots(plot_size, plot_size, num="Multi-arm bandit solver")

    # sorting the machines makes it easier to conceptualise how the convergence happens.
    machines.sort(reverse=True, key=lambda a: a.probability)

    # draw first all the initial plots.
    for machine, pos in zip(machines, machine_pos):
        ax[pos[0], pos[1]].plot(X, machine.distribution)
        ax[pos[0], pos[1]].set_yticks([])
        ax[pos[0], pos[1]].tick_params(axis="x",direction="in", pad=-15)
        ax[pos[0], pos[1]].set_title(f"p = {machine.p[:6]}, est = {str(machine.estimate)[:6]}", fontsize=10)

    for i in range(settings.iterations):

        # so we only draw plots that *change*. saves a lot of computing power.
        machines_played_this_round = []
        figure.suptitle(f"Iteration {i+1}")

        # first play all the games for this iteration.
        for i in range(settings.plays_per_iteration):
            best_machine_index = pick_best(machines)
            machines_played_this_round.append(best_machine_index)
            machines[best_machine_index].trial()

        # draw relevant plots.
        for (machine, pos) in [(machines[index], machine_pos[index]) for index in machines_played_this_round]:
            if settings.hide_history:
                ax[pos[0], pos[1]].clear() # this is stupidly slow.
            ax[pos[0], pos[1]].plot(X, machine.distribution)
            ax[pos[0], pos[1]].set_yticks([])
            ax[pos[0], pos[1]].tick_params(axis="x",direction="in", pad=-15)
            ax[pos[0], pos[1]].set_title(f"p = {machine.p[:6]}, est = {str(machine.estimate)[:6]}", fontsize=10)

        clear_output(wait=True)
        plt.pause(settings.tick_rate)

    plt.show()

if __name__ == "__main__":

    if input("Multi-arm Bandit Solver.\n=====================\nuse default settings? y/n ")[0].lower() == "y":
        main()

    # extremely rudimentary error-handling.
    while True:
        try:
            choices = {"n": input("number of machines? (at least 2) "), 
                    "plays_per_iteration": input("plays per iteration? (at least 1) "), 
                    "iterations": input("total iterations? "), 
                    "sample_rate": input("sample rate? (100-200+ recommended) "),
                    }

            settings.tick_rate = float(input("tick rate? (in seconds) "))
            settings.hide_history = (input("show history? y/n ")[0].lower() == "n")

            # rest of the settings set.
            settings.n = int(choices["n"].strip())
            settings.iterations = int(choices["iterations"].strip())
            settings.sample_rate = int(choices["sample_rate"].strip())
            settings.plays_per_iteration = int(choices["plays_per_iteration"].strip())
            break
        except:
            print("at least one of the given values was of the incorrect format.")
            continue

    # update defaults.json here.
    save_defaults()
    main()