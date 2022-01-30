# Multi-Arm Bandit Solver

This program implements and animates the [Thompson sampling](https://en.wikipedia.org/wiki/Thompson_sampling) method through matplotlib in python and illustrates how it can be used to figure out the hidden probabilities of *n* slot machines.

An alternate `custom_solver.py` takes a `data.txt` file as input, with as many probabilities of as many machines as you like, and solves it substantially faster,, without animating. You can generate arbitrarily long random data using `generate_random_data.py`, included in the repository.

## Config and `defaults.json`
At the beginning, you will be prompted for whether you want to use the default settings found in `defaults.json`, or if you want to input your own. If you input your own, they will be saved as the new defaults. The relevant settings are:

    number of machines - how many slot machines we simulate
    plays per iteration - how many times we play each time the plots get updated
    total iterations - how many rounds total
    sample rate - how precise the probability distributions are
    animated - see the action or skip to the end
    tick rate - maximum speed of the animation
    show history - curves get overlaid on top of old ones

if you want the program to run as fast as possible in the animated mode, just set tick rate to 0.005 or some other very small decimal.

History off with 9 machines, 9 plays per iteration
![](https://i.imgur.com/4Go3kBS.png)

History on with 9 machines, 9 plays per iteration
![](https://i.imgur.com/bb02leY.png)

## Custom Solver
The format of the `data.txt` file is decimal probabilities (0 to 1) separated by commas, such as '0.5,0.7,0.2,0.4,0.8', which generates 5 machines with the corresponding true, hidden probabilities of paying out. `generate_random_data.py` makes this a lot easier.

## The data and process
the *n* machines are shown in a grid with each machine's respective *true probability* of paying out, **p**, and the current estimate according to the beta distribution, **est**.

Each distribution is drawn by taking the corresponding slot machine's current win rate to the binomial distribution's probability mass function for each x between 0 and 1 and then normalising this data with the numerical integration of the graph.

We then pick a random value in each distribution's *area*, and play the machine whose x-value was the highest- this guarantees that we *favour* the better-performing machines, but do not neglect the others, as having a very spread out beta distribution means our certainty of the estimate is low.

Once the distribution becomes very concentrated, it becomes increasingly unlikely to select a value very far from the estimate, which leads to all the machines eventually converging to dirac delta distributions- at least theoretically. In practice, the program will crash due to division by zero.

## Many machines with `custom_solver.py`

The custom solver can deal with *many* machines. Once you go past 1000 or so, however, the Thompson sampling process mostly ignores a good chunk of them.

Custom solver without labels, 400 machines, 10000 iterations.
![](https://i.imgur.com/d0NXXl7.png)