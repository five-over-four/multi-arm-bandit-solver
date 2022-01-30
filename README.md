# Multi-Arm Bandit Solver

This program implements and animates the [Thompson sampling](https://en.wikipedia.org/wiki/Thompson_sampling) method through matplotlib in python and illustrates how it can be used to figure out the hidden probabilities of *n* slot machines.

An alternate version (custom_solver.py) takes a text file as input, with as many probabilities of as many machines as you like, and solves it substantially faster- though without animating. The format for the data.txt is p1,p2,p3,p4,...,pn with pk between 0 and 1.0 for each k. An example is provided in the directory.

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

History off with 9 machines
![](https://i.imgur.com/GppNKwE.png)

History on with 9 machines
![](https://i.imgur.com/bxaNHFy.png)

## Custom Solver
Place custom probabilities into `data.txt` and run custom_solver.py. You will be prompted for a number of iterations and sample rate. Recommended sample rate is at least 100-200. If you eg. want 10 machines with probabilities 10%, 20%, ... , 100%, your file will look like '0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0', without the quotes.

## The data and process
the *n* machines are shown in a grid with each machine's respective *true probability* of paying out, **p**, and the current estimate according to the beta distribution, **est**.

Each distribution is drawn by taking the corresponding slot machine's current win rate to the binomial distribution's probability mass function for each x between 0 and 1 and then normalising this data with the numerical integration of the graph.

We then pick a random value in each distribution's *area*, and play the machine whose x-value was the highest- this guarantees that we *favour* the better-performing machines, but do not neglect the others, as having a very spread out beta distribution means our certainty of the estimate is low.

Once the distribution becomes very concentrated, it becomes increasingly unlikely to select a value very far from the estimate, which leads to all the machines eventually converging to dirac delta distributions- at least theoretically. In practice, the program will crash due to division by zero.

## Requisite packages
The non-standard library imports are

    Numpy
    Matplotlib.pyplot
    IPython.display *

\* only required for the animated mode.

## About
This was a little project I wrote in an evening as I study statistics and probability for fun.

## Extra Images

History on with 25 machines
![](https://i.imgur.com/K6OTQQB.png)

Custom solver without labels, 81 machines.
![](https://i.imgur.com/ojrv7Hc.png)