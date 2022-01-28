# multi-arm-bandit-solver

This program implements and animates the Thompson sampling method through matplotlib in python and illustrates how it can be used to figure out the hidden probabilities of *n* slot machines.

## Default config and settings
Default values are

    9 machines
    0.05s tick rate
    200 point sampling for distributions
    200 max plays
    history off - show only one curve per graph, instead of all history

Additionally, an unanimated version is available by exchanging the function main() -> main_not_animated(). All the settings can be changed at the bottom of the file. Note: history *on* is significantly faster- turns out wiping the history each step is computationally expensive. Experiment also with sample_rate.

| History off with 9 machines |
| --- |
![](https://i.imgur.com/GppNKwE.png)

| History on with 9 machines |
| --- |
![](https://i.imgur.com/bxaNHFy.png)

## The data and process
the *n* machines are shown in a grid with each machine's respective *true probability* of paying out, **p**, and the current estimate according to the beta distribution, **est**.

Each distribution is drawn by taking the corresponding slot machine's current win rate to the binomial distribution's probability mass function for each x between 0 and 1 and then normalising this data with the numerical integration of the graph.

We then pick a random value in each distribution's *area*, and play the machine whose x-value was the highest- this guarantees that we *favour* the better-performing machines, but do not neglect the others, as having a very spread out beta distribution means our certainty of the estimate is low.

Once the distribution becomes very concentrated, it becomes increasingly unlikely to select a value very far from the estimate, which leads to all the machines eventually converging to dirac delta distributions- at least theoretically. In practice, the program will crash due to division by zero.

## Requisite packages
The non-standard library imports are

    Numpy
    Matplotlib.pyplot
    IPython.display

## About
This was a little project I wrote in an evening as I study statistics and probability, I may update it later with further improvements and/or features, such as better controls. Thank you.

| History on with 25 machines |
| --- |
![](https://i.imgur.com/K6OTQQB.png)