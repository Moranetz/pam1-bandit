# PAM-1 Bandit

A model of a fruit fly's learning circuit playing a three-lever slot machine.

Play it: https://moranetz.github.io/pam1-bandit/

In March 2026 Eon Systems ran all 139,255 neurons of an adult fruit fly brain inside a simulated body. The synapses in that model are frozen. Pay it every time it pulls a lever and it pulls exactly as often on the thousandth pull as on the first.

This repo is the part that would have to be added. It models the mushroom body, where a fly learns what pays. 400 Kenyon cells feed an approach pathway and an avoid pathway through synapses that can only weaken. Dopamine arriving while a Kenyon cell is active weakens that cell's synapse. Reward neurons (PAM) act on the avoid synapses and omission neurons (PPL1) act on the approach synapses. The output neurons feed back onto the dopamine neurons, so dopamine fires on surprise.

The three levers:

- **Sucrose** pays sugar on a win.
- **Opto-PAM** fires the reward dopamine neurons directly with a red-light pulse. That dopamine arrives whether or not the fly expected it, which is Redish's 2004 addiction model moved into fly circuitry.
- **Food cup** pays a small amount on every pull.

## What the sweep found

Each schedule trains 8 fresh flies for 300 pulls. The fly then chooses between the gamble and the food cup, and after that a shock is added to every gamble pull.

| Schedule | Picks it over food | With shock 0.4 | Consolidated memory |
|---|---|---|---|
| Sugar, win 100%, pays 0.5 | 62% | 34% | 0.01 |
| Sugar, win 25%, pays 2.0 | 60% | 51% | 0.94 |
| Opto, fires 25%, strength 2.0 | 58% | 53% | 0.96 |
| Sugar, win 50%, pays 3 s late | 34% | 21% | 0.00 |

Direct stimulation of the dopamine neurons ranked first. A sugar machine that pays 2.0 on one pull in four came within two points of it under shock. The sure thing stops surprising the fly within a few dozen pulls. Almost nothing consolidates, and under shock the fly picks it on 34% of pulls.

The page runs the full sweep of 9 schedules on load, and every slider feeds the live fly.

## Limits

- Weights are hand-set. None come from the FlyWire connectome, and the parameters are plausible without being fitted to data.
- A real fly has about 2,000 Kenyon cells per hemisphere. This model has 400.
- Nobody has shown a near-miss effect in flies. Here a near-miss is a cue that shares 60% of its Kenyon cells with the win cue.
- Consolidated memories never fully extinguish in this model. The size of the leftover pull cycle on an unpaid machine depends on a recovery rate I picked.

## Run it

One HTML file with no build step. Open `index.html` in a browser.

## Sources

- Bennett, Philippides & Nowotny 2021, [Learning with reinforcement prediction errors in a model of the Drosophila mushroom body](https://www.nature.com/articles/s41467-021-22592-4)
- Springer & Nawrot 2021, [A mechanistic model for reward prediction and extinction learning in the fruit fly](https://www.eneuro.org/content/8/3/ENEURO.0549-20.2021)
- Redish 2004, [Addiction as a computational process gone awry](https://www.science.org/doi/10.1126/science.1102384)
- [Dopaminergic systems create reward seeking despite adverse consequences](https://www.nature.com/articles/s41586-023-06671-8), Nature 2023
- [Dual roles of Drosophila reward-encoding dopamine neurons](https://pmc.ncbi.nlm.nih.gov/articles/PMC12661156/)
- [Eon Systems fly-brain](https://github.com/eonsystemspbc/fly-brain)
