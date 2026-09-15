# Running the game on the full fly brain

The game in `index.html` runs a 400-cell learning circuit with hand-set weights. This file is the plan for swapping that circuit out for the whole-brain model built from the FlyWire connectome, about 139,000 spiking neurons, with learning added to the mushroom body so the machine can actually train the fly.

Every command, file name, count and timing below was checked against the repos and data on 2026-09-14. Recheck versions before a long run.

## What exists

- **Shiu et al. brain model.** [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model), MIT license, Brian2. It drives chosen neurons with Poisson input and records every spike.
- **Eon Systems port.** [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain), GPL-2.0-or-later. The same model on Brian2, Brian2CUDA, PyTorch, NEST GPU and GeNN, with FlyWire version 783 data files (`2025_Connectivity_783.parquet`, `2025_Completeness_783.csv`).
- **Cell labels.** [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations), file `supplemental_files/Supplemental_file1_neuron_annotations.tsv`. 139,248 neurons, root IDs from materialization 783. The repo states no license.
- **Fly body.** NeuroMechFly, `pip install flygym`, [NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym). Latest release v2.1.0.

Neither brain model learns. Both set every synapse once from the connectome and never change it. Eon's embodied fly syncs brain and body every 15 ms, and their write-up links no code for that integration.

## 1. Run the frozen brain

```bash
git clone https://github.com/eonsystemspbc/fly-brain.git
cd fly-brain
conda env create -f environment.yml
conda activate brain-fly
python main.py --brian2-cpu --t_run 1 --n_run 1 --no_log_file
```

Eon tests on Linux with an NVIDIA GPU and CUDA 12. The Brian2 CPU backend needs no GPU.

The experiment API is easiest to read in the Shiu repo's `example.ipynb`:

```python
from model import run_exp
from model import default_params as params

run_exp(exp_name='sugarR', neu_exc=neu_sugar, **config)
```

`neu_sugar` there is a list of 21 FlyWire root IDs. The feeding motor neuron MN9 is `720575940660219265`.

**Done when** MN9 fires during sugar stimulation and stays quiet without it.

The notebook's `config` loads version 630 files (`2023_03_23_connectivity_630_final.parquet`). Switch to the 783 files so IDs match the cell labels. A root ID changes when someone edits that neuron between versions. The sugar and MN9 IDs I spot-checked exist in the 783 table, but a stale ID fails silently, so check every ID you stimulate or read against `Completeness_783.csv`.

## 2. Find the learning circuit

Select from the annotation table:

| Role in the game | Selector |
|---|---|
| Cue code | `cell_class == "Kenyon_Cell"` (5,177 cells, 2,597 right and 2,580 left) |
| Approach and avoid outputs | `cell_class == "MBON"` (96 cells) |
| Reward and omission dopamine | `cell_class == "DAN"`, `cell_type` starting `PAM` or `PPL1` (331 DANs in all) |
| Sucrose payout | `cell_class == "gustatory"` and `cell_sub_class == "sugar/water"` |
| Odor cue input | `cell_class == "ALPN"` (685 projection neurons) |

The labels give type numbers (`MBON01`, `MBON11`, `PAM01`, `PPL101`). They don't say which mushroom body compartment each type covers, and the learning rule depends on it: a dopamine neuron only depresses Kenyon cell synapses onto the output neurons in its own compartment. Take the type-to-compartment map from [Li et al. 2020](https://doi.org/10.7554/eLife.62576) and [Aso et al. 2014a](https://doi.org/10.7554/eLife.04577). Take each output neuron's valence from [Aso et al. 2014b](https://doi.org/10.7554/eLife.04580).

**Done when** every MBON and DAN type in use has a compartment and a valence, each with its citation in the table.

## 3. Make the Kenyon cell synapses plastic

Shiu's `create_model` builds every synapse the same way:

```python
syn = Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=params['t_dly'], name='default_synapses')
syn.w = df_con.loc[:,'Excitatory x Connectivity'].values * params['w_syn']
```

`w_syn` is 0.275 mV. Pull the Kenyon cell to MBON rows out of `df_con` into their own synapse group, so they aren't counted twice, and give that group the rule the web game uses. A Kenyon cell spike leaves a decaying eligibility trace. Dopamine arriving in the same compartment converts the trace into depression. Weights drift back toward their connectome value. A starting sketch, untested:

```python
# add to the neuron equations: dopamine level on each neuron
#   dda/dt = -da / tau_da : 1

kc_mbon = Synapses(neu, neu,
    model='''w : volt
             w0 : volt
             delig/dt = -elig / tau_e : 1 (clock-driven)''',
    on_pre='g += w; elig += 1',
    delay=params['t_dly'])
kc_mbon.run_regularly(
    'w = clip(w - eta * elig * da_post * w * dt/ms + rho * (w0 - w) * dt/ms, 0*mV, w0)',
    dt=1*ms)

# DAN -> MBON pairs come from the compartment table in step 2, not from connectome synapses
dan_to_mbon = Synapses(neu, neu, on_pre='da_post += 1')
```

`tau_e`, `tau_da`, `eta` and `rho` are free parameters. Fit them to published conditioning curves before using them for gambling.

**Trap:** `run_exp` runs 30 independent trials in parallel, and `run_trial` calls `create_model` every time, so each trial starts from a fresh brain. Learning needs one network, built once and advanced with `net.run(...)` pull by pull. Write `kc_mbon.w` to disk after every block so a crash doesn't erase a trained fly.

**Done when** plain conditioning works before any gambling. Pair an odor with sugar about 10 times and the odor alone should shift MBON output toward approach. Present the odor without sugar and the shift should fade.

## 4. Build the machine as a closed loop

One pull:

1. **Cue.** Poisson input to the projection neurons for that lever's odor, 1 s. Each lever gets its own set.
2. **Choice.** Read approach and avoid MBON rates during the cue. The web game picks by softmax over approach minus avoid. Eon steered their body with `DNa01` and `DNa02`, which carry those labels in the annotation table, if you'd rather read the choice out of descending neurons.
3. **Payout.** Poisson input to the `sugar/water` gustatory neurons for the sucrose lever, or straight into the PAM types for the opto lever. For shock, drive PPL1.
4. **Carry over.** Weights persist into the next pull.

`trial()` in `index.html` is the reference. Keep its schedule knobs (win odds, payout size, near-miss rate, delay, shock) and its measurements (share against food, share under shock, dead-machine play, consolidation) so full-brain numbers line up with the table on the page. A near-miss becomes a cue that shares part of its projection-neuron input with the win cue.

## 5. Budget the compute

Eon's `data/benchmark-results.csv` has Brian2 on CPU at 2.7 s of compute for 1 s of brain time (one run), and 269 s for 100 s. Brian2CUDA took 11.9 s for 1 s in the same file.

Each simulated fly in the web game's sweep plays about 1,000 pulls. 9 schedules × 8 flies × 1,000 one-second pulls is 72,000 s of brain time. At 2.7× that comes to roughly 54 hours on Eon's CPU setup before plasticity overhead. Start with 2 flies per schedule and 0.5 s cues. Each fly is an independent network, so run flies as separate processes.

## 6. Add the body (optional)

`pip install flygym`. In v2.1.0 the code lives in `src/flygym` and `src/flygym_demo`, and a turning controller is at `src/flygym_demo/complex_terrain/turning_controller.py`. `OdorArena` is in `flygym/arena/sensory_environment.py` as of v1.2.1. Confirm v2 still ships it before building on it.

With a body the lever becomes a place: the fly walks into the arm that smells like the lever. Nothing from Eon's brain-body loop is public, so this step means writing the 15 ms sync yourself.

## 7. Bring the results back to the page

A browser can't run 139,000 spiking neurons live. Export every pull (lever, outcome, PAM and PPL1 rates, MBON rates, mean Kenyon cell to MBON weight per lever) to JSON, and add a full-brain mode to `index.html` that replays those runs through the same reels, charts and sweep table. The 400-cell model stays as the live mode.

## Licenses

Anything copied from `eonsystemspbc/fly-brain` is GPL-2.0-or-later and makes the repo that contains it GPL. The Shiu model is MIT. The annotation repo states no license, so cite [Schlegel et al. 2024](https://www.nature.com/articles/s41586-024-07686-5) when using it. This repo has no license yet.
