# A cadmium-based quantification of marine organic carbon burial during the PETM
We use the δ<sup>114</sup>Cd of PETM age shales to quantify the burial of marine organic carbon during the PETM.

# Data
Our measured cadmium isotope data are available in the `data/input` folder.

# Code
The code used to generate the figures in the manuscript is available in the `code` folder. The code is written in Python and requires the following packages:
- `numpy`
- `pandas`
- `matplotlib`
- `cdbury`
These are already installed if running this as a CodeOcean capsule.


## Running
If running this inside the CodeOcean capsule, all scripts are run automatically. Otherwise, run the scripts as follows.

## Forward model
`run_forward.py` runs three forward models of the cadmium system in the ocean, one for steady state, one for an increase in marine organic carbon burial, and one for a decrease in marine organic carbon burial. The models are run using the `cdbury` [package](https://pypi.org/project/CdBURY/).

`plot_forward_simulations.py` is a Jupyter notebook that plots age vs δ<sup>114</sup>Cd for the three forward models.

## Inverse model
To calculate the burial flux of marine organic carbon during the PETM, we use `cdbury` in inverse mode. This is done in `run_invert_organic.py` - which produces several data files in the `/results` folder. The main outputs are `results/d114cd_gp.json` (a Gaussian Process fit to our δ<sup>114</sup>Cd data), and `results/invert_organic_prior_results.json` which is an array of `cdbury` simulations - each containing all properties of the cadmium system in the ocean.

Alternatively, to account for possible time warping of the PETM onset, run `run_invert_organic_timewarp.py`, which implements a time warp on the age model before running the inverse model. This produces results in `results/d114cd_gp_timewarp.json` and `results/invert_organic_prior_results_timewarp.json`.

To interpret the results, run `calculate_invert_organic_metrics.py` and `calculate_organic_burial.py`, which produce additional data files in the `/results` folder (by default targeting the outputs of `run_invert_organic.py`, but can be modified to target the time warped results).

Results are plotted in the Jupyter notebook by `plot_d114cd_data.ipynb`, `plot_invert_organic.ipynb`, and `plot_total_burial.ipynb`.