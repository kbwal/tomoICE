# TomoICE

**Interior Cargo Estimation in cryo-electron tomograms**

TomoICE estimates how many cargo molecules are packed inside individual hollow protein shells imaged by cryo-electron tomography. For each particle, it fits a known atomic model of the shell into the tomogram, measures the percentage of the shell interior occupied by density, and converts that percentage into a cargo number using the occupancy produced by a single cargo molecule in a simulated density map. Every particle is measured individually in its own tomogram, with no subtomogram averaging.

This repository contains the scripts (not the tomograms or any other large files) used in the paper "Vault Packaging Dynamics: Cargo Loading Stabilizes MVP Conformation for High-Payload Cytokine Delivery in Cancer Therapy." In that paper the shell is the MVP vault (PDB [7PKR](https://www.rcsb.org/structure/7PKR)) and the cargo is IL-2, and this release is configured for that system. See [Using other shells](#using-other-shells) for what to change for a different shell or cargo.

## Scripts

- `fit.py` - takes in the original 4 tomograms along with their corresponding centers (available in `center_coordinates`, manually chosen) and fits an atomic model (7PKR) to that region of the tomogram using point-cloud registration. It outputs a transformation matrix (along with scaling factor required) that says how the model should be translated and rotated to fit the vault in the density.
- `find_thresholds_per_z_slice.py` - calculates MCC thresholds for every z-slice in every tomogram.
- `get_percentage_of_fits_given_slice_threshold.py` - uses the per-z slice thresholds and the fitted model to calculate the percentage occupancy of each vault that is density.
- `find_atomic_model_thresholds.py` - calculates MCC thresholds for every z-slice of a simulated density map of a vault containing a single IL-2 (`atomic_model.mrc`, generated using `molmap` in ChimeraX), in the same way as `find_thresholds_per_z_slice.py`. The vault wall positions come from `fit_atomic_model.cif`, which is 7PKR (vault only, no IL-2) fitted into `atomic_model.mrc`.
- `get_percentage_of_fits_given_slice_threshold_atomic_model.py` - uses those thresholds to calculate the percentage occupancy of the simulated single-IL-2 vault (~0.25%), i.e. how much of the vault interior one IL-2 occupies.
- `plot_overlaid_scale_distributions_incorporating_scale.py` - produces a figure of this data, used in Figure 3(c) of the paper.

## Cargo number

The plotting script converts each vault's percentage occupancy into a cargo number using

```
cargo number = percentage occupancy × scale² / p
```

where `scale` is the x/y scaling factor from that vault's fit and `p` is the percentage occupancy produced by a single IL-2.

## Setup

Requires Python 3.11. Tested on Ubuntu 24.04.4, where installing the dependencies below takes about 3 minutes.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The scripts use relative paths, so run them from a folder laid out as follows:

```
tomos/Position_1_1_SIRT_rec_corrected.mrc   # ... through Position_1_4
7PKR.cif                                    # from https://www.rcsb.org/structure/7PKR
center_coordinates/
atomic_model.mrc                            # only for the atomic model scripts; molmap of a vault + single IL-2 model in ChimeraX; included in this folder
fit_atomic_model.cif                        # only for the atomic model scripts; included in this folder
```

Run the scripts in this order:

1. `python fit.py`
2. `python find_thresholds_per_z_slice.py`
3. `python get_percentage_of_fits_given_slice_threshold.py`
4. `python plot_overlaid_scale_distributions_incorporating_scale.py`

and, separately, for the single-IL-2 atomic model:

1. `python find_atomic_model_thresholds.py`
2. `python get_percentage_of_fits_given_slice_threshold_atomic_model.py`

Fits are written to `fits/tomo_N/regions/regionK/` (`fit.cif` and `fit_metadata.txt`), thresholds to `slice_thresholds/`, and percentage occupancies to `fits/tomo_N/percentage-z-slices-precomputed-thresholds.txt`. The atomic model percentage is written to `atomic_model_percentages/`.

`fits/` already contains the `fit_metadata.txt` files and percentage occupancies from the paper, so `plot_overlaid_scale_distributions_incorporating_scale.py` can be run directly to reproduce Figure 3(c). The fitted models (`fit.cif`) are not included, so steps 2 and 3 require running `fit.py` first. Running `fit.py` overwrites the included `fit_metadata.txt` files, and step 3 overwrites the included percentage files.

Note that `fit.py` uses RANSAC without a fixed random seed, so re-running it can give different fits. Because the fit score varies little across nearby scales, the fitted scale of an individual vault can differ between runs (e.g. 1.17 in one run and 1.29 in another), which can change which expansion group that vault falls into. The fits used in the paper are included in `fits/` and were visually inspected. All later steps are deterministic given the fits.

## Demo

The tomograms are not part of this repository, but two parts of the pipeline run on data included in this folder:

- `python plot_overlaid_scale_distributions_incorporating_scale.py` reads the included `fits/` and writes `incorporating_scale_num_il2_figure.png`, the cargo-number distributions shown in Figure 3(c). It takes about 15 seconds.
- `python find_atomic_model_thresholds.py` followed by `python get_percentage_of_fits_given_slice_threshold_atomic_model.py` reads `atomic_model.mrc` and `fit_atomic_model.cif` and prints `Percentage of vault with density inside: 0.25%`, the occupancy of a single IL-2. Each script takes about 75 seconds.

Run times were measured on an Apple M2 MacBook Air with 16 GB of RAM running macOS 15.7.4.

## Using other shells

The method applies to any hollow protein shell with a known atomic model, such as viral capsids, encapsulins, ferritin cages or virus-like particles. The following settings are specific to the vault and IL-2 and would need to be changed for another system:

- **Shell model**: `7PKR.cif` is loaded in `fit.py`, `find_thresholds_per_z_slice.py` and `get_percentage_of_fits_given_slice_threshold.py`.
- **Cap weighting**: `weight_vault_caps` in `fit.py` up-weights the two vault caps during registration. Remove it or adapt it to the new shell's distinctive features.
- **Expansion scan**: `fit.py` scans an x/y scale from 1.00 to 1.30, matching the vault's expansion. Adjust the range and axes to how the new shell deforms, and update the scale term in the cargo-number formula to match.
- **Box and wall sizes**: `BOX_RADIUS_VOXELS` / `box_radius` (75 voxels) and `WALL_WIDTH_ANGSTROMS` (20 Å) in the threshold and occupancy scripts should fit the new shell's size and wall thickness.
- **Single-cargo reference**: regenerate `atomic_model.mrc` (a `molmap` of the shell with one cargo molecule) and `fit_atomic_model.cif` for the new shell and cargo.
- **Plot groups**: the expansion groups in `plot_overlaid_scale_distributions_incorporating_scale.py` are specific to the paper's figure.

## License

This code is licensed under the GNU General Public License v3.0. See `LICENSE` for details.
