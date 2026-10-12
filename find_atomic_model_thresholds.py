from pathlib import Path

import biotite.structure.io as strucio
import mrcfile
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree

TOMO_PATH = Path("./atomic_model.mrc")
FIT_CIF_PATH = Path("./fit_atomic_model.cif")

BOX_RADIUS_VOXELS = 75
WALL_WIDTH_ANGSTROMS = 20.0
WALL_POSITIVE_RADIUS_ANGSTROMS = WALL_WIDTH_ANGSTROMS / 2.0
NEGATIVE_BAND_INNER_ANGSTROMS = WALL_WIDTH_ANGSTROMS * 2.0
NEGATIVE_BAND_OUTER_ANGSTROMS = WALL_WIDTH_ANGSTROMS * 2.5
MIN_POSITIVE_SAMPLES_PER_Z = 10
MIN_NEGATIVE_SAMPLES_PER_Z = 10
SMOOTHING_SIGMA_Z = 2


def get_mrc_file_header_info(file_path):
    with mrcfile.open(file_path, mode="r", permissive=True) as mrc:
        vox_size = mrc.voxel_size
        nx = mrc.header.nx
        ny = mrc.header.ny
        nz = mrc.header.nz
        return float(vox_size["x"]), int(nx), int(ny), int(nz)


def get_vault_coordinates(cif_path):
    """Loads perfectly aligned coordinates directly from the CIF file."""
    atom_array = strucio.load_structure(str(cif_path))
    ca_atoms = atom_array[atom_array.atom_name == "CA"]
    return ca_atoms.coord


def load_section_volume(file_path, region):
    x_start, y_start, z_start, x_end, y_end, z_end = region
    with mrcfile.open(file_path, mode="r", permissive=True) as mrc:
        section = mrc.data[z_start:z_end, y_start:y_end, x_start:x_end]
    return np.array(section)


def compute_mcc_threshold(positive_vals, negative_vals, n_candidates=800):
    positive_vals = np.asarray(positive_vals)
    negative_vals = np.asarray(negative_vals)
    positive_vals = positive_vals[np.isfinite(positive_vals)]
    negative_vals = negative_vals[np.isfinite(negative_vals)]

    if (
        positive_vals.size < MIN_POSITIVE_SAMPLES_PER_Z
        or negative_vals.size < MIN_NEGATIVE_SAMPLES_PER_Z
    ):
        return np.nan

    all_vals = np.concatenate([positive_vals, negative_vals])
    lo = np.percentile(all_vals, 1)
    hi = np.percentile(all_vals, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return np.nan

    positive_sorted = np.sort(positive_vals)
    negative_sorted = np.sort(negative_vals)
    thresholds = np.linspace(lo, hi, n_candidates)

    n_pos = positive_sorted.size
    n_neg = negative_sorted.size
    false_negatives = np.searchsorted(positive_sorted, thresholds, side="left")
    true_positives = n_pos - false_negatives
    true_negatives = np.searchsorted(negative_sorted, thresholds, side="left")
    false_positives = n_neg - true_negatives

    numerator = (true_positives * true_negatives) - (false_positives * false_negatives)
    denominator = np.sqrt(
        (true_positives + false_positives)
        * (true_positives + false_negatives)
        * (true_negatives + false_positives)
        * (true_negatives + false_negatives)
    )

    mcc = np.full(thresholds.shape, -np.inf, dtype=float)
    valid_denominator = denominator > 0
    mcc[valid_denominator] = (
        numerator[valid_denominator] / denominator[valid_denominator]
    )

    sensitivity = true_positives / n_pos
    specificity = true_negatives / n_neg
    balanced_accuracy = 0.5 * (sensitivity + specificity)

    best_mcc = np.max(mcc)
    print("best mcc: ", best_mcc)
    best_mask = np.isclose(mcc, best_mcc)
    best_balanced_accuracy = np.max(balanced_accuracy[best_mask])
    best_mask &= np.isclose(balanced_accuracy, best_balanced_accuracy)

    return thresholds[np.where(best_mask)[0][-1]]


def fill_missing_thresholds(thresholds):
    valid = np.isfinite(thresholds)
    if not valid.any():
        raise ValueError("No z-slices had enough wall/background samples.")

    missing = ~valid
    if missing.any():
        z_indices = np.arange(thresholds.size)
        thresholds[missing] = np.interp(
            z_indices[missing], z_indices[valid], thresholds[valid]
        )

    return gaussian_filter1d(thresholds, sigma=SMOOTHING_SIGMA_Z)


def append_values_by_z(values_by_z, z_indices, values):
    if values.size == 0:
        return

    order = np.argsort(z_indices)
    sorted_z = z_indices[order]
    sorted_values = values[order]
    unique_z, starts = np.unique(sorted_z, return_index=True)
    ends = np.r_[starts[1:], sorted_z.size]

    for z, start, end in zip(unique_z, starts, ends):
        values_by_z[int(z)].append(sorted_values[start:end])


def collect_wall_and_boundary_values(tomo_path, voxel_size, nz, region, vault_coords):
    positive_values_by_z = [[] for _ in range(nz)]
    negative_values_by_z = [[] for _ in range(nz)]

    wall_tree = KDTree(vault_coords)
    vault_center = np.mean(vault_coords, axis=0)

    z_start, z_end = region[2], region[5]
    section_volume = load_section_volume(tomo_path, region)

    snz, sny, snx = section_volume.shape
    grid_z, grid_y, grid_x = np.indices((snz, sny, snx))
    all_indices = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)

    # Offset local section indices back to global voxel indices
    global_indices = all_indices + np.array([region[0], region[1], region[2]])

    # The map's origin puts it in the same coordinate frame as the fitted CIF
    with mrcfile.open(tomo_path, mode="r", permissive=True) as mrc:
        origin = np.array(
            [mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z],
            dtype=float,
        )
    all_points_angstroms = global_indices * voxel_size + origin

    # Query ideal vault distances using aligned coordinates
    dists_to_wall, wall_indices = wall_tree.query(all_points_angstroms)
    nearest_wall_points = vault_coords[wall_indices]

    dist_wall_to_center = np.linalg.norm(nearest_wall_points - vault_center, axis=1)
    dist_candidate_to_center = np.linalg.norm(
        all_points_angstroms - vault_center, axis=1
    )

    wall_positive_mask = dists_to_wall <= WALL_POSITIVE_RADIUS_ANGSTROMS
    adjacent_non_wall_mask = (dists_to_wall >= NEGATIVE_BAND_INNER_ANGSTROMS) & (
        dists_to_wall <= NEGATIVE_BAND_OUTER_ANGSTROMS
    )
    negative_mask = adjacent_non_wall_mask & (
        dist_candidate_to_center > dist_wall_to_center
    )

    flat_vals = section_volume.ravel()
    global_z = global_indices[:, 2]

    append_values_by_z(
        positive_values_by_z,
        global_z[wall_positive_mask],
        flat_vals[wall_positive_mask],
    )
    append_values_by_z(
        negative_values_by_z,
        global_z[negative_mask],
        flat_vals[negative_mask],
    )

    print(
        f"  sampled region z={z_start}:{z_end}, "
        f"wall voxels={wall_positive_mask.sum()}, "
        f"non-wall voxels={negative_mask.sum()}"
    )

    return positive_values_by_z, negative_values_by_z


def compute_thresholds_per_z(tomo_path, voxel_size, nz, region, vault_coords):
    positive_values_by_z, negative_values_by_z = collect_wall_and_boundary_values(
        tomo_path=tomo_path,
        voxel_size=voxel_size,
        nz=nz,
        region=region,
        vault_coords=vault_coords,
    )

    thresholds = np.full(nz, np.nan)
    for z in range(nz):
        if not positive_values_by_z[z] or not negative_values_by_z[z]:
            continue

        positive_vals = np.concatenate(positive_values_by_z[z])
        negative_vals = np.concatenate(negative_values_by_z[z])
        thresholds[z] = compute_mcc_threshold(positive_vals, negative_vals)

    return fill_missing_thresholds(thresholds)


def write_thresholds(output_path, thresholds):
    with open(output_path, "w") as f:
        for z, threshold in enumerate(thresholds):
            f.write(f"{z}: {threshold:.6f}\n")


def process_tomo():
    print(f"Processing tomogram: {TOMO_PATH}")
    voxel_size, _, _, nz = get_mrc_file_header_info(TOMO_PATH)
    vault_coords = get_vault_coordinates(FIT_CIF_PATH)

    region = [0, 0, 0, 188, 188, 305]

    thresholds = compute_thresholds_per_z(
        tomo_path=TOMO_PATH,
        voxel_size=voxel_size,
        nz=nz,
        region=region,
        vault_coords=vault_coords,
    )

    output_path = (
        Path("./slice_thresholds/atomic_model")
        / "thresholds_by_z_slice_atomic_model.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_thresholds(output_path, thresholds)

    print(
        f"Wrote {output_path} with threshold range "
        f"{np.min(thresholds):.6f} to {np.max(thresholds):.6f}"
    )


if __name__ == "__main__":
    process_tomo()
