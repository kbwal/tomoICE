import math
import re
from pathlib import Path

import biotite.structure.io as strucio
import mrcfile
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree


TOMOS_DIR = Path("./tomos")
CENTERS_DIR = Path("./center_coordinates")
REGIONS_ROOT = Path("./fits")
IDEAL_VAULT_PATH = Path("./7PKR.cif")

BOX_RADIUS_VOXELS = 75
WALL_WIDTH_ANGSTROMS = 20.0
WALL_POSITIVE_RADIUS_ANGSTROMS = WALL_WIDTH_ANGSTROMS / 2.0
NEGATIVE_BAND_INNER_ANGSTROMS = WALL_WIDTH_ANGSTROMS * 2.0
NEGATIVE_BAND_OUTER_ANGSTROMS = WALL_WIDTH_ANGSTROMS * 2.5
MIN_POSITIVE_SAMPLES_PER_Z = 10
MIN_NEGATIVE_SAMPLES_PER_Z = 10
SMOOTHING_SIGMA_Z = 2


def parse_fit_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    scale = float(re.search(r"Scale:\s*([\d.]+)", content).group(1))
    matrix_str = re.search(r"Transformation:\s*\[(.*?)\]", content, re.DOTALL).group(1)
    values = [float(x) for x in matrix_str.split()]
    transformation = np.array(values).reshape(4, 4)

    return scale, transformation


def get_mrc_file_header_info(file_path):
    with mrcfile.open(file_path, mode="r", permissive=True) as mrc:
        vox_size = mrc.voxel_size
        nx = mrc.header.nx
        ny = mrc.header.ny
        nz = mrc.header.nz
        return float(vox_size["x"]), int(nx), int(ny), int(nz)


def get_ideal_vault():
    atom_array = strucio.load_structure(str(IDEAL_VAULT_PATH))
    ca_atoms = atom_array[atom_array.atom_name == "CA"]
    points_angstrom = ca_atoms.coord
    center_of_mass = np.mean(points_angstrom, axis=0)
    return points_angstrom - center_of_mass


def load_section_volume(file_path, region):
    x_start, y_start, z_start, x_end, y_end, z_end = region
    with mrcfile.open(file_path, mode="r", permissive=True) as mrc:
        section = -mrc.data[z_start:z_end, y_start:y_end, x_start:x_end]
    return np.array(section)


def center_to_region(center, nx, ny, nz, voxel_size, box_radius=BOX_RADIUS_VOXELS):
    center_x, center_y, center_z = center
    center_x = center_x / voxel_size
    center_y = center_y / voxel_size
    center_z = center_z / voxel_size
    return [
        max(0, math.floor(center_x - box_radius)),
        max(0, math.floor(center_y - box_radius)),
        max(0, math.floor(center_z - box_radius)),
        min(nx, math.ceil(center_x + box_radius)),
        min(ny, math.ceil(center_y + box_radius)),
        min(nz, math.ceil(center_z + box_radius)),
    ]


def recover_centroid(fit_cif_path, fit_metadata_path, ideal_vault, region, voxel_size):
    scale, transformation = parse_fit_file(fit_metadata_path)

    scaling_matrix = np.eye(4)
    scaling_matrix[0][0] = scale
    scaling_matrix[1][1] = scale
    full_transform = transformation @ scaling_matrix

    ideal_h = np.hstack([ideal_vault, np.ones((ideal_vault.shape[0], 1))])
    aligned_ideal = (full_transform @ ideal_h.T).T[:, :3]
    mean_aligned_ideal = np.mean(aligned_ideal, axis=0)

    fit_atoms = strucio.load_structure(str(fit_cif_path))
    mean_fit = np.mean(fit_atoms.coord, axis=0)
    region_start = np.array([region[0], region[1], region[2]]) * voxel_size

    return mean_fit - region_start - mean_aligned_ideal


def extract_tomo_index(tomo_path):
    match = re.search(r"Position_1_(\d+)_", tomo_path.name)
    if match is not None:
        return int(match.group(1))

    numbers = re.findall(r"\d+", tomo_path.stem)
    if numbers:
        return int(numbers[-1])

    raise ValueError(f"Could not infer tomo index from {tomo_path}")


def find_tomo_files():
    tomo_files = sorted(TOMOS_DIR.glob("*.mrc"), key=extract_tomo_index)
    return tomo_files


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
    # lo = min(np.min(positive_vals), np.min(negative_vals))
    lo = np.percentile(all_vals, 1)
    # hi = max(np.max(positive_vals), np.max(negative_vals))
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

    # Prefer the strictest threshold among equivalent optima so a low threshold
    # is not chosen when it only adds off-wall false positives.
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


def collect_wall_and_boundary_values(
    tomo_path,
    voxel_size,
    nz,
    regions,
    fit_metadata_paths,
    fit_cif_paths,
    ideal_vault,
):
    positive_values_by_z = [[] for _ in range(nz)]
    negative_values_by_z = [[] for _ in range(nz)]
    wall_tree = KDTree(ideal_vault)

    for region, meta_path, cif_path in zip(regions, fit_metadata_paths, fit_cif_paths):
        z_start, z_end = region[2], region[5]
        section_volume = load_section_volume(tomo_path, region)

        scale, transformation = parse_fit_file(meta_path)
        scaling_matrix = np.eye(4)
        scaling_matrix[0][0] = scale
        scaling_matrix[1][1] = scale

        broken_centroid = recover_centroid(
            cif_path, meta_path, ideal_vault, region, voxel_size
        )

        snz, sny, snx = section_volume.shape
        grid_z, grid_y, grid_x = np.indices((snz, sny, snx))
        all_indices = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)

        all_points_angstroms = all_indices * voxel_size
        all_points_centered = all_points_angstroms - broken_centroid

        inv_transform = np.linalg.inv(transformation @ scaling_matrix)
        points_h = np.hstack(
            [all_points_centered, np.ones((all_points_centered.shape[0], 1))]
        )
        all_points_ideal = (points_h @ inv_transform.T)[:, :3]

        dists_to_wall, wall_indices = wall_tree.query(all_points_ideal)
        nearest_wall_points = ideal_vault[wall_indices]
        dist_wall_to_center = np.linalg.norm(nearest_wall_points, axis=1)
        dist_candidate_to_center = np.linalg.norm(all_points_ideal, axis=1)

        wall_positive_mask = dists_to_wall <= WALL_POSITIVE_RADIUS_ANGSTROMS
        adjacent_non_wall_mask = (dists_to_wall >= NEGATIVE_BAND_INNER_ANGSTROMS) & (
            dists_to_wall <= NEGATIVE_BAND_OUTER_ANGSTROMS
        )
        negative_mask = adjacent_non_wall_mask & (
            dist_candidate_to_center > dist_wall_to_center
        )

        flat_vals = section_volume.ravel()
        global_z = all_indices[:, 2] + z_start

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


def compute_thresholds_per_z(
    tomo_path,
    voxel_size,
    nz,
    regions,
    fit_metadata_paths,
    fit_cif_paths,
    ideal_vault,
):
    positive_values_by_z, negative_values_by_z = collect_wall_and_boundary_values(
        tomo_path=tomo_path,
        voxel_size=voxel_size,
        nz=nz,
        regions=regions,
        fit_metadata_paths=fit_metadata_paths,
        fit_cif_paths=fit_cif_paths,
        ideal_vault=ideal_vault,
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


def process_tomo(tomo_path, ideal_vault):
    tomo_index = extract_tomo_index(tomo_path)
    print(f"Processing tomo {tomo_index}: {tomo_path}")

    voxel_size, nx, ny, nz = get_mrc_file_header_info(tomo_path)
    centers_path = CENTERS_DIR / f"center_coordinates_{tomo_index}.txt"
    main_region_dir = REGIONS_ROOT / f"tomo_{tomo_index}" / "regions"

    centers = np.loadtxt(centers_path, delimiter=",", ndmin=2).tolist()
    regions = [center_to_region(center, nx, ny, nz, voxel_size) for center in centers]
    fit_metadata_paths = [
        main_region_dir / f"region{i + 1}" / "fit_metadata.txt"
        for i in range(len(regions))
    ]
    fit_cif_paths = [
        main_region_dir / f"region{i + 1}" / "fit.cif" for i in range(len(regions))
    ]

    thresholds = compute_thresholds_per_z(
        tomo_path=tomo_path,
        voxel_size=voxel_size,
        nz=nz,
        regions=regions,
        fit_metadata_paths=fit_metadata_paths,
        fit_cif_paths=fit_cif_paths,
        ideal_vault=ideal_vault,
    )

    output_path = Path("./slice_thresholds") / f"thresholds_by_z_slice_tomo_{tomo_index}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_thresholds(output_path, thresholds)
    print(
        f"Wrote {output_path} with threshold range "
        f"{np.min(thresholds):.6f} to {np.max(thresholds):.6f}"
    )


ideal_vault = get_ideal_vault()
for tomo_path in find_tomo_files():
    process_tomo(tomo_path, ideal_vault)
