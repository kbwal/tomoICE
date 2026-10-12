import mrcfile
import numpy as np
from scipy.spatial import KDTree
import math
import biotite.structure.io as strucio
import re


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
        print(f"voxel size is: {vox_size}")
        nx = mrc.header.nx
        ny = mrc.header.ny
        nz = mrc.header.nz
        return float(vox_size["x"]), int(nx), int(ny), int(nz)


def get_ideal_vault():
    atom_array = strucio.load_structure("7PKR.cif")
    ca_atoms = atom_array[atom_array.atom_name == "CA"]
    points_angstrom = ca_atoms.coord  # (n, 3)
    center_of_mass = np.mean(points_angstrom, axis=0)
    points_centered_at_zero = points_angstrom - center_of_mass
    return points_centered_at_zero


def load_section_volume(file_path, region):
    x_start, y_start, z_start, x_end, y_end, z_end = region
    with mrcfile.open(file_path, mode="r", permissive=True) as mrc:
        section = -mrc.data[z_start:z_end, y_start:y_end, x_start:x_end]
    return np.array(section)


def count_percentage(
    voxel_size,
    transformation_matrix,
    ideal_vault_points,
    global_centroid_offset,
    section_volume,
    thresholds_per_z,
    region_origin,
    wall_tolerance,
    save_masks=False,
    output_dir=None,
):
    nz, ny, nx = section_volume.shape
    grid_z, grid_y, grid_x = np.indices((nz, ny, nx))
    all_indices = np.stack(
        [grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=1
    )  # shape of (nx*ny*nz, 3)

    all_points_angstroms = all_indices * voxel_size
    all_points_centered = all_points_angstroms - global_centroid_offset

    inv_transform = np.linalg.inv(transformation_matrix)

    # this is cuz all_points_centered is a tensor of (nx*ny*nz, 3) and the
    # transformation is 4x4 which can't multiply unless you add a col
    points_no_inv = np.hstack(
        [all_points_centered, np.ones((all_points_centered.shape[0], 1))]
    )

    points_inv = points_no_inv @ inv_transform.T  # should be (nx * ny * nz, 4)

    all_points_broken = points_inv[:, :3]  # cut off the ones

    ideal_z_coords = ideal_vault_points[:, 2]
    min_z = np.min(ideal_z_coords)
    max_z = np.max(ideal_z_coords)
    vault_height = max_z - min_z
    z_margin = vault_height * 0.18
    cutoff_bottom = min_z + z_margin
    cutoff_top = max_z - z_margin

    candidate_z = all_points_broken[:, 2]
    not_in_caps_mask = (candidate_z > cutoff_bottom) & (candidate_z < cutoff_top)

    # this KD tree stuff is absolute magic
    tree = KDTree(ideal_vault_points)
    dists_to_wall, wall_indices = tree.query(all_points_broken)
    nearest_wall_points = ideal_vault_points[wall_indices]

    dist_wall_to_center = np.linalg.norm(nearest_wall_points, axis=1)
    dist_candidate_to_center = np.linalg.norm(all_points_broken, axis=1)

    not_wall_mask = dists_to_wall >= wall_tolerance
    inside_vault_mask = dist_candidate_to_center < dist_wall_to_center
    considered_volume_mask = not_wall_mask & inside_vault_mask

    local_z_indices = all_indices[:, 2]
    global_z_indices = local_z_indices + region_origin[2]
    per_voxel_thresholds = thresholds_per_z[global_z_indices]
    above_threshold_mask = section_volume.flatten() > per_voxel_thresholds

    amount_above_threshold = np.sum(
        considered_volume_mask & above_threshold_mask & not_in_caps_mask
    )
    total = np.sum(considered_volume_mask)

    percentage_occupancy = (amount_above_threshold / total) * 100.0

    mask_shape = (nz, ny, nx)

    origin_angstrom = (
        region_origin[0] * voxel_size,
        region_origin[1] * voxel_size,
        region_origin[2] * voxel_size,
    )

    if save_masks:
        with mrcfile.new(
            f"./{output_dir}/numerator_z_slices_precomputed_thresholds.mrc",
            overwrite=True,
        ) as mrc:
            mrc.set_data(
                (considered_volume_mask & above_threshold_mask & not_in_caps_mask)
                .reshape(mask_shape)
                .astype(np.float32)
            )
            mrc.voxel_size = voxel_size
            mrc.header.origin.x = origin_angstrom[0]
            mrc.header.origin.y = origin_angstrom[1]
            mrc.header.origin.z = origin_angstrom[2]

        with mrcfile.new(
            f"./{output_dir}/denominator_z_slices_precomputed_thresholds.mrc",
            overwrite=True,
        ) as mrc:
            mrc.set_data(
                (considered_volume_mask).reshape(mask_shape).astype(np.float32)
            )
            mrc.voxel_size = voxel_size
            mrc.header.origin.x = origin_angstrom[0]
            mrc.header.origin.y = origin_angstrom[1]
            mrc.header.origin.z = origin_angstrom[2]

    return percentage_occupancy


def recover_centroid(fit_cif_path, fit_metadata_path, ideal_vault, region, voxel_size):
    scale, transformation = parse_fit_file(fit_metadata_path)

    scaling_matrix = np.eye(4)
    scaling_matrix[0][0] = scale
    scaling_matrix[1][1] = scale
    full_transform = transformation @ scaling_matrix

    # transform ideal vault the same way the fitting script did
    ideal_h = np.hstack([ideal_vault, np.ones((ideal_vault.shape[0], 1))])
    aligned_ideal = (full_transform @ ideal_h.T).T[:, :3]
    mean_aligned_ideal = np.mean(aligned_ideal, axis=0)

    # load the saved fit cif
    fit_atoms = strucio.load_structure(fit_cif_path)
    mean_fit = np.mean(fit_atoms.coord, axis=0)

    region_start = np.array([region[0], region[1], region[2]]) * voxel_size

    broken_centroid = mean_fit - region_start - mean_aligned_ideal
    return broken_centroid


def center_to_region(center, nx, ny, nz, voxel_size, box_radius=75):
    center_x, center_y, center_z = center
    center_x = center_x / voxel_size
    center_y = center_y / voxel_size
    center_z = center_z / voxel_size
    region = [
        max(0, math.floor(center_x - box_radius)),
        max(0, math.floor(center_y - box_radius)),
        max(0, math.floor(center_z - box_radius)),
        min(nx, math.ceil(center_x + box_radius)),
        min(ny, math.ceil(center_y + box_radius)),
        min(nz, math.ceil(center_z + box_radius)),
    ]
    return region


def load_thresholds_per_z(tomo_number, nz, threshold_dir="./slice_thresholds"):
    threshold_path = f"{threshold_dir}/thresholds_by_z_slice_tomo_{tomo_number}.txt"
    threshold_data = np.loadtxt(threshold_path, delimiter=":", ndmin=2)

    z_indices = threshold_data[:, 0].astype(int)
    threshold_values = threshold_data[:, 1].astype(float)

    return threshold_values


for j in range(1, 5):
    main_region_dir = f"./fits/tomo_{j}/regions"
    file_path = f"./tomos/Position_1_{j}_SIRT_rec_corrected.mrc"
    voxel_size, nx, ny, nz = get_mrc_file_header_info(file_path=file_path)
    wall_tolerance = 20.0
    centers = np.loadtxt(
        f"./center_coordinates/center_coordinates_{j}.txt",
        delimiter=",",
        ndmin=2,
    ).tolist()
    regions = [center_to_region(center, nx, ny, nz, voxel_size) for center in centers]
    print(f"loaded {len(regions)} regions")

    ideal_vault = get_ideal_vault()
    thresholds_per_z = load_thresholds_per_z(tomo_number=j, nz=nz)
    percentage_output_file = f"./fits/tomo_{j}/percentage-z-slices-precomputed-thresholds.txt"
    with open(percentage_output_file, "w") as f:
        f.write("")

    for i, region in enumerate(regions):
        section_volume = load_section_volume(file_path, region)

        region_dir = f"./{main_region_dir}/region{i+1}"
        scale, transformation = parse_fit_file(f"./{region_dir}/fit_metadata.txt")

        broken_centroid = recover_centroid(
            fit_cif_path=f"./{region_dir}/fit.cif",
            fit_metadata_path=f"./{region_dir}/fit_metadata.txt",
            ideal_vault=ideal_vault,
            region=region,
            voxel_size=voxel_size,
        )

        scaling_matrix = np.eye(4)
        scaling_matrix[0][0] = scale
        scaling_matrix[1][1] = scale

        percentage = count_percentage(
            voxel_size=voxel_size,
            transformation_matrix=transformation @ scaling_matrix,
            ideal_vault_points=ideal_vault,
            global_centroid_offset=broken_centroid,
            section_volume=section_volume,
            thresholds_per_z=thresholds_per_z,
            region_origin=region[0:3],
            wall_tolerance=wall_tolerance,
            save_masks=True,
            output_dir=region_dir,
        )

        z_start = region[2]
        z_end = region[5]
        local_thresholds = thresholds_per_z[z_start:z_end]
        with open(percentage_output_file, "a") as f:
            f.write(
                f"Region {i+1}. Percentage: {percentage}. "
                f"Threshold range: [{np.min(local_thresholds):.4f}, "
                f"{np.max(local_thresholds):.4f}]\n"
            )

        print("percentage of vault with density inside: ", percentage)
    #     break
    # break
