import os
import mrcfile
import numpy as np
from scipy.spatial import KDTree
import biotite.structure.io as strucio


def get_mrc_file_header_info(file_path):
    with mrcfile.open(file_path, mode="r", permissive=True) as mrc:
        vox_size = mrc.voxel_size
        print(f"voxel size is: {vox_size}")
        nx = int(mrc.header.nx)
        ny = int(mrc.header.ny)
        nz = int(mrc.header.nz)
        origin = np.array(
            [mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z]
        )
        return float(vox_size["x"]), nx, ny, nz, origin


def get_vault_points(cif_path):
    """Loads the pre-aligned vault directly from the CIF file."""
    atom_array = strucio.load_structure(cif_path)
    ca_atoms = atom_array[atom_array.atom_name == "CA"]
    coords = ca_atoms.coord
    center_of_mass = np.mean(coords, axis=0)
    coords_centered = coords - center_of_mass
    return coords_centered, center_of_mass


def load_section_volume(file_path, region):
    x_start, y_start, z_start, x_end, y_end, z_end = region
    with mrcfile.open(file_path, mode="r", permissive=True) as mrc:
        section = mrc.data[z_start:z_end, y_start:y_end, x_start:x_end]
    return np.array(section)


def load_thresholds_per_z(threshold_path):
    threshold_data = np.loadtxt(threshold_path, delimiter=":", ndmin=2)
    threshold_values = threshold_data[:, 1].astype(float)
    return threshold_values


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

    amount_above_threshold = np.sum(considered_volume_mask & above_threshold_mask)
    total = np.sum(considered_volume_mask)

    percentage_occupancy = (amount_above_threshold / total) * 100.0

    mask_shape = (nz, ny, nx)

    origin_angstrom = (
        region_origin[0] * voxel_size,
        region_origin[1] * voxel_size,
        region_origin[2] * voxel_size,
    )
    _, _, _, _, mrc_origin = get_mrc_file_header_info(file_path=mrc_file_path)
    origin_angstrom += mrc_origin

    if save_masks:
        with mrcfile.new(
            f"./{output_dir}/numerator_z_slices_precomputed_thresholds.mrc",
            overwrite=True,
        ) as mrc:
            mrc.set_data(
                (considered_volume_mask & above_threshold_mask)
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


if __name__ == "__main__":
    mrc_file_path = "./atomic_model.mrc"
    cif_file_path = "./fit_atomic_model.cif"
    threshold_path = (
        "./slice_thresholds/atomic_model/thresholds_by_z_slice_atomic_model.txt"
    )
    output_dir = "./atomic_model_percentages"

    wall_tolerance = 20.0
    region = [0, 0, 0, 305, 305, 305]

    os.makedirs(output_dir, exist_ok=True)
    percentage_output_file = (
        f"{output_dir}/percentage-z-slices-precomputed-thresholds.txt"
    )

    voxel_size, nx, ny, nz, mrc_origin = get_mrc_file_header_info(
        file_path=mrc_file_path
    )

    thresholds_per_z = load_thresholds_per_z(threshold_path)

    vault_points, c_m = get_vault_points(cif_file_path)
    region_start_angstroms = (
        np.array([region[0], region[1], region[2]]) * voxel_size
    ) + mrc_origin
    centroid_offset = c_m - region_start_angstroms

    section_volume = load_section_volume(mrc_file_path, region)

    percentage = count_percentage(
        voxel_size=voxel_size,
        section_volume=section_volume,
        thresholds_per_z=thresholds_per_z,
        wall_tolerance=wall_tolerance,
        save_masks=True,
        output_dir=output_dir,
        global_centroid_offset=centroid_offset,
        ideal_vault_points=vault_points,
        transformation_matrix=np.eye(4),
        region_origin=[0, 0, 0],
    )

    z_start = region[2]
    z_end = region[5]
    local_thresholds = thresholds_per_z[z_start:z_end]

    with open(percentage_output_file, "w") as f:
        f.write(
            f"Region: {region}. Percentage: {percentage:.2f}%. "
            f"Threshold range: [{np.min(local_thresholds):.4f}, "
            f"{np.max(local_thresholds):.4f}]\n"
        )

    print(f"Percentage of vault with density inside: {percentage:.2f}%")
