# type: ignore
import mrcfile
import numpy as np
import biotite.structure.io as strucio
import biotite.structure as struc
from joblib import Parallel, delayed
import open3d as o3d
import time
import math
import os
from scipy.ndimage import binary_erosion, gaussian_filter


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


def weight_vault_caps(ideal_vault, cap_fraction=0.15, cap_copies=3):
    z_coords = ideal_vault[:, 2]
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    z_range = z_max - z_min

    cap_mask = (z_coords > z_max - (z_range * cap_fraction)) | (
        z_coords < z_min + (z_range * cap_fraction)
    )
    cap_points = ideal_vault[cap_mask]

    # Duplicate cap points only for ICP weighting; do not save these duplicates.
    return np.vstack([ideal_vault] + [cap_points] * cap_copies)


def get_points_of_broken_vault(
    file_path,
    voxel_size,
    threshold_percentile,
    region,
    save_region_seen=False,
    output_dir=None,
):
    x_start, y_start, z_start, x_end, y_end, z_end = region

    with mrcfile.open(file_path, mode="r", permissive=True) as mrc:
        section = -mrc.data[z_start:z_end, y_start:y_end, x_start:x_end]
        section_volume = np.array(section)

    section_volume = gaussian_filter(section_volume, sigma=1.5)

    threshold_value = np.percentile(section_volume, threshold_percentile)

    mask = section_volume > threshold_value
    eroded_mask = binary_erosion(mask, iterations=1)
    mask = mask ^ eroded_mask
    voxel_coords = np.argwhere(mask)

    z, y, x = voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]

    points_angstrom = np.stack([x, y, z], axis=1) * voxel_size

    pcd_temp = o3d.geometry.PointCloud()
    pcd_temp.points = o3d.utility.Vector3dVector(points_angstrom)

    region_start = [
        (x_start) * voxel_size,
        (y_start) * voxel_size,
        (z_start) * voxel_size,
    ]

    centroid = np.mean(points_angstrom, axis=0)
    points_centered = points_angstrom - centroid

    if save_region_seen:
        shifted_broken_pdb = struc.AtomArray(length=len(points_angstrom))
        shifted_broken_pdb.coord = points_angstrom + region_start
        shifted_broken_pdb.atom_name[:] = "CA"
        shifted_broken_pdb.res_name[:] = "IDL"
        shifted_broken_pdb.element[:] = "C"
        shifted_broken_pdb.chain_id[:] = "K"
        shifted_broken_pdb.res_id = np.arange(1, len(points_angstrom) + 1)
        output_path = f"./{output_dir}/region_seen.cif"
        strucio.save_structure(output_path, shifted_broken_pdb)

    return points_centered, centroid, section_volume, threshold_value


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=100)
    )

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def find_best_fit_one_scale(scale, target_tuple, ideal_vault, voxel_size):
    t_points, t_normals, t_down_points, t_down_normals, t_fpfh_data = target_tuple
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(t_points.copy())
    target.normals = o3d.utility.Vector3dVector(t_normals.copy())
    target_down = o3d.geometry.PointCloud()
    target_down.points = o3d.utility.Vector3dVector(t_down_points.copy())
    target_down.normals = o3d.utility.Vector3dVector(t_down_normals.copy())
    target_fpfh = o3d.pipelines.registration.Feature()
    target_fpfh.data = t_fpfh_data.copy()

    x_y_scaling_matrix = np.eye(4)
    x_y_scaling_matrix[0, 0] = scale
    x_y_scaling_matrix[1, 1] = scale
    x_y_scaled_ideal_vault = (
        x_y_scaling_matrix
        @ (np.hstack([ideal_vault, np.ones((ideal_vault.shape[0], 1))])).T
    )[:3, :].T

    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(x_y_scaled_ideal_vault)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)

    ransac_threshold = voxel_size * 3.0
    icp_corse_threshold = voxel_size * 4.0
    icp_threshold = voxel_size * 1.5
    n_tries = 3
    max_fitness = -np.inf
    best_result = None

    for _ in range(n_tries):
        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=ransac_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False
            ),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    ransac_threshold
                ),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                max_iteration=100_000, confidence=9.999000e-01
            ),
        )

        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=100)
        )

        result_icp_coarse = o3d.pipelines.registration.registration_icp(
            source,
            target,
            icp_corse_threshold,
            result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )

        result_icp = o3d.pipelines.registration.registration_icp(
            source,
            target,
            icp_threshold,
            result_icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )

        source_aligned = o3d.geometry.PointCloud(source)
        source_aligned.transform(result_icp.transformation)
        model_tree = o3d.geometry.KDTreeFlann(source_aligned)
        sq_dists = np.array(
            [model_tree.search_knn_vector_3d(p, 1)[2][0] for p in t_points]
        )

        target_inliers = np.sum(sq_dists < (icp_threshold**2))
        target_coverage = target_inliers / len(t_points)
        combined_score = result_icp.fitness * target_coverage

        if combined_score > max_fitness:
            max_fitness = combined_score
            best_result = {
                "fitness": result_icp.fitness,
                "transformation": result_icp.transformation,
                "rmse": result_icp.inlier_rmse,
                "scale": scale,
                "x_y_scaling_matrix": x_y_scaling_matrix,
                "target_coverage": target_coverage,
                "combined_score": combined_score,
            }

    return best_result


def find_best_fit(
    broken_vault,
    ideal_vault,
    voxel_size,
    output_dir,
    region,
):
    prev_highest_fitness = -np.inf
    best_result = None
    best_xy_scaling_matrix = None

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(broken_vault)
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=100)
    )
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    target_tuple = (
        np.asarray(target.points),
        np.asarray(target.normals),
        np.asarray(target_down.points),
        np.asarray(target_down.normals),
        np.asarray(target_fpfh.data),
    )

    scales = np.round(np.arange(1.0, 1.301, 0.01), 2)
    results = Parallel(n_jobs=-1)(
        delayed(find_best_fit_one_scale)(scale, target_tuple, ideal_vault, voxel_size)
        for scale in scales
    )

    for result in results:
        if result["combined_score"] > prev_highest_fitness:
            prev_highest_fitness = result["combined_score"]
            best_result = result
            best_xy_scaling_matrix = result["x_y_scaling_matrix"]
    print(f"choosing the result with scale {best_result['scale']}")

    with open(f"{output_dir}/fit_metadata.txt", "w") as f:
        f.write(f"Region: {region}\n")
        f.write(f"Scale: {best_result['scale']}\n")
        f.write(f"Fitness: {best_result['fitness']}\n")
        f.write(f"RMSE: {best_result['rmse']}\n")
        f.write(f"TargetCoverage: {best_result['target_coverage']}\n")
        f.write(f"CombinedScore: {best_result['combined_score']}\n")
        f.write(f"Transformation: {best_result['transformation'].flatten()}\n")

    return best_result, best_xy_scaling_matrix


def center_to_region(center, nx, ny, nz, voxel_size):
    center_x, center_y, center_z = center
    center_x = center_x / voxel_size
    center_y = center_y / voxel_size
    center_z = center_z / voxel_size
    box_radius = 50
    region = [
        max(0, math.floor(center_x - box_radius)),
        max(0, math.floor(center_y - box_radius)),
        max(0, math.floor(center_z - box_radius)),
        min(nx, math.ceil(center_x + box_radius)),
        min(ny, math.ceil(center_y + box_radius)),
        min(nz, math.ceil(center_z + box_radius)),
    ]
    return region


ideal_vault = get_ideal_vault()
weighted_ideal_vault = weight_vault_caps(ideal_vault)
for tomo_idx in range(1, 5):
    file_path = f"./tomos/Position_1_{tomo_idx}_SIRT_rec_corrected.mrc"
    voxel_size, nx, ny, nz = get_mrc_file_header_info(file_path=file_path)
    threshold_percentile = 85

    centers = np.loadtxt(
        f"./center_coordinates/center_coordinates_{tomo_idx}.txt",
        delimiter=",",
        ndmin=2,
    ).tolist()
    regions = [center_to_region(center, nx, ny, nz, voxel_size) for center in centers]
    print(f"loaded {len(regions)} regions")

    main_output_dir = f"./fits/tomo_{tomo_idx}"
    os.makedirs(main_output_dir, exist_ok=True)
    vaults_output_dir = f"./{main_output_dir}/ideal_vaults"
    os.makedirs(vaults_output_dir, exist_ok=True)

    for i, region in enumerate(regions):
        print("doing region: ", region)
        t1 = time.time()
        output_dir = f"./{main_output_dir}/regions/region{i + 1}"
        os.makedirs(output_dir, exist_ok=True)

        broken_vault_points, broken_centroid, section_volume, threshold_value = (
            get_points_of_broken_vault(
                file_path=file_path,
                voxel_size=voxel_size,
                threshold_percentile=threshold_percentile,
                region=region,
                save_region_seen=True,
                output_dir=output_dir,
            )
        )

        output_path = f"./{output_dir}/fit.cif"
        output_path_ideal_vaults = (
            f"./{vaults_output_dir}/fit_{'_'.join(str(num) for num in region)}.cif"
        )
        print("threshold value: ", threshold_value)
        result, x_y_scaling_matrix = find_best_fit(
            broken_vault_points,
            weighted_ideal_vault,
            voxel_size=voxel_size,
            output_dir=output_dir,
            region=region,
        )

        pcd_ideal = o3d.geometry.PointCloud()
        pcd_ideal.points = o3d.utility.Vector3dVector(ideal_vault)
        pcd_ideal.transform(result["transformation"] @ x_y_scaling_matrix)
        aligned_ideal_points = np.asarray(pcd_ideal.points)
        offset = (
            np.array([region[0], region[1], region[2]]) * voxel_size
        ) + broken_centroid
        shifted_aligned_ideal_points = aligned_ideal_points + offset
        shifted_ideal_pdb = struc.AtomArray(length=len(shifted_aligned_ideal_points))
        shifted_ideal_pdb.coord = shifted_aligned_ideal_points
        shifted_ideal_pdb.atom_name[:] = "CA"
        shifted_ideal_pdb.res_name[:] = "IDL"
        shifted_ideal_pdb.element[:] = "C"
        shifted_ideal_pdb.chain_id[:] = "A"
        shifted_ideal_pdb.res_id = np.arange(1, len(shifted_aligned_ideal_points) + 1)
        strucio.save_structure(output_path, shifted_ideal_pdb)
        strucio.save_structure(output_path_ideal_vaults, shifted_ideal_pdb)
        print(f"time taken to do this region was: {time.time() - t1}s")
        print(f"done with {i + 1} / {len(regions)} regions")
        print("-------------------------------")
