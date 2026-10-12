import re
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.interpolate import make_interp_spline


def main():
    buckets_config = [
        {"range": (1.0, 1.15), "image": "scale_1.08_eg_for_plot.png"},
        {"range": (1.15, 1.25), "image": "scale_1.18_eg_for_plot.png"},
        {"range": (1.25, 1.35), "image": "scale_1.29_eg_for_plot.png"},
    ]
    buckets = [b["range"] for b in buckets_config]

    all_data = []  # list of (percent, scale) tuples
    bucket_data = {b: [] for b in buckets}

    files = glob.glob("./fits/*/percentage-z-slices-precomputed-thresholds.txt")

    for percentage_file in sorted(files):
        tomo_dir = os.path.dirname(percentage_file)

        with open(percentage_file, "r") as f:
            content = f.read()

        matches = re.findall(r"Region\s+(\d+)\.\s+Percentage:\s*(\d+\.?\d*)", content)
        for region_str, percent_str in matches:
            region_id = int(region_str)
            percent = float(percent_str)

            # Default scale for entries without metadata
            entry_scale = 1.0

            metadata_path = os.path.join(
                tomo_dir, "regions", f"region{region_id}", "fit_metadata.txt"
            )
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as mf:
                    meta_content = mf.read()

                scale_match = re.search(r"Scale:\s*(\d+\.?\d*)", meta_content)
                if scale_match:
                    scale = float(scale_match.group(1))

                    entry_scale = scale

                    for b in buckets:
                        # Standard range assignment: min <= x < max
                        if b[0] <= scale < b[1]:
                            bucket_data[b].append((percent, scale))
                        # Ensure we capture exact upper bound for the very last bucket
                        elif b == buckets[-1] and scale == b[1]:
                            bucket_data[b].append((percent, scale))

            all_data.append((percent, entry_scale))

    def process_data(data_list):
        """Convert (percent, scale) tuples to estimated cargo count.

        The percentage is measured relative to a 1.0-scale vault.  In a
        vault scaled by S the same single cargo occupies S**2 less area,
        so the effective single-cargo percentage is 0.24 / S**2.  Thus:
            num_cargo = percent / (percent_single_il2 / scale**2)
        """
        results = np.array([p * (s**2) / 0.24 for p, s in data_list])
        return results

    fig, ax = plt.subplots(figsize=(14, 5))

    legend_data = []

    def plot_smooth(
        data_arr, label, color, image_path=None, linestyle="-", linewidth=2.5, bins=100
    ):
        counts, bin_edges = np.histogram(data_arr, bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        k_val = min(3, len(bin_centers) - 1)
        spline = make_interp_spline(bin_centers, counts, k=k_val)
        x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 500)
        y_smooth = np.maximum(spline(x_smooth), 0)

        full_label = f"{label} (n={len(data_arr)})"
        ax.plot(
            x_smooth,
            y_smooth,
            label=full_label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )

        legend_data.append((full_label))

    all_arr = process_data(all_data)

    min_val = np.min(all_arr)
    max_val = np.max(all_arr)
    common_bins = np.linspace(min_val, max_val, 50)

    plot_smooth(
        all_arr,
        "All particles",
        color="#555752",
        linestyle="-",
        linewidth=3,
        bins=common_bins,
    )

    bucket_labels = ["Minimal expansion", "Moderate expansion", "Large expansion"]
    colors = ["#65c7cf", "#b387ff", "#c0272d"]
    for idx, b_config in enumerate(buckets_config):
        b = b_config["range"]
        img_path = b_config.get("image")
        arr = process_data(bucket_data[b])
        plot_smooth(
            arr,
            bucket_labels[idx],
            color=colors[idx % len(colors)],
            image_path=img_path,
            linestyle="-",
            bins=common_bins,
        )

    ax.set_xlabel("Cargo Number")
    ax.set_ylabel("Particle Number")

    ax.legend(loc="upper right", ncol=2)

    output_filename = "./incorporating_scale_num_il2_figure.png"
    plt.savefig(output_filename, bbox_inches="tight")
    print(f"Saved overlaid plot to {output_filename}")


if __name__ == "__main__":
    main()
