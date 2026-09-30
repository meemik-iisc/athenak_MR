#!/usr/bin/env python3

import os

# Prevent each worker/subprocess from spawning many BLAS/OpenMP threads.
# This avoids CPU oversubscription when several .bin files run in parallel.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import math
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

MAX_WORKERS = 4  # Start with 6. Try 8 later if RAM usage is comfortable.
plot_title_fontsize = 24
plot_suptitile_fontsize = 40

config = {
    # File processing
    "dt": 0.01,
    "time_label": "Myr",
    "plot_dimension": "z",

    # Figure sizes
    "fig_size_single": (10.0, 10.0),
    "fig_size_combined": (20.0, 20.0),

    # Plot variables
    "plot_vars": {
        "dens": {
            "label": r"Density [$\mathbf{m_p/cm^3}$]",
            "folder_key": "dens",
            "quantity": "dens",
            "cmap": "Greens",
            "norm": "log",
            "vmin": 1.0e-2,
            "vmax": 1.0,
        },
        "pres": {
            "label": r"Pressure [$\mathbf{dyne/cm^2}$]",
            "folder_key": "pres",
            "quantity": "eint",
            "cmap": "viridis",
            "norm": "log",
            "vmin": 1.0e-14,
            "vmax": 1.0e-10,
        },
        "entropy": {
            "label": "Entropy [cgs]",
            "folder_key": "entropy",
            "quantity": "derived:entropy",
            "cmap": "magma",
            "norm": None,
            "vmin": -20.0,
            "vmax": 0.0,
        },
        "velx": {
            "label": "X Velocity [km/s]",
            "folder_key": "velx",
            "quantity": "velx",
            "cmap": "Blues_r",
            "norm": None,
            "vmin": -1.0e3,
            "vmax": 0.0,
        },
        "vely": {
            "label": "Y Velocity [km/s]",
            "folder_key": "vely",
            "quantity": "vely",
            "cmap": "seismic",
            "norm": None,
            "vmin": -500.0,
            "vmax": 500.0,
        },
        "temp": {
            "label": "Temperature [K]",
            "folder_key": "temp",
            "quantity": "derived:T",
            "cmap": "coolwarm",
            "norm": "log",
            "vmin": 1.0e4,
            "vmax": 1.0e6,
        },
        "t_cool": {
            "label": "Cooling Time",
            "folder_key": "t_cool",
            "quantity": "derived:t_cool",
            "cmap": "turbo",
            "norm": "log",
            "vmin": 1.0e-3,
            "vmax": 1.0,
        },
        "tracer": {
            "label": "Outflow Tracer",
            "folder_key": "tracer",
            "quantity": "s_00",
            "cmap": "winter",
            "norm": "log",
            "vmin": 1.0e-5,
            "vmax": 1.0,
        },
    },

    # Two columns, arranged left-to-right then top-to-bottom
    "plot_order": [
        "dens",
        "pres",
        "temp",
        # "tracer",
        # "velx",
        # "vely",
        "t_cool"
    ],
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_dir(path: Path):
    """Create directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def get_plot_cmd(
    plotter_path: Path,
    bin_file: Path,
    quantity: str,
    out_file: Path,
    var_config: dict,
    dimension: str = "z",
    figsize: tuple = (16.0, 4.0),
) -> list:
    """Build the command for plot_slice.py."""
    cmd = [
        "python",
        str(plotter_path),
        str(bin_file),
        quantity,
        str(out_file),
        f"--dimension={dimension}",
        f"--cmap={var_config['cmap']}",
        "--notex",
        f"--figsize={figsize[0]},{figsize[1]}",
    ]

    if var_config["norm"]:
        cmd.append(f"--norm={var_config['norm']}")

    if var_config["vmin"] is not None:
        cmd.append(f"--vmin={var_config['vmin']}")

    if var_config["vmax"] is not None:
        cmd.append(f"--vmax={var_config['vmax']}")

    return cmd


def plot_single_quantity(
    bin_file: Path,
    var_name: str,
    out_root: Path,
    plotter: Path,
    config: dict,
) -> Path:
    """Generate a PNG plot for one variable and one .bin file."""
    var_config = config["plot_vars"][var_name]

    out_dir = out_root / var_config["folder_key"]
    ensure_dir(out_dir)

    out_file = out_dir / f"{bin_file.stem}_{var_name}.png"

    cmd = get_plot_cmd(
        plotter,
        bin_file,
        var_config["quantity"],
        out_file,
        var_config,
        config["plot_dimension"],
        figsize=config["fig_size_single"],  # use the configured size
    )

    subprocess.run(cmd, check=True)
    return out_file


def create_combined_subplot(
    out_root: Path,
    basename: str,
    timestep: float,
    config: dict,
) -> Path:
    """Create a 2-column combined PNG from all individual variable plots."""
    out_dir = out_root / "combined"
    ensure_dir(out_dir)

    plot_order = config["plot_order"]
    n_plots = len(plot_order)

    ncols = 2
    nrows = math.ceil(n_plots / ncols)

    fig_width, fig_height = config["fig_size_combined"]

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    axes = axes.flatten()

    for ax, var_name in zip(axes, plot_order):
        plot_config = config["plot_vars"][var_name]

        img_path = (
            out_root
            / plot_config["folder_key"]
            / f"{basename}_{var_name}.png"
        )

        if img_path.exists():
            img = plt.imread(img_path)
            ax.imshow(img)
        else:
            ax.text(
                0.5,
                0.5,
                f"Missing:\n{img_path.name}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=20,
            )

        ax.set_title(plot_config["label"], fontsize=plot_title_fontsize, fontweight="bold")
        ax.axis("off")

    # Hide unused panels if plot_order has an odd number of plots.
    for ax in axes[n_plots:]:
        ax.axis("off")

    fig.suptitle(
        f"t = {timestep:.2f} {config['time_label']}",
        fontsize=plot_suptitile_fontsize,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    combo_file = out_dir / f"{basename}_combined.png"

    fig.savefig(
        combo_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    return combo_file


from PIL import Image
import tempfile
import logging

def create_pdf_from_combined(out_root: Path):
    """Combine PNGs into a PDF after flattening transparency."""
    try:
        import img2pdf
    except ImportError:
        print("Skipping PDF: install img2pdf with: pip install img2pdf")
        return

    combined_dir = out_root / "combined"
    pdf_path = combined_dir / "output.pdf"
    combo_files = sorted(combined_dir.glob("*_combined.png"))

    if not combo_files:
        print("No combined PNGs found; PDF was not created.")
        return

    print(f"Creating PDF with {len(combo_files)} pages...")

    with tempfile.TemporaryDirectory() as tmpdir:
        flattened_files = []

        for index, source_path in enumerate(combo_files):
            output_path = Path(tmpdir) / f"frame_{index:04d}.png"

            with Image.open(source_path) as image:
                if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    image = Image.alpha_composite(background, rgba)

                image.convert("RGB").save(output_path, format="PNG")
                flattened_files.append(str(output_path))

        pdf_bytes = img2pdf.convert(flattened_files)

    pdf_path.write_bytes(pdf_bytes)
    print(f"✓ PDF saved: {pdf_path}")


# ============================================================================
# PARALLEL WORKER
# ============================================================================

def process_one_bin(task):
    """
    Process one binary snapshot.

    It makes all individual plots first, then creates its combined frame.
    This function is module-level so ProcessPoolExecutor can use it.
    """
    bin_file, index, out_root, plotter, config = task

    timestep = index * config["dt"]
    basename = bin_file.stem

    for var_name in config["plot_order"]:
        plot_single_quantity(
            bin_file=bin_file,
            var_name=var_name,
            out_root=out_root,
            plotter=plotter,
            config=config,
        )

    combo_file = create_combined_subplot(
        out_root=out_root,
        basename=basename,
        timestep=timestep,
        config=config,
    )

    return bin_file.name, combo_file.name


# ============================================================================
# MAIN
# ============================================================================

def main():
    input_dir = Path(input("Enter path to binary folder: ").strip())
    bin_dir   = Path(os.path.join(input_dir, "bin"))

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    out_root = input_dir / f"bin_outputs"
    ensure_dir(out_root)

    plotter = Path(__file__).parent / "plot_slice2.py"
    if not plotter.exists():
        raise FileNotFoundError(f"Missing plotting script: {plotter}")

    bin_files = sorted(bin_dir.glob("*.bin"))

    if not bin_files:
        print("No .bin files found.")
        return

    # ── Start-index filter ──────────────────────────────────────────────────
    start_input = input(
        f"Start from file number (0–{len(bin_files) - 1}, or press Enter for 0): "
    ).strip()
    start_index = int(start_input) if start_input else 0

    if not (0 <= start_index < len(bin_files)):
        raise ValueError(
            f"Start index {start_index} out of range (0–{len(bin_files) - 1})"
        )

    bin_files = bin_files[start_index:]
    print(f"Skipping first {start_index} file(s). Processing {len(bin_files)} file(s).")
    # ────────────────────────────────────────────────────────────────────────

    num_workers = min(MAX_WORKERS, len(bin_files))

    print(f"Found {len(bin_files)} .bin files.")
    print(f"Using {num_workers} parallel workers.")
    print(f"Output directory: {out_root}")

    tasks = [
        (bin_file, start_index + index, out_root, plotter, config)
        for index, bin_file in enumerate(bin_files)
    ]

    completed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_one_bin, task)
            for task in tasks
        ]

        for future in as_completed(futures):
            bin_name, combined_name = future.result()
            completed += 1
            print(
                f"✓ [{completed}/{len(bin_files)}] "
                f"{combined_name}"
            )

    print("\nAll PNG frames completed.")
    # create_pdf_from_combined(out_root)

    print(f"\nAll done. Output saved to:\n{out_root}")
    print(f"\nInput Folder: {input_dir}")


if __name__ == "__main__":
    main()