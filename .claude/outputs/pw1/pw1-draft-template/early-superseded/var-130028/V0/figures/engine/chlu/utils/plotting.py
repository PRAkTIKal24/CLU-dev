"""Plotting utilities for CHLU experiments."""

import jax
import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def plot_three_panel_trajectories(
    trajectories: dict,
    ground_truth: jnp.ndarray,
    titles: list,
    save_path: str,
    steps_per_cycle: int = None,
    n_cycles_to_show: int = 3,
):
    """
    Plot three-panel figure comparing model trajectories.

    Used for Experiment A: Stability comparison.
    Shows only the last N cycles for focused comparison.

    Args:
        trajectories: Dict with keys "LSTM", "NODE", "CHLU" and trajectory arrays
        ground_truth: Ground truth trajectory (T, 4) [x, y, vx, vy]
        titles: List of 3 subplot titles
        save_path: Path to save figure
        steps_per_cycle: If provided, only plot the last n_cycles_to_show cycles
        n_cycles_to_show: Number of final cycles to display (default: 3)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Extract last N cycles if steps_per_cycle is provided
    if steps_per_cycle is not None:
        steps_to_show = n_cycles_to_show * steps_per_cycle
        gt_plot = ground_truth[-steps_to_show:]
    else:
        gt_plot = ground_truth

    # Plot ground truth on all panels (in gray)
    gt_label = (
        f"Ground Truth (Last {n_cycles_to_show} Cycles)"
        if steps_per_cycle
        else "Ground Truth"
    )
    for ax in axes:
        ax.plot(gt_plot[:, 0], gt_plot[:, 1], "gray", alpha=0.3, linewidth=2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Plot LSTM (left panel) - last N cycles
    lstm_traj = trajectories["LSTM"]
    if steps_per_cycle is not None:
        steps_to_show = n_cycles_to_show * steps_per_cycle
        lstm_plot = lstm_traj[-steps_to_show:]
    else:
        lstm_plot = lstm_traj
    axes[0].plot(lstm_plot[:, 0], lstm_plot[:, 1], "r-", linewidth=1.5)
    axes[0].set_title(titles[0])

    # Plot NODE (middle panel) - last N cycles
    node_traj = trajectories["NODE"]
    if steps_per_cycle is not None:
        steps_to_show = n_cycles_to_show * steps_per_cycle
        node_plot = node_traj[-steps_to_show:]
    else:
        node_plot = node_traj
    axes[1].plot(node_plot[:, 0], node_plot[:, 1], "orange", linewidth=1.5)
    axes[1].set_title(titles[1])

    # Plot CHLU (right panel) - last N cycles
    chlu_traj = trajectories["CHLU"]
    if steps_per_cycle is not None:
        steps_to_show = n_cycles_to_show * steps_per_cycle
        chlu_plot = chlu_traj[-steps_to_show:]
    else:
        chlu_plot = chlu_traj
    axes[2].plot(chlu_plot[:, 0], chlu_plot[:, 1], "g-", linewidth=1.5)
    axes[2].set_title(titles[2])

    # Create unified legend outside the plot area
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=2, alpha=0.3, label=gt_label),
        Line2D([0], [0], color="r", linewidth=1.5, label="LSTM"),
        Line2D([0], [0], color="orange", linewidth=1.5, label="NODE"),
        Line2D([0], [0], color="g", linewidth=1.5, label="CHLU"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        frameon=True,
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved three-panel trajectory plot to {save_path}")


def plot_noise_curves(
    sigmas: jnp.ndarray,
    mse_dict: dict,
    save_path: str,
):
    """
    Plot noise robustness curves.

    Used for Experiment B: Noise rejection comparison.

    Args:
        sigmas: Array of noise levels
        mse_dict: Dict with keys "LSTM", "NODE", "CHLU" and MSE arrays
        save_path: Path to save figure
    """
    plt.figure(figsize=(8, 6))

    # Plot curves
    plt.plot(sigmas, mse_dict["LSTM"], "r-o", linewidth=2, markersize=6, label="LSTM")
    plt.plot(
        sigmas,
        mse_dict["NODE"],
        "orange",
        marker="s",
        linewidth=2,
        markersize=6,
        label="NODE",
    )
    plt.plot(sigmas, mse_dict["CHLU"], "g-^", linewidth=2, markersize=6, label="CHLU")

    plt.xlabel("Noise Sigma (σ)", fontsize=12)
    plt.ylabel("Reconstruction MSE", fontsize=12)
    plt.title("Noise Robustness: The Filter Effect", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved noise curve plot to {save_path}")


def plot_dreaming_grid(
    images: jnp.ndarray,
    save_path: str,
    n_rows: int = 4,
    n_cols: int = 8,
    image_shape: tuple = (28, 28),
):
    """
    Plot grid of evolving images for generative dreaming.

    Used for Experiment C: MNIST dreaming visualization.
    Automatically unnormalizes images from [-1, 1] to [0, 255] for display.

    Args:
        images: Array of images (n_images, height * width) or (n_images, height, width)
                Expected to be in [-1, 1] range (will be unnormalized for display)
        save_path: Path to save figure
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        image_shape: Shape to reshape images to (height, width)
    """
    # Reshape images if needed
    if images.ndim == 2:
        images = images.reshape(-1, *image_shape)

    # Unnormalize from [-1, 1] to [0, 255]
    images = np.array(images)
    images = (images + 1.0) * 127.5
    images = np.clip(images, 0, 255).astype(np.uint8)

    n_images = min(len(images), n_rows * n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    # Handle case where axes is a single Axes object (when n_rows=1 and n_cols=1)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    for i in range(n_images):
        axes[i].imshow(images[i], cmap="gray", vmin=0, vmax=255)
        axes[i].axis("off")

    # Hide unused subplots
    for i in range(n_images, n_rows * n_cols):
        axes[i].axis("off")

    plt.suptitle(
        "CHLU Generative Dreaming: Noise → Digit", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved dreaming grid to {save_path}")


def plot_trajectory_evolution(
    trajectories: dict,
    ground_truth: jnp.ndarray,
    titles: list,
    save_path: str,
    n_snapshots: int = 10,
    steps_per_cycle: int = None,
    n_cycles_solid: int = 3,
):
    """
    Plot trajectory evolution with transparent intermediate steps and final trajectory.

    Shows how each model's trajectory evolves over time with progressive snapshots.
    If steps_per_cycle is provided, shows first cycles very lightly and
    only the last N cycles in solid color.

    Args:
        trajectories: Dict with keys "LSTM", "NODE", "CHLU" and trajectory arrays
        ground_truth: Ground truth trajectory (T, 4) [x, y, vx, vy]
        titles: List of 3 subplot titles
        save_path: Path to save figure
        n_snapshots: Number of intermediate snapshots to show
        steps_per_cycle: If provided, plot first cycles lightly, last n_cycles_solid solid
        n_cycles_solid: Number of final cycles to show in solid color (default: 3)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    model_names = ["LSTM", "NODE", "CHLU"]
    colors = ["red", "orange", "green"]

    for _idx, (ax, model_name, color, title) in enumerate(
        zip(axes, model_names, colors, titles, strict=False)
    ):
        # Plot ground truth - first cycles lightly, last N cycles solid
        if steps_per_cycle is not None:
            steps_solid = n_cycles_solid * steps_per_cycle
            # First cycles slightly transparent
            gt_early = ground_truth[:-steps_solid]
            if len(gt_early) > 0:
                ax.plot(
                    gt_early[:, 0],
                    gt_early[:, 1],
                    "gray",
                    alpha=0.15,
                    linewidth=1,
                    zorder=1,
                )
            # Last N cycles solid
            gt_last = ground_truth[-steps_solid:]
            ax.plot(
                gt_last[:, 0],
                gt_last[:, 1],
                "gray",
                alpha=0.5,
                linewidth=2,
                label=f"Ground Truth (Last {n_cycles_solid} Cycles)",
                zorder=2,
            )
        else:
            ax.plot(
                ground_truth[:, 0],
                ground_truth[:, 1],
                "gray",
                alpha=0.3,
                linewidth=2,
                label="Ground Truth",
                zorder=1,
            )

        traj = trajectories[model_name]
        n_steps = len(traj)

        if steps_per_cycle is not None:
            steps_solid = n_cycles_solid * steps_per_cycle
            # Plot first cycles slightly transparent
            traj_early = traj[:-steps_solid]
            if len(traj_early) > 0:
                ax.plot(
                    traj_early[:, 0],
                    traj_early[:, 1],
                    color=color,
                    alpha=0.15,
                    linewidth=1,
                    zorder=3,
                )

            # Plot last N cycles solid
            traj_last = traj[-steps_solid:]
            ax.plot(
                traj_last[:, 0],
                traj_last[:, 1],
                color=color,
                linewidth=2.5,
                label=f"{model_name} (Last {n_cycles_solid} Cycles)",
                zorder=4,
            )
        else:
            # Original behavior: intermediate snapshots with increasing transparency
            snapshot_indices = np.linspace(
                n_steps // n_snapshots, n_steps, n_snapshots, dtype=int
            )

            for i, snap_idx in enumerate(snapshot_indices[:-1]):
                alpha = 0.1 + (i / n_snapshots) * 0.3  # Fade from 0.1 to 0.4
                ax.plot(
                    traj[:snap_idx, 0],
                    traj[:snap_idx, 1],
                    color=color,
                    alpha=alpha,
                    linewidth=0.8,
                    zorder=2,
                )

            # Plot final trajectory with solid line
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=color,
                linewidth=2,
                label=f"{model_name} (final)",
                zorder=3,
            )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    # Collect handles and labels from all subplots for unified legend
    handles, labels = [], []
    for ax in axes:
        ha, la = ax.get_legend_handles_labels()
        for handle, label in zip(ha, la, strict=False):
            if label not in labels:
                handles.append(handle)
                labels.append(label)

    # Create unified legend outside the plot area
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(labels),
        frameon=True,
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved trajectory evolution plot to {save_path}")


def create_trajectory_animation(
    trajectories: dict,
    ground_truth: jnp.ndarray,
    titles: list,
    save_path: str,
    fps: int = 30,
    n_frames: int = 100,
):
    """
    Create animated GIF showing trajectory evolution over time.

    Args:
        trajectories: Dict with keys "LSTM", "NODE", "CHLU" and trajectory arrays
        ground_truth: Ground truth trajectory (T, 4) [x, y, vx, vy]
        titles: List of 3 subplot titles
        save_path: Path to save GIF (should end in .gif)
        fps: Frames per second
        n_frames: Number of frames in animation
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    model_names = ["LSTM", "NODE", "CHLU"]
    colors = ["red", "orange", "green"]

    # Determine max length
    max_len = max(len(trajectories[name]) for name in model_names)
    frame_indices = np.linspace(10, max_len, n_frames, dtype=int)

    # Initialize plots
    lines = {}
    for _idx, (ax, model_name, color, title) in enumerate(
        zip(axes, model_names, colors, titles, strict=False)
    ):
        # Plot ground truth (static)
        ax.plot(
            ground_truth[:, 0],
            ground_truth[:, 1],
            "gray",
            alpha=0.3,
            linewidth=2,
            label="Ground Truth",
        )

        # Initialize trajectory line
        (line,) = ax.plot([], [], color=color, linewidth=2, label=model_name)
        lines[model_name] = line

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Set axis limits based on data
        all_x = np.concatenate([ground_truth[:, 0], trajectories[model_name][:, 0]])
        all_y = np.concatenate([ground_truth[:, 1], trajectories[model_name][:, 1]])
        margin = 0.1
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

    # Collect handles and labels from first subplot for unified legend
    handles, labels = axes[0].get_legend_handles_labels()

    # Create unified legend outside the plot area
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(labels),
        frameon=True,
        fontsize=10,
    )

    def update(frame_idx):
        """Update function for animation."""
        idx = frame_indices[frame_idx]
        for model_name in model_names:
            traj = trajectories[model_name]
            end_idx = min(idx, len(traj))
            lines[model_name].set_data(traj[:end_idx, 0], traj[:end_idx, 1])
        return list(lines.values())

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 / fps, blit=True
    )

    plt.tight_layout()
    anim.save(save_path, writer="pillow", fps=fps)
    plt.close()

    print(f"Saved trajectory animation to {save_path}")


def plot_sine_wave_comparison(
    clean_data: jnp.ndarray,
    noisy_data: jnp.ndarray,
    predictions: dict,
    save_path: str,
    n_examples: int = 3,
    sigma: float = 0.5,
):
    """
    Plot expected vs generated sine waves for each algorithm.

    Args:
        clean_data: Clean test data (n_waves, steps, 2)
        noisy_data: Noisy test data (n_waves, steps, 2)
        predictions: Dict with keys "LSTM", "NODE", "CHLU" and prediction arrays
        save_path: Path to save figure
        n_examples: Number of example waves to show
        sigma: Noise level used
    """
    n_examples = min(n_examples, len(clean_data))
    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 4 * n_examples))

    if n_examples == 1:
        axes = axes.reshape(1, -1)

    model_names = ["LSTM", "NODE", "CHLU"]
    colors = ["red", "orange", "green"]

    for row in range(n_examples):
        clean_seq = clean_data[row]
        noisy_seq = noisy_data[row]
        time_steps = np.arange(len(clean_seq))

        for col, (model_name, color) in enumerate(
            zip(model_names, colors, strict=False)
        ):
            ax = axes[row, col]
            pred_seq = predictions[model_name][row]

            # Plot clean (expected)
            ax.plot(
                time_steps,
                clean_seq[:, 0],
                "k-",
                linewidth=2,
                label="Expected",
                alpha=0.7,
            )

            # Plot noisy input
            ax.scatter(
                time_steps[::5],
                noisy_seq[::5, 0],
                c="gray",
                s=10,
                alpha=0.4,
                label=f"Noisy Input (σ={sigma})",
            )

            # Plot prediction
            ax.plot(
                time_steps,
                pred_seq[:, 0],
                color=color,
                linewidth=2,
                label=f"{model_name} Output",
                linestyle="--",
            )

            ax.set_xlabel("Time Step")
            ax.set_ylabel("Amplitude")
            ax.set_title(f"{model_name} - Wave {row + 1}")
            ax.grid(True, alpha=0.3)

    # Collect handles and labels from first row for unified legend
    handles, labels = axes[0, 0].get_legend_handles_labels()

    plt.suptitle(
        f"Sine Wave Reconstruction (σ = {sigma})", fontsize=14, fontweight="bold"
    )
    # Create unified legend outside the plot area
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=True,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved sine wave comparison to {save_path}")


def plot_phase_space(
    clean_data: jnp.ndarray,
    noisy_data: jnp.ndarray,
    predictions: dict,
    save_path: str,
    n_examples: int = 3,
    sigma: float = 0.5,
):
    """
    Plot phase space (q vs p) for sine wave predictions.

    Args:
        clean_data: Clean test data (n_waves, steps, 2) where dim 0 = q, dim 1 = p
        noisy_data: Noisy test data (n_waves, steps, 2)
        predictions: Dict with keys "LSTM", "NODE", "CHLU" and prediction arrays
        save_path: Path to save figure
        n_examples: Number of example waves to show
        sigma: Noise level used
    """
    n_examples = min(n_examples, len(clean_data))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    model_names = ["LSTM", "NODE", "CHLU"]
    colors = ["red", "orange", "green"]

    for _col, (model_name, color, ax) in enumerate(
        zip(model_names, colors, axes, strict=False)
    ):
        # Plot all examples for this model
        for row in range(n_examples):
            clean_seq = clean_data[row]
            noisy_seq = noisy_data[row]
            pred_seq = predictions[model_name][row]

            # Plot clean trajectory (expected)
            if row == 0:
                ax.plot(
                    clean_seq[:, 0],
                    clean_seq[:, 1],
                    "k-",
                    linewidth=1.5,
                    alpha=0.5,
                    label="Expected",
                )
            else:
                ax.plot(
                    clean_seq[:, 0], clean_seq[:, 1], "k-", linewidth=1.5, alpha=0.5
                )

            # Plot noisy input points
            if row == 0:
                ax.scatter(
                    noisy_seq[::10, 0],
                    noisy_seq[::10, 1],
                    c="gray",
                    s=15,
                    alpha=0.3,
                    label=f"Noisy (σ={sigma})",
                )
            else:
                ax.scatter(
                    noisy_seq[::10, 0], noisy_seq[::10, 1], c="gray", s=15, alpha=0.3
                )

            # Mark initial condition with blue triangle
            if row == 0:
                ax.scatter(
                    noisy_seq[0, 0],
                    noisy_seq[0, 1],
                    marker="^",
                    c="blue",
                    s=150,
                    edgecolors="darkblue",
                    linewidths=1.5,
                    zorder=10,
                    label="Initial Conditions",
                )
            else:
                ax.scatter(
                    noisy_seq[0, 0],
                    noisy_seq[0, 1],
                    marker="^",
                    c="blue",
                    s=150,
                    edgecolors="darkblue",
                    linewidths=1.5,
                    zorder=10,
                )

            # Plot prediction
            if row == 0:
                ax.plot(
                    pred_seq[:, 0],
                    pred_seq[:, 1],
                    color=color,
                    linewidth=2,
                    linestyle="--",
                    label=f"{model_name}",
                )
            else:
                ax.plot(
                    pred_seq[:, 0],
                    pred_seq[:, 1],
                    color=color,
                    linewidth=2,
                    linestyle="--",
                )

        ax.set_xlabel("Position (q)")
        ax.set_ylabel("Momentum (p)")
        ax.set_title(f"{model_name} Phase Space")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    # Collect handles and labels from first subplot for unified legend
    handles, labels = axes[0].get_legend_handles_labels()

    plt.suptitle(
        f"Phase Space Trajectories (σ = {sigma})", fontsize=14, fontweight="bold"
    )
    # Create unified legend outside the plot area
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(labels),
        frameon=True,
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved phase space plot to {save_path}")


def plot_multi_noise_grid(
    clean_data: jnp.ndarray,
    noise_levels_data: dict,
    save_path: str,
    example_idx: int = 0,
):
    """
    Plot multi-level noise comparison grid.

    Shows predictions at low, medium, and high noise levels for each model.
    Layout: 3 rows (LSTM, NODE, CHLU) x 3 columns (low, medium, high noise)

    Args:
        clean_data: Clean test data (n_waves, steps, 2)
        noise_levels_data: Dict with structure:
            {
                'sigmas': [low_sigma, mid_sigma, high_sigma],
                'noisy_inputs': [low_noisy, mid_noisy, high_noisy],
                'predictions': {
                    'LSTM': [low_pred, mid_pred, high_pred],
                    'NODE': [low_pred, mid_pred, high_pred],
                    'CHLU': [low_pred, mid_pred, high_pred]
                }
            }
        save_path: Path to save figure
        example_idx: Which test example to show (default: 0)
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    model_names = ["LSTM", "NODE", "CHLU"]
    colors = ["red", "orange", "green"]
    sigmas = noise_levels_data["sigmas"]
    noisy_inputs = noise_levels_data["noisy_inputs"]
    predictions = noise_levels_data["predictions"]

    clean_seq = clean_data[example_idx]
    time_steps = np.arange(len(clean_seq))

    # Column titles
    noise_labels = ["Low Noise", "Medium Noise", "High Noise"]

    for row, (model_name, color) in enumerate(zip(model_names, colors, strict=False)):
        for col, (sigma, noisy_data, noise_label) in enumerate(
            zip(sigmas, noisy_inputs, noise_labels, strict=False)
        ):
            ax = axes[row, col]

            noisy_seq = noisy_data[example_idx]
            pred_seq = predictions[model_name][col][example_idx]

            # Plot clean signal (ground truth)
            ax.plot(
                time_steps,
                clean_seq[:, 0],
                "k-",
                linewidth=2.5,
                label="Clean Signal",
                alpha=0.7,
                zorder=3,
            )

            # Plot noisy input (scatter to show noise)
            ax.scatter(
                time_steps[::3],
                noisy_seq[::3, 0],
                c="gray",
                s=12,
                alpha=0.4,
                label="Noisy Input",
                zorder=1,
            )

            # Plot model prediction
            ax.plot(
                time_steps,
                pred_seq[:, 0],
                color=color,
                linewidth=2,
                label=f"{model_name} Prediction",
                linestyle="--",
                zorder=2,
            )

            # Styling
            ax.set_xlabel("Time Step", fontsize=10)
            ax.set_ylabel("Amplitude", fontsize=10)
            ax.set_ylim(-3, 3)  # Fixed y-axis range for easy comparison
            ax.grid(True, alpha=0.3)

            # Title at top of each column
            if row == 0:
                ax.set_title(
                    f"{noise_label}\n(σ = {sigma:.2f})", fontsize=11, fontweight="bold"
                )

            # Y-axis label on left side
            if col == 0:
                ax.text(
                    -0.25,
                    0.5,
                    model_name,
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    rotation=90,
                    va="center",
                    ha="center",
                )

    # Collect handles and labels from first subplot for unified legend
    handles, labels = axes[0, 0].get_legend_handles_labels()

    plt.suptitle(
        "Multi-Level Noise Comparison: Model Predictions Across Noise Levels",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    # Create unified legend outside the plot area
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=True,
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved multi-noise grid to {save_path}")


def plot_noise_heatmap(
    sigmas: jnp.ndarray,
    temporal_errors: dict,
    save_path: str,
):
    """
    Plot noise level heatmap showing error evolution over time.

    Creates a 2D heatmap for each model showing how reconstruction error
    varies across noise levels (y-axis) and time steps (x-axis).

    Args:
        sigmas: Array of noise levels (n_sigma,)
        temporal_errors: Dict with structure:
            {
                'LSTM': array of shape (n_sigma, n_steps),
                'NODE': array of shape (n_sigma, n_steps),
                'CHLU': array of shape (n_sigma, n_steps)
            }
            Each entry [i, t] contains the mean squared error at noise level i and timestep t
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    model_names = ["LSTM", "NODE", "CHLU"]
    cmaps = ["Reds", "Oranges", "Greens"]

    for ax, model_name, cmap in zip(axes, model_names, cmaps, strict=False):
        error_matrix = np.array(temporal_errors[model_name])
        n_sigma, n_steps = error_matrix.shape

        # Create heatmap
        im = ax.imshow(
            error_matrix,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            extent=[0, n_steps, float(sigmas[0]), float(sigmas[-1])],
            interpolation="bilinear",
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Mean Squared Error", fontsize=10)

        # Styling
        ax.set_xlabel("Time Step", fontsize=11)
        ax.set_ylabel("Noise Level (σ)", fontsize=11)
        ax.set_title(f"{model_name} Error Heatmap", fontsize=12, fontweight="bold")
        ax.grid(False)

    plt.suptitle(
        "Temporal Error Evolution Across Noise Levels", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved noise heatmap to {save_path}")


def plot_kinetic_energy_vs_time(
    clean_data: jnp.ndarray,
    predictions: dict,
    save_path: str,
    chlu_model=None,
    dt: float = 0.01,
    n_examples: int = 3,
    sigma: float = 0.5,
    rest_mass: float = 1.0,
    c: float = 5.0,
):
    """
    Plot Kinetic Energy vs time for all three models (CHLU with learned KE).

    LSTM/NODE use relativistic kinetic energy from momentum values.
    CHLU computes kinetic energy from its learned relativistic Hamiltonian.

    Args:
        clean_data: Clean test data (n_waves, steps, 2)
        predictions: Dict with keys "LSTM", "NODE", "CHLU" and prediction arrays
        save_path: Path to save figure
        chlu_model: CHLU model instance (to compute learned kinetic energy)
        dt: Time step size
        n_examples: Number of example waves to show
        sigma: Noise level used
        rest_mass: Rest mass for relativistic kinetic energy (default: 1.0)
        c: Speed of causality (default: 5.0)
    """
    n_examples = min(n_examples, len(clean_data))
    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 4 * n_examples))

    if n_examples == 1:
        axes = axes.reshape(1, -1)

    model_names = ["LSTM", "NODE", "CHLU"]
    colors = ["red", "orange", "green"]

    def compute_relativistic_kinetic(p, rest_mass, c):
        """Compute relativistic kinetic energy: sqrt(p^2 + (m*c)^2)."""
        return jnp.sqrt(p**2 + (rest_mass * c) ** 2)

    def compute_chlu_kinetic(q, p, chlu_model):
        """Extract kinetic energy from CHLU's Hamiltonian."""
        # Get mass parameters
        M = jax.nn.softplus(chlu_model.log_mass)
        M_inv = 1.0 / (M + 1e-6)

        if chlu_model.kinetic_mode == "relativistic":
            p_norm_squared = jnp.sum((p * p) * M_inv)
            rest_energy = (chlu_model.rest_mass * chlu_model.c) ** 2
            return jnp.sqrt(p_norm_squared + rest_energy)
        elif chlu_model.kinetic_mode == "newtonian_learned":
            return 0.5 * jnp.sum((p * p) * M_inv)
        else:  # newtonian_identity
            return 0.5 * jnp.sum(p * p)

    for row in range(n_examples):
        clean_seq = clean_data[row]

        # Limit to first 50 timesteps
        n_steps = min(50, len(clean_seq))
        clean_seq = clean_seq[:n_steps]
        time_steps = np.arange(n_steps) * dt

        # Compute clean trajectory kinetic energy
        clean_KE = np.array(
            [
                compute_relativistic_kinetic(clean_seq[t, 1], rest_mass, c)
                for t in range(len(clean_seq))
            ]
        )

        for col, (model_name, color) in enumerate(
            zip(model_names, colors, strict=False)
        ):
            ax = axes[row, col]
            pred_seq = predictions[model_name][row][:n_steps]

            # Plot clean trajectory kinetic energy
            ax.plot(
                time_steps, clean_KE, "k-", linewidth=2, label="Clean Signal", alpha=0.7
            )

            # Compute and plot model prediction kinetic energy
            if model_name == "CHLU" and chlu_model is not None:
                # Use CHLU's learned kinetic energy
                pred_KE = np.array(
                    [
                        compute_chlu_kinetic(
                            pred_seq[t, 0:1], pred_seq[t, 1:2], chlu_model
                        )
                        for t in range(len(pred_seq))
                    ]
                )
            else:
                # Use relativistic kinetic energy from p for LSTM/NODE
                pred_KE = np.array(
                    [
                        compute_relativistic_kinetic(pred_seq[t, 1], rest_mass, c)
                        for t in range(len(pred_seq))
                    ]
                )

            ax.plot(
                time_steps,
                pred_KE,
                color=color,
                linewidth=2,
                label=f"{model_name} Prediction",
                linestyle="--",
            )

            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_ylabel("Kinetic Energy", fontsize=10)
            ax.set_title(f"{model_name} - Wave {row + 1}", fontsize=11)
            ax.grid(True, alpha=0.3)

    # Collect handles and labels from first row for unified legend
    handles, labels = axes[0, 0].get_legend_handles_labels()

    plt.suptitle(
        f"Kinetic Energy vs Time (σ = {sigma}) - Relativistic",
        fontsize=14,
        fontweight="bold",
    )
    # Create unified legend outside the plot area
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        frameon=True,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved Kinetic Energy vs time (relativistic) plot to {save_path}")


def plot_kinetic_energy_vs_time_unified(
    clean_data: jnp.ndarray,
    predictions: dict,
    save_path: str,
    dt: float = 0.01,
    n_examples: int = 3,
    sigma: float = 0.5,
):
    """
    Plot Kinetic Energy vs time using same formula for all models.

    All models use Newtonian kinetic energy: KE = 0.5 * v^2
    where v = dq/dt is computed from consecutive position values.
    This provides an apples-to-apples comparison based solely on predicted trajectories.

    Args:
        clean_data: Clean test data (n_waves, steps, 2)
        predictions: Dict with keys "LSTM", "NODE", "CHLU" and prediction arrays
        save_path: Path to save figure
        dt: Time step size
        n_examples: Number of example waves to show
        sigma: Noise level used
    """
    n_examples = min(n_examples, len(clean_data))
    fig, axes = plt.subplots(n_examples, 3, figsize=(15, 4 * n_examples))

    if n_examples == 1:
        axes = axes.reshape(1, -1)

    model_names = ["LSTM", "NODE", "CHLU"]
    colors = ["red", "orange", "green"]

    def compute_newtonian_kinetic_from_trajectory(q_trajectory, dt):
        """Compute KE = 0.5 * v^2 where v = dq/dt from consecutive positions."""
        # Compute velocity using finite differences
        v = np.gradient(q_trajectory, dt)
        # Compute kinetic energy
        return 0.5 * v**2

    for row in range(n_examples):
        clean_seq = clean_data[row]

        # Limit to first 50 timesteps
        n_steps = min(50, len(clean_seq))
        clean_seq = clean_seq[:n_steps]
        time_steps = np.arange(n_steps) * dt

        # Compute clean trajectory kinetic energy from position gradient
        clean_KE = compute_newtonian_kinetic_from_trajectory(
            np.array(clean_seq[:, 0]), dt
        )

        for col, (model_name, color) in enumerate(
            zip(model_names, colors, strict=False)
        ):
            ax = axes[row, col]
            pred_seq = predictions[model_name][row][:n_steps]

            # Plot clean trajectory kinetic energy
            ax.plot(
                time_steps, clean_KE, "k-", linewidth=2, label="Clean Signal", alpha=0.7
            )

            # Compute prediction kinetic energy from position gradient (same for all)
            pred_KE = compute_newtonian_kinetic_from_trajectory(
                np.array(pred_seq[:, 0]), dt
            )

            ax.plot(
                time_steps,
                pred_KE,
                color=color,
                linewidth=2,
                label=f"{model_name} Prediction",
                linestyle="--",
            )

            ax.set_xlabel("Time (s)", fontsize=10)
            ax.set_ylabel("Kinetic Energy", fontsize=10)
            ax.set_title(f"{model_name} - Wave {row + 1}", fontsize=11)
            ax.grid(True, alpha=0.3)

    # Collect handles and labels from first row for unified legend
    handles, labels = axes[0, 0].get_legend_handles_labels()

    plt.suptitle(
        f"Kinetic Energy vs Time (σ = {sigma}) - Newtonian (v=dq/dt)",
        fontsize=14,
        fontweight="bold",
    )
    # Create unified legend outside the plot area
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=2,
        frameon=True,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved Kinetic Energy vs time (Newtonian) plot to {save_path}")


def plot_potential_landscape_2d(
    chlu_model,
    trajectory: jnp.ndarray,
    save_path: str,
    grid_resolution: int = 100,
    trajectory_label: str = "CHLU Trajectory",
):
    """
    Plot 2D potential landscape V(q) with trajectory overlay.

    Creates a heatmap/contour plot of the learned potential energy function
    over the position space with the trajectory overlaid on top.

    Args:
        chlu_model: Trained CHLU model with learned potential_net
        trajectory: Trajectory array (T, 4) with [x, y, vx, vy]
        save_path: Path to save figure
        grid_resolution: Number of grid points along each axis (default: 100)
        trajectory_label: Label for the trajectory (default: "CHLU Trajectory")
    """
    # Extract position coordinates from trajectory
    x_traj = np.array(trajectory[:, 0])
    y_traj = np.array(trajectory[:, 1])

    # Create grid bounds with some margin around trajectory
    margin = 0.3
    x_min, x_max = x_traj.min() - margin, x_traj.max() + margin
    y_min, y_max = y_traj.min() - margin, y_traj.max() + margin

    # Create meshgrid
    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x, y)

    # Compute potential at each grid point
    V = np.zeros_like(X)
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            q = jnp.array([X[i, j], Y[i, j]])
            V[i, j] = chlu_model.potential_net(q)

    # Create figure with two panels: contour + 3D view
    fig = plt.figure(figsize=(16, 6))

    # Left panel: 2D contour plot
    ax1 = fig.add_subplot(121)

    # Plot filled contours
    contourf = ax1.contourf(X, Y, V, levels=20, cmap="viridis", alpha=0.8)

    # Plot contour lines
    contour = ax1.contour(X, Y, V, levels=10, colors="white", alpha=0.4, linewidths=0.5)
    ax1.clabel(contour, inline=True, fontsize=8, fmt="%.2f")

    # Overlay trajectory
    ax1.plot(x_traj, y_traj, "r-", linewidth=2.5, label=trajectory_label, alpha=0.9)
    ax1.scatter(
        x_traj[0],
        y_traj[0],
        c="lime",
        s=150,
        marker="o",
        edgecolors="darkgreen",
        linewidths=2,
        zorder=10,
        label="Start",
    )
    ax1.scatter(
        x_traj[-1],
        y_traj[-1],
        c="red",
        s=150,
        marker="X",
        edgecolors="darkred",
        linewidths=2,
        zorder=10,
        label="End",
    )

    # Formatting
    ax1.set_xlabel("x", fontsize=12)
    ax1.set_ylabel("y", fontsize=12)
    ax1.set_title("Learned Potential Landscape V(q)", fontsize=14, fontweight="bold")
    ax1.set_aspect("equal")
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(contourf, ax=ax1)
    cbar.set_label("Potential Energy V(q)", fontsize=11)

    # Right panel: 3D surface plot
    ax2 = fig.add_subplot(122, projection="3d")

    # Plot surface
    surf = ax2.plot_surface(
        X, Y, V, cmap="viridis", alpha=0.7, edgecolor="none", antialiased=True
    )

    # Plot trajectory on surface
    V_traj = np.array(
        [
            chlu_model.potential_net(jnp.array([x_traj[i], y_traj[i]]))
            for i in range(len(x_traj))
        ]
    )
    ax2.plot(x_traj, y_traj, V_traj, "r-", linewidth=2.5, label=trajectory_label)
    ax2.scatter(
        x_traj[0],
        y_traj[0],
        V_traj[0],
        c="lime",
        s=100,
        marker="o",
        edgecolors="darkgreen",
        linewidths=2,
        zorder=10,
    )
    ax2.scatter(
        x_traj[-1],
        y_traj[-1],
        V_traj[-1],
        c="red",
        s=100,
        marker="X",
        edgecolors="darkred",
        linewidths=2,
        zorder=10,
    )

    # Formatting
    ax2.set_xlabel("x", fontsize=11)
    ax2.set_ylabel("y", fontsize=11)
    ax2.set_zlabel("V(q)", fontsize=11)
    ax2.set_title("3D Potential Surface", fontsize=14, fontweight="bold")
    ax2.view_init(elev=25, azim=45)

    # Add colorbar
    cbar2 = plt.colorbar(surf, ax=ax2, shrink=0.5, aspect=10)
    cbar2.set_label("V(q)", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved 2D potential landscape to {save_path}")


def plot_potential_surface_3d(
    chlu_model,
    trajectory: jnp.ndarray,
    save_path: str,
    grid_resolution: int = 50,
):
    """
    Plot 3D potential surface V(q) with trajectory.

    Creates a 3D surface plot showing the learned potential energy landscape
    with the trajectory plotted as a path on the surface.

    Args:
        chlu_model: Trained CHLU model with learned potential_net
        trajectory: Trajectory array (T, 4) with [x, y, vx, vy]
        save_path: Path to save figure
        grid_resolution: Number of grid points along each axis (default: 50)
    """

    # Extract position coordinates from trajectory
    x_traj = np.array(trajectory[:, 0])
    y_traj = np.array(trajectory[:, 1])

    # Create grid bounds with some margin around trajectory
    margin = 0.3
    x_min, x_max = x_traj.min() - margin, x_traj.max() + margin
    y_min, y_max = y_traj.min() - margin, y_traj.max() + margin

    # Create meshgrid
    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x, y)

    # Compute potential at each grid point
    V = np.zeros_like(X)
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            q = jnp.array([X[i, j], Y[i, j]])
            V[i, j] = chlu_model.potential_net(q)

    # Compute potential along trajectory
    V_traj = np.array(
        [
            chlu_model.potential_net(jnp.array([x_traj[i], y_traj[i]]))
            for i in range(len(x_traj))
        ]
    )

    # Create figure with multiple viewing angles
    fig = plt.figure(figsize=(18, 6))

    for idx, (elev, azim, title_suffix) in enumerate(
        [(30, 45, "(View 1)"), (20, 135, "(View 2)"), (60, 225, "(View 3)")]
    ):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")

        # Plot surface with gradient coloring
        ax.plot_surface(
            X,
            Y,
            V,
            cmap="viridis",
            alpha=0.6,
            edgecolor="none",
            antialiased=True,
            shade=True,
        )

        # Plot trajectory on the surface
        ax.plot(
            x_traj,
            y_traj,
            V_traj,
            "r-",
            linewidth=3,
            label="CHLU Trajectory",
            zorder=10,
        )

        # Mark start and end points
        ax.scatter(
            x_traj[0],
            y_traj[0],
            V_traj[0],
            c="lime",
            s=150,
            marker="o",
            edgecolors="darkgreen",
            linewidths=2,
            zorder=15,
            label="Start",
        )
        ax.scatter(
            x_traj[-1],
            y_traj[-1],
            V_traj[-1],
            c="red",
            s=150,
            marker="X",
            edgecolors="darkred",
            linewidths=2,
            zorder=15,
            label="End",
        )

        # Formatting
        ax.set_xlabel("x Position", fontsize=10)
        ax.set_ylabel("y Position", fontsize=10)
        ax.set_zlabel("Potential V(q)", fontsize=10)
        ax.set_title(
            f"Learned Potential Surface {title_suffix}", fontsize=12, fontweight="bold"
        )
        ax.view_init(elev=elev, azim=azim)

        if idx == 0:
            ax.legend(fontsize=9, loc="upper left")

    plt.suptitle(
        "3D Potential Energy Landscape - Multiple Views", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved 3D potential surface to {save_path}")


def plot_force_field(
    chlu_model,
    trajectory: jnp.ndarray,
    save_path: str,
    grid_resolution: int = 20,
):
    """
    Plot force field F = -∇V(q) with trajectory overlay.

    Creates a vector field showing the forces derived from the learned potential,
    with arrows color-coded by magnitude. The trajectory is overlaid to show
    how it flows through the force field.

    Args:
        chlu_model: Trained CHLU model with learned potential_net
        trajectory: Trajectory array (T, 4) with [x, y, vx, vy]
        save_path: Path to save figure
        grid_resolution: Number of grid points along each axis (default: 20)
    """
    # Extract position coordinates from trajectory
    x_traj = np.array(trajectory[:, 0])
    y_traj = np.array(trajectory[:, 1])

    # Create grid bounds with some margin around trajectory
    margin = 0.3
    x_min, x_max = x_traj.min() - margin, x_traj.max() + margin
    y_min, y_max = y_traj.min() - margin, y_traj.max() + margin

    # Create meshgrid for vector field
    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x, y)

    # Compute gradient of potential at each grid point: F = -∇V
    Fx = np.zeros_like(X)
    Fy = np.zeros_like(Y)

    # Use JAX's automatic differentiation to compute gradient
    grad_V = jax.grad(chlu_model.potential_net)

    for i in range(grid_resolution):
        for j in range(grid_resolution):
            q = jnp.array([X[i, j], Y[i, j]])
            grad = grad_V(q)
            # Force is negative gradient of potential
            Fx[i, j] = -grad[0]
            Fy[i, j] = -grad[1]

    # Compute force magnitude for coloring
    F_mag = np.sqrt(Fx**2 + Fy**2)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot vector field with color based on magnitude
    quiver = ax.quiver(
        X,
        Y,
        Fx,
        Fy,
        F_mag,
        cmap="plasma",
        scale=20,
        scale_units="xy",
        width=0.004,
        alpha=0.7,
        pivot="mid",
    )

    # Add colorbar for force magnitude
    cbar = plt.colorbar(quiver, ax=ax)
    cbar.set_label("Force Magnitude |F|", fontsize=12)

    # Overlay trajectory
    ax.plot(
        x_traj,
        y_traj,
        "cyan",
        linewidth=3,
        label="CHLU Trajectory",
        alpha=0.9,
        zorder=10,
    )
    ax.scatter(
        x_traj[0],
        y_traj[0],
        c="lime",
        s=200,
        marker="o",
        edgecolors="darkgreen",
        linewidths=2.5,
        zorder=15,
        label="Start",
    )
    ax.scatter(
        x_traj[-1],
        y_traj[-1],
        c="red",
        s=200,
        marker="X",
        edgecolors="darkred",
        linewidths=2.5,
        zorder=15,
        label="End",
    )

    # Formatting
    ax.set_xlabel("x Position", fontsize=13)
    ax.set_ylabel("y Position", fontsize=13)
    ax.set_title("Force Field: F = -∇V(q)", fontsize=15, fontweight="bold")
    ax.set_aspect("equal")
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved force field plot to {save_path}")


def plot_energy_conservation(
    chlu_model,
    trajectories: dict,
    save_path: str,
    dt: float = 0.01,
    n_steps_to_plot: int = None,
):
    """
    Plot energy conservation comparison across models.

    Shows total energy H(q,p), kinetic energy T(p), and potential energy V(q)
    over time for CHLU, NODE, and LSTM. CHLU should show flat energy (conserved),
    while baselines drift.

    Args:
        chlu_model: Trained CHLU model (to compute Hamiltonian)
        trajectories: Dict with keys "LSTM", "NODE", "CHLU" and trajectory arrays
        save_path: Path to save figure
        dt: Time step size
        n_steps_to_plot: Number of steps to plot (default: all)
    """
    model_names = ["LSTM", "NODE", "CHLU"]
    colors = ["red", "orange", "green"]

    # Create figure with 4 panels: Total Energy, Kinetic, Potential, and Combined
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for model_name, color in zip(model_names, colors, strict=False):
        traj = trajectories[model_name]

        # Limit steps if specified
        if n_steps_to_plot is not None:
            traj = traj[:n_steps_to_plot]

        n_steps = len(traj)
        time = np.arange(n_steps) * dt

        # Extract positions and momenta
        q_seq = traj[:, :2]  # First 2 components are positions
        p_seq = traj[:, 2:]  # Last 2 components are momenta

        # Compute energies at each timestep
        H_seq = []
        T_seq = []
        V_seq = []

        for i in range(n_steps):
            q = jnp.array(q_seq[i])
            p = jnp.array(p_seq[i])

            # Compute Hamiltonian (total energy)
            H = chlu_model.H(q, p)
            H_seq.append(H)

            # Compute kinetic energy
            M = jax.nn.softplus(chlu_model.log_mass)
            M_inv = 1.0 / (M + 1e-6)

            if chlu_model.kinetic_mode == "relativistic":
                p_norm_squared = jnp.sum((p * p) * M_inv)
                rest_energy = (chlu_model.rest_mass * chlu_model.c) ** 2
                T = chlu_model.c * jnp.sqrt(p_norm_squared + rest_energy)
            elif chlu_model.kinetic_mode == "newtonian_learned":
                T = 0.5 * jnp.sum((p * p) * M_inv)
            else:  # newtonian_identity
                T = 0.5 * jnp.sum(p * p)

            T_seq.append(T)

            # Compute potential energy
            V = chlu_model.potential_net(q)
            V_seq.append(V)

        H_seq = np.array(H_seq)
        T_seq = np.array(T_seq)
        V_seq = np.array(V_seq)

        # Panel 1: Total Energy (Hamiltonian)
        axes[0].plot(time, H_seq, color=color, linewidth=2, label=model_name, alpha=0.8)

        # Panel 2: Kinetic Energy
        axes[1].plot(time, T_seq, color=color, linewidth=2, label=model_name, alpha=0.8)

        # Panel 3: Potential Energy
        axes[2].plot(time, V_seq, color=color, linewidth=2, label=model_name, alpha=0.8)

        # Panel 4: Kinetic + Potential (stacked or separate)
        axes[3].plot(
            time,
            T_seq,
            color=color,
            linewidth=1.5,
            linestyle="--",
            alpha=0.6,
            label=f"{model_name} T",
        )
        axes[3].plot(
            time,
            V_seq,
            color=color,
            linewidth=1.5,
            linestyle=":",
            alpha=0.6,
            label=f"{model_name} V",
        )

    # Formatting for each panel
    axes[0].set_xlabel("Time (s)", fontsize=12)
    axes[0].set_ylabel("Total Energy H(q,p)", fontsize=12)
    axes[0].set_title("Total Energy (Hamiltonian)", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10, loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Time (s)", fontsize=12)
    axes[1].set_ylabel("Kinetic Energy T(p)", fontsize=12)
    axes[1].set_title("Kinetic Energy", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10, loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Time (s)", fontsize=12)
    axes[2].set_ylabel("Potential Energy V(q)", fontsize=12)
    axes[2].set_title("Potential Energy", fontsize=13, fontweight="bold")
    axes[2].legend(fontsize=10, loc="best")
    axes[2].grid(True, alpha=0.3)

    axes[3].set_xlabel("Time (s)", fontsize=12)
    axes[3].set_ylabel("Energy", fontsize=12)
    axes[3].set_title("Energy Components (T and V)", fontsize=13, fontweight="bold")
    axes[3].legend(fontsize=9, loc="best", ncol=2)
    axes[3].grid(True, alpha=0.3)

    plt.suptitle(
        "Energy Conservation Comparison: CHLU vs Baselines",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved energy conservation plot to {save_path}")


def plot_goldstone_summary(
    mu_sq,
    retention_curves: dict,
    noether: tuple,
    theta,
    save_path: str,
    gamma: float = None,
):
    """
    Four-panel summary for Experiment D (SO(2) Goldstone memory, V2).

    Panels:
        1. Spectral-mass spectrum mu_k^2 at the settled point (log-y stems).
        2. Per-mode retention |amplitude(n)|/|amplitude(0)| vs steps (log-y).
        3. Noether charge Q(n) vs the exact (1-gamma)^n * Q0 decay law.
        4. Coset angle theta(n) (unwrapped) — the latch plateau.

    Args:
        mu_sq: (dim,) spectral masses squared (ascending).
        retention_curves: dict label -> 1D retention series.
        noether: (Q, Q_pred) 1D arrays (pass (None, None) to skip the panel).
        theta: 1D unwrapped coset-angle series (or None to skip).
        save_path: output PNG path.
        gamma: probe friction (annotation only).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Spectrum
    mu_sq = np.asarray(mu_sq)
    idx = np.arange(len(mu_sq))
    axes[0, 0].bar(idx, np.abs(mu_sq), color=np.where(mu_sq < 0, "red", "steelblue"))
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("mode k", fontsize=12)
    axes[0, 0].set_ylabel(r"$|\mu_k^2|$", fontsize=12)
    axes[0, 0].set_title(
        "Spectral-mass spectrum (red = negative / saddle)", fontsize=13, fontweight="bold"
    )
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Retention
    for label, series in retention_curves.items():
        s = np.asarray(series)
        axes[0, 1].plot(np.clip(s, 1e-20, None), label=label)
    axes[0, 1].axhline(0.5, color="gray", linestyle="--", alpha=0.7, label="half-life")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("steps n", fontsize=12)
    axes[0, 1].set_ylabel("retention a(n)/a(0)", fontsize=12)
    title = "Per-mode retention"
    if gamma is not None:
        title += f" (gamma={gamma})"
    axes[0, 1].set_title(title, fontsize=13, fontweight="bold")
    axes[0, 1].legend(fontsize=9, loc="best")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Noether charge decay
    Q, Q_pred = noether
    if Q is not None:
        axes[1, 0].plot(np.asarray(Q), label="measured Q(n)")
        axes[1, 0].plot(
            np.asarray(Q_pred), linestyle="--", label=r"$(1-\gamma)^n Q_0$ (exact law)"
        )
        axes[1, 0].legend(fontsize=10, loc="best")
    axes[1, 0].set_xlabel("steps n", fontsize=12)
    axes[1, 0].set_ylabel(r"$Q = q_0 p_1 - q_1 p_0$", fontsize=12)
    axes[1, 0].set_title("Noether charge (write current)", fontsize=13, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Coset angle
    if theta is not None:
        axes[1, 1].plot(np.asarray(theta), color="darkgreen")
    axes[1, 1].set_xlabel("steps n", fontsize=12)
    axes[1, 1].set_ylabel(r"coset angle $\vartheta(n)$ [rad]", fontsize=12)
    axes[1, 1].set_title(
        "Coset coordinate (where the memory lives)", fontsize=13, fontweight="bold"
    )
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        "Experiment D: SO(2) Goldstone memory — spectrum, retention, charge, latch",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved Goldstone summary plot to {save_path}")
def plot_v1_gate_calibration(
    R0_by_level: list,
    correct0_by_level: list,
    level_names: list,
    aurocs: list,
    save_path: str,
    n_bins: int = 5,
):
    """
    Reliability-diagram-style calibration of residual energy vs correctness
    (Experiment V1-Gate, Q1). One panel per difficulty level: retrieval
    accuracy within quantile bins of the residual R = H(settled) - floor,
    with the AUROC of R predicting incorrectness in the title.

    Args:
        R0_by_level: list of (T,) residual arrays, one per level
        correct0_by_level: list of (T,) boolean arrays
        level_names: list of level label strings
        aurocs: list of AUROC floats (may contain NaN)
        save_path: output path
        n_bins: number of quantile bins
    """
    n = len(R0_by_level)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), squeeze=False)
    for i, (R, c, name, auc) in enumerate(
        zip(R0_by_level, correct0_by_level, level_names, aurocs, strict=True)
    ):
        ax = axes[0, i]
        R = np.asarray(R)
        c = np.asarray(c).astype(float)
        edges = np.quantile(R, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9
        accs, centers, counts = [], [], []
        for b in range(n_bins):
            m = (R >= edges[b]) & (R < edges[b + 1])
            if m.sum() > 0:
                accs.append(c[m].mean())
                centers.append(R[m].mean())
                counts.append(int(m.sum()))
        ax.plot(centers, accs, "o-", color="tab:blue")
        for x, y, cnt in zip(centers, accs, counts, strict=True):
            ax.annotate(str(cnt), (x, y), fontsize=7, xytext=(2, 4),
                        textcoords="offset points")
        ax.set_xlabel("residual R (binned)", fontsize=10)
        if i == 0:
            ax.set_ylabel("retrieval accuracy", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        auc_str = "nan" if auc != auc else f"{auc:.3f}"
        ax.set_title(f"{name}\nAUROC(R→wrong)={auc_str}", fontsize=10)
    plt.suptitle("V1 gate Q1: residual-energy calibration", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved calibration plot to {save_path}")


def plot_v1_gate_compute_curves(
    curves: dict,
    hopfield_acc: float,
    save_path: str,
    title: str = "V1 gate: accuracy vs compute",
):
    """
    Compute-matched accuracy curves for the V1-Gate cascade arms.

    Args:
        curves: dict arm_name -> (mean_steps array, accuracy array)
        hopfield_acc: modern-Hopfield accuracy (horizontal reference; its
            cost is one matvec, incommensurable with Verlet steps)
        save_path: output path
        title: figure title
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {
        "mass": ("tab:blue", "o-"),
        "raw": ("tab:orange", "s--"),
        "kick": ("tab:green", "^--"),
        "margin": ("tab:purple", "v--"),
        "relax-longer": ("tab:gray", "d-"),
    }
    for name, (steps, acc) in curves.items():
        color, style = styles.get(name, ("black", "x-"))
        order = np.argsort(np.asarray(steps))
        ax.plot(np.asarray(steps)[order], np.asarray(acc)[order], style,
                color=color, label=name, alpha=0.85)
    ax.axhline(hopfield_acc, color="tab:red", linestyle=":",
               label=f"modern Hopfield ({hopfield_acc:.3f}; ~1 matvec)")
    ax.set_xlabel("mean Verlet steps per query", fontsize=11)
    ax.set_ylabel("retrieval accuracy", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved compute curves to {save_path}")


def plot_v1_gate_mass_scatter(scatter_pool: list, save_path: str):
    """
    Per-mode displacement vs inverse effective inertial mass under the first
    S^(M) boost retry (Thread-5 falsifiable (ii); F5 §5.4).

    Left: |Δq_i| of boost+re-relax vs 1/M_eff,i (log-log, pooled).
    Right: instantaneous check — the p-dependent part of the squeeze's Δq
    against its exact prediction sinh(ζ)·p_i/M_eff,i.

    Args:
        scatter_pool: list of dicts with keys m_eff, p_before, q_before,
            dq_instant, dq_total, zeta_chosen
        save_path: output path
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    inv_m, dq_tot, pred_inst, obs_inst = [], [], [], []
    for sc in scatter_pool:
        m = np.asarray(sc["m_eff"])
        T = sc["dq_total"].shape[0]
        inv_m.append(np.tile(1.0 / m, (T, 1)).ravel())
        dq_tot.append(np.abs(np.asarray(sc["dq_total"])).ravel())
        z = np.asarray(sc["zeta_chosen"])[:, None]
        q_b = np.asarray(sc["q_before"])
        p_b = np.asarray(sc["p_before"])
        pred = np.sinh(z) * p_b / m[None, :]
        obs = np.asarray(sc["dq_instant"]) - (np.cosh(z) - 1.0) * q_b
        pred_inst.append(np.abs(pred).ravel())
        obs_inst.append(np.abs(obs).ravel())
    inv_m = np.concatenate(inv_m)
    dq_tot = np.concatenate(dq_tot)
    pred_inst = np.concatenate(pred_inst)
    obs_inst = np.concatenate(obs_inst)

    axes[0].scatter(inv_m, dq_tot + 1e-12, s=4, alpha=0.3, color="tab:blue")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"$1/M_{\mathrm{eff},i}$", fontsize=11)
    axes[0].set_ylabel(r"$|\Delta q_i|$ (boost + re-relax)", fontsize=11)
    axes[0].set_title("per-mode displacement vs inverse inertial mass",
                      fontsize=10)
    axes[0].grid(True, alpha=0.3)

    lim = max(pred_inst.max(), obs_inst.max()) + 1e-12
    axes[1].scatter(pred_inst + 1e-15, obs_inst + 1e-15, s=4, alpha=0.3,
                    color="tab:green")
    axes[1].plot([1e-15, lim], [1e-15, lim], "k--", lw=1, label="y = x")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$|\sinh(\zeta)\, p_i / M_{\mathrm{eff},i}|$ (exact)",
                       fontsize=11)
    axes[1].set_ylabel(r"$|\Delta q_i^{\mathrm{boost}} - (\cosh\zeta - 1) q_i|$",
                       fontsize=11)
    axes[1].set_title("instantaneous S^(M) position response", fontsize=10)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("V1 gate: mass-weighted boost response", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved mass scatter to {save_path}")


def plot_lattice_pricing(
    kappas,
    sync_measured,
    sync_predicted,
    hl_measured,
    hl_predicted,
    mu_rel_sq_measured,
    mu_rel_sq_predicted,
    latch_freeze_drift,
    mu_sym_sq_abs,
    save_path: str,
    gamma: float = None,
    slopes: tuple = None,
):
    """
    The V3 acceptance centerpiece (F5 §7.2): coupling strength prices
    communication speed against relative-memory lifetime.

    Panels:
        1. Sync timescale of the relative angle vs kappa_c (log-log) with the
           quarter-period prediction pi/(2 mu_rel dt) — slope -1/2.
        2. Relative-information retention half-life vs kappa_c (log-log) with
           the exact overdamped prediction — slope -1.
        3. Quadratic-order law parity: measured mu_rel^2 vs 4 kappa_c / M.
        4. The shared (diagonal) channel stays an exact latch at every
           kappa_c: freeze drift and |mu_sym^2| (both ~ machine zero).

    Args:
        kappas: (K,) coupling strengths.
        sync_measured / sync_predicted: (K,) sync steps (first alignment).
        hl_measured / hl_predicted: (K,) relative-mode half-lives (steps).
        mu_rel_sq_measured / mu_rel_sq_predicted: (K,) relative spectral
            masses squared.
        latch_freeze_drift: (K,) shared-channel drift over the probe's last
            half (0 = frozen latch).
        mu_sym_sq_abs: (K,) |mu^2| of the shared channel.
        save_path: output PNG path.
        gamma: probe friction (annotation only).
        slopes: optional (sync_slope, hl_slope) fitted log-log slopes.
    """
    kappas = np.asarray(kappas, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Sync timescale (communication speed)
    ax = axes[0, 0]
    ax.loglog(kappas, np.asarray(sync_measured), "o-", label="measured (first alignment)")
    ax.loglog(
        kappas,
        np.asarray(sync_predicted),
        "k--",
        alpha=0.7,
        label=r"$\pi/(2\mu_{\rm rel}\,\varepsilon)$  ($\propto \kappa_c^{-1/2}$)",
    )
    title = "Sync timescale of the relative angle"
    if slopes is not None:
        title += f"  (fit slope {slopes[0]:.3f})"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(r"coupling strength $\kappa_c$", fontsize=12)
    ax.set_ylabel("steps to first alignment", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # 2. Relative-memory lifetime
    ax = axes[0, 1]
    ax.loglog(kappas, np.asarray(hl_measured), "s-", color="tab:red", label="measured $n_{1/2}$")
    ax.loglog(
        kappas,
        np.asarray(hl_predicted),
        "k--",
        alpha=0.7,
        label=r"exact overdamped prediction ($\propto 1/\kappa_c$)",
    )
    title = "Relative-information retention"
    if gamma is not None:
        title += rf"  ($\gamma$={gamma})"
    if slopes is not None:
        title += f"  (fit slope {slopes[1]:.3f})"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(r"coupling strength $\kappa_c$", fontsize=12)
    ax.set_ylabel(r"half-life $n_{1/2}$ (steps)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # 3. Quadratic-order law parity
    ax = axes[1, 0]
    pred = np.asarray(mu_rel_sq_predicted)
    meas = np.asarray(mu_rel_sq_measured)
    ax.loglog(pred, meas, "o", color="tab:green")
    lim_lo, lim_hi = pred.min() * 0.5, pred.max() * 2.0
    ax.loglog([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, label="y = x")
    ax.set_title(
        r"Communication has a mass: $\mu_{\rm rel}^2 = 4\kappa_c/M$",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel(r"predicted $4\kappa_c/M$", fontsize=12)
    ax.set_ylabel(r"measured $\mu_{\rm rel}^2$", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # 4. Shared channel = exact latch at every kappa
    ax = axes[1, 1]
    ax.semilogy(
        kappas,
        np.clip(np.asarray(latch_freeze_drift), 1e-20, None),
        "^-",
        color="tab:purple",
        label="latch freeze drift (last half)",
    )
    ax.semilogy(
        kappas,
        np.clip(np.asarray(mu_sym_sq_abs), 1e-20, None),
        "v-",
        color="tab:gray",
        label=r"$|\mu_{\rm sym}^2|$ (shared channel)",
    )
    ax.set_xscale("log")
    ax.set_title(
        "Shared channel stays an exact latch", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel(r"coupling strength $\kappa_c$", fontsize=12)
    ax.set_ylabel("drift / spectral mass (log)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    plt.suptitle(
        "CLU lattice: coupling strength prices communication speed "
        "against relative-memory lifetime (F5 §7.2)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved lattice pricing plot to {save_path}")
_V1_CALIB_STYLES = {
    "clu_calib_rm": ("tab:blue", "-", "CLU calibrated (R+margin)"),
    "clu_calib_r": ("tab:cyan", "-", "CLU calibrated (R only)"),
    "clu_calib_margin": ("tab:purple", "-", "CLU calibrated (margin)"),
    "clu_margin_raw": ("tab:purple", "--", "CLU margin (naive)"),
    "clu_R_raw": ("tab:gray", "--", "CLU raw -R (naive)"),
    "hop_msp": ("tab:red", "--", "Hopfield max-softmax (naive)"),
    "hop_logit_margin": ("tab:orange", "--", "Hopfield logit margin (naive)"),
    "hop_calib": ("tab:red", "-", "Hopfield calibrated"),
    "calib_deployed": ("tab:blue", "-", "learned gate (swept)"),
    "margin_raw": ("tab:purple", "--", "margin-gated (swept)"),
    "R_raw": ("tab:gray", "--", "raw-R-gated (swept)"),
}


def plot_v1_calib_risk_coverage(panels: list, save_path: str,
                                risk_line: float = 0.05):
    """
    Risk-coverage curves of the abstention head-to-head (exp_v1_calibration).
    One panel per difficulty level (+ pooled); mean curve across seeds with a
    ±1 std band.

    Args:
        panels: list of dicts {"title": str, "methods":
            {name: (coverage_grid, mean_risk, std_risk)}}
        save_path: output path
        risk_line: horizontal reference (e.g. 0.05 = 95% precision)
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.8), squeeze=False)
    for i, panel in enumerate(panels):
        ax = axes[0, i]
        for name, (grid, mean, std) in panel["methods"].items():
            color, ls, label = _V1_CALIB_STYLES.get(name, ("black", "-", name))
            ax.plot(grid, mean, ls, color=color, label=label, alpha=0.9, lw=1.6)
            if std is not None and np.any(std > 0):
                ax.fill_between(grid, mean - std, mean + std, color=color,
                                alpha=0.12, lw=0)
        ax.axhline(risk_line, color="k", linestyle=":", lw=1,
                   label=f"risk = {risk_line}")
        ax.set_xlabel("coverage", fontsize=10)
        if i == 0:
            ax.set_ylabel("selective risk (error rate)", fontsize=10)
        ax.set_title(panel["title"], fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=-0.02)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="upper left")
    plt.suptitle("V1 calibration: abstention risk-coverage "
                 "(base stage; mean ± std over seeds)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved risk-coverage plot to {save_path}")


def plot_v1_calib_compute(panels: list, save_path: str):
    """
    Compute-allocation curves of the gated escalation ladder
    (exp_v1_calibration). One panel per difficulty level: swept-threshold
    accuracy-vs-cost curves (mean ± std band over seeds), the learned
    operating point with error bars, always-small/always-full markers and
    the Hopfield reference line.

    Args:
        panels: list of dicts {"title", "curves": {name: (cost_grid,
            mean_acc, std_acc)}, "points": {label: (cost_mean, cost_std,
            acc_mean, acc_std)}, "hopfield_acc": float}
        save_path: output path
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for i, panel in enumerate(panels):
        ax = axes[0, i]
        for name, (cost, mean, std) in panel["curves"].items():
            color, ls, label = _V1_CALIB_STYLES.get(name, ("black", "-", name))
            ax.plot(cost, mean, ls, color=color, label=label, alpha=0.9, lw=1.6)
            if std is not None and np.any(std > 0):
                ax.fill_between(cost, mean - std, mean + std, color=color,
                                alpha=0.12, lw=0)
        markers = {"learned gate (p_exit)": ("tab:blue", "o"),
                   "always-small": ("tab:green", "s"),
                   "always-full": ("black", "D")}
        for label, (cm, cs, am, as_) in panel["points"].items():
            color, mk = markers.get(label, ("black", "x"))
            ax.errorbar([cm], [am], xerr=[cs], yerr=[as_], fmt=mk,
                        color=color, capsize=3, ms=7, label=label, zorder=5)
        ax.axhline(panel["hopfield_acc"], color="tab:red", linestyle=":",
                   lw=1.2,
                   label=f"Hopfield ({panel['hopfield_acc']:.3f}; ~1 matvec)")
        ax.set_xlabel("mean Verlet steps per query", fontsize=10)
        if i == 0:
            ax.set_ylabel("retrieval accuracy", fontsize=10)
        ax.set_title(panel["title"], fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="best")
    plt.suptitle("V1 calibration: gated compute allocation "
                 "(mean ± std over seeds)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved compute-allocation plot to {save_path}")


def plot_v1_calib_reliability(rel: dict, save_path: str):
    """
    Reliability diagrams of the probability-valued confidence signals
    (exp_v1_calibration): predicted P(correct) vs empirical accuracy in
    quantile bins, pooled over all cells; ECE (mean ± std across seeds)
    annotated per method.

    Args:
        rel: {method: {"bins": (conf_bin_means, acc_bin_means),
              "ece": (mean, std)}}
        save_path: output path
    """
    n = len(rel)
    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 3.6), squeeze=False)
    for i, (name, d) in enumerate(rel.items()):
        ax = axes[0, i]
        conf, acc = d["bins"]
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
        ax.plot(conf, acc, "o-", color=_V1_CALIB_STYLES.get(
            name, ("tab:blue", "-", name))[0])
        em, es = d["ece"]
        _, _, label = _V1_CALIB_STYLES.get(name, (None, None, name))
        ax.set_title(f"{label}\nECE = {em:.3f} ± {es:.3f}", fontsize=10)
        ax.set_xlabel("predicted P(correct)", fontsize=10)
        if i == 0:
            ax.set_ylabel("empirical accuracy", fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
    plt.suptitle("V1 calibration: reliability of confidence signals",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved reliability plot to {save_path}")
def plot_gamma_field_landscape(
    chlu_model,
    save_path: str,
    trajectory: jnp.ndarray = None,
    noise_center: np.ndarray = None,
    grid_resolution: int = 120,
    margin: float = 0.8,
    title: str = None,
):
    """
    Two-panel view of a CHLU with a friction field (trash regions, S1 pilot):
    learned potential V(q) contours (left) and the friction field gamma_phi(q)
    heatmap with hole horizons (right), with the data trajectory overlaid.

    Args:
        chlu_model: CHLU carrying a ``friction_field`` (2D position space).
        save_path: Path to save figure.
        trajectory: Optional (T, >=2) array; columns 0:2 = data positions.
        noise_center: Optional (2,) known noise locus to mark (oracle target).
        grid_resolution: Grid points per axis.
        margin: Padding around the union of trajectory/holes/noise locus.
        title: Optional figure title.
    """
    field = chlu_model.friction_field
    centers, radii, strengths = field.hole_params()
    centers = np.asarray(centers)

    # Extent = union of trajectory, holes (center +/- radius), noise locus
    pts = [centers - np.asarray(radii)[:, None], centers + np.asarray(radii)[:, None]]
    if trajectory is not None:
        pts.append(np.asarray(trajectory[:, :2]))
    if noise_center is not None:
        pts.append(np.asarray(noise_center)[None, :])
    allpts = np.concatenate(pts, axis=0)
    x_min, y_min = allpts.min(axis=0) - margin
    x_max, y_max = allpts.max(axis=0) + margin

    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x, y)
    grid = jnp.asarray(np.stack([X.ravel(), Y.ravel()], axis=1))
    V = np.asarray(jax.vmap(chlu_model.potential_net)(grid)).reshape(X.shape)
    G = np.asarray(jax.vmap(field)(grid)).reshape(X.shape)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    cf0 = axes[0].contourf(X, Y, V, levels=25, cmap="viridis", alpha=0.85)
    plt.colorbar(cf0, ax=axes[0], label="V(q)")
    axes[0].set_title("Learned potential V(q)", fontsize=13, fontweight="bold")

    cf1 = axes[1].contourf(X, Y, G, levels=25, cmap="inferno", alpha=0.9)
    plt.colorbar(cf1, ax=axes[1], label=r"$\gamma_\varphi(q)$")
    for k in range(centers.shape[0]):
        circle = plt.Circle(
            centers[k], float(radii[k]), fill=False, color="cyan",
            linewidth=1.5, linestyle="--",
        )
        axes[1].add_patch(circle)
        axes[1].annotate(
            f"$\\gamma_{{{k}}}$={float(strengths[k]):.2f}",
            centers[k], color="cyan", fontsize=9, fontweight="bold",
        )
    axes[1].set_title(
        r"Friction field $\gamma_\varphi(q)$ (horizons dashed)",
        fontsize=13, fontweight="bold",
    )

    for ax in axes:
        if trajectory is not None:
            ax.plot(
                np.asarray(trajectory[:, 0]), np.asarray(trajectory[:, 1]),
                "w-", linewidth=1.2, alpha=0.8, label="data attractor",
            )
        if noise_center is not None:
            ax.scatter(
                [noise_center[0]], [noise_center[1]], marker="*", s=220,
                c="red", edgecolors="white", zorder=10, label="noise locus",
            )
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_aspect("equal")
        ax.legend(fontsize=9, loc="upper left")

    if title:
        plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved gamma-field landscape to {save_path}")


def plot_s1_pareto(
    arm_records: list,
    save_path: str,
    retention_key: str = "coverage",
    rejection_key: str = "rejection_pos",
):
    """
    S1 Pareto plot: signal retention (y) vs noise rejection (x).

    The global-gamma arm is drawn as a trade-off curve (seed-averaged per
    gamma, annotated); all other arms as per-seed scatter + seed-mean marker.

    Args:
        arm_records: list of dicts with keys "arm", "label", "seed",
            retention_key, rejection_key (and "gamma" for the sweep arm).
        save_path: output PNG path.
        retention_key / rejection_key: which metric pair to plot.
    """
    fig, ax = plt.subplots(figsize=(8.5, 7))

    # (i) global gamma: curve over the sweep, averaged across seeds
    sweep = [r for r in arm_records if r["arm"] == "global_gamma"]
    gammas = sorted({r["gamma"] for r in sweep})
    xs, ys = [], []
    for g in gammas:
        pts = [r for r in sweep if r["gamma"] == g]
        xs.append(np.nanmean([p[rejection_key] for p in pts]))
        ys.append(np.nanmean([p[retention_key] for p in pts]))
    if xs:
        ax.plot(xs, ys, "o-", color="gray", linewidth=1.5, label="(i) global $\\gamma$ sweep")
        for g, x_, y_ in zip(gammas, xs, ys, strict=True):
            ax.annotate(f"{g:g}", (x_, y_), fontsize=8, color="gray",
                        xytext=(4, 4), textcoords="offset points")

    styles = {
        "governor": dict(color="tab:blue", marker="s", label="(ii) governor"),
        "oracle": dict(color="tab:green", marker="D", label="(iv) oracle hole"),
    }
    other_arms = sorted(
        {r["arm"] for r in arm_records if r["arm"] != "global_gamma"}
    )
    palette = plt.cm.autumn(np.linspace(0.0, 0.6, max(1, len(other_arms))))
    for i, arm in enumerate(other_arms):
        pts = [r for r in arm_records if r["arm"] == arm]
        style = styles.get(
            arm,
            dict(color=palette[i], marker="*",
                 label=f"(iii) {pts[0]['label']}" if arm.startswith("learned") else arm),
        )
        x_ = [p[rejection_key] for p in pts]
        y_ = [p[retention_key] for p in pts]
        ax.scatter(x_, y_, s=60, alpha=0.45, color=style["color"], marker=style["marker"])
        ax.scatter(
            [np.nanmean(x_)], [np.nanmean(y_)], s=220, color=style["color"],
            marker=style["marker"], edgecolors="black", linewidths=1.2,
            label=style["label"] + " (mean)", zorder=10,
        )

    ax.set_xlabel(f"noise rejection ({rejection_key})", fontsize=12)
    ax.set_ylabel(f"signal retention ({retention_key})", fontsize=12)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower left")
    ax.set_title(
        "S1: signal-retention vs noise-rejection Pareto\n"
        "(up-right dominates; prediction: learned $\\gamma_\\varphi$ "
        "dominates global $\\gamma$)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved S1 Pareto plot to {save_path}")


def plot_v1_regime_map(regime: dict, save_path: str):
    """CLU-gate-vs-Hopfield regime map (exp_v1 hopfield-stress).

    Four panels over the capacity (rows) x stress (cols) grid:
      1. classification map (Hopfield-dominant / comparable / CLU-gate-advantage),
         annotated with the accuracy delta (clu_gate - hopfield);
      2. accuracy delta heatmap (diverging; >0 = CLU better);
      3. abstention delta heatmap (hop_aurc - clu_aurc; >0 = CLU better);
      4. accuracy vs stress line plot (CLU gate solid, Hopfield dashed) per
         capacity, with compute-savings annotated.

    Args:
        regime: dict from run_v1_hopfield_regime_map (arrays shaped
            (n_cap, n_stress)).
        save_path: output path.
    """
    caps = regime["cap_labels"]
    grid = regime["stress_grid"]
    axis = regime["axis"]
    cat = np.asarray(regime["cat"])
    d_acc = np.asarray(regime["d_acc"])
    d_aurc = np.asarray(regime["d_aurc"])
    clu_acc = np.asarray(regime["clu_acc"])
    hop_acc = np.asarray(regime["hop_acc"])
    savings = np.asarray(regime["savings"])
    n_cap, n_str = cat.shape

    from matplotlib.colors import ListedColormap

    cat_colors = ["#c44e52", "#dddddd", "#55a868"]  # dom / comparable / clu-adv
    cat_labels = ["Hopfield-dominant", "comparable", "CLU-gate advantage"]
    cmap_cat = ListedColormap(cat_colors)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    xt = [f"{g:g}" for g in grid]
    yt = caps

    # panel 1: classification map
    ax = axes[0, 0]
    ax.imshow(cat, cmap=cmap_cat, vmin=0, vmax=2, aspect="auto")
    for i in range(n_cap):
        for j in range(n_str):
            ax.text(j, i, f"{d_acc[i, j]:+.2f}", ha="center", va="center",
                    fontsize=9, color="black")
    ax.set_xticks(range(n_str))
    ax.set_xticklabels(xt)
    ax.set_yticks(range(n_cap))
    ax.set_yticklabels(yt)
    ax.set_xlabel(f"stress: {axis}")
    ax.set_ylabel("capacity (N/kv)")
    ax.set_title("regime map (cell text = acc(CLU gate) - acc(Hopfield))")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cat_colors]
    ax.legend(handles, cat_labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)

    # panel 2: accuracy delta
    ax = axes[0, 1]
    vmax = float(np.nanmax(np.abs(d_acc))) or 1.0
    im = ax.imshow(d_acc, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(n_cap):
        for j in range(n_str):
            ax.text(j, i, f"{d_acc[i, j]:+.2f}", ha="center", va="center",
                    fontsize=8)
    ax.set_xticks(range(n_str))
    ax.set_xticklabels(xt)
    ax.set_yticks(range(n_cap))
    ax.set_yticklabels(yt)
    ax.set_xlabel(f"stress: {axis}")
    ax.set_title("acc(CLU gate) - acc(Hopfield)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # panel 3: abstention delta (AURC)
    ax = axes[1, 0]
    vmax = float(np.nanmax(np.abs(d_aurc))) or 1.0
    im = ax.imshow(d_aurc, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(n_cap):
        for j in range(n_str):
            ax.text(j, i, f"{d_aurc[i, j]:+.2f}", ha="center", va="center",
                    fontsize=8)
    ax.set_xticks(range(n_str))
    ax.set_xticklabels(xt)
    ax.set_yticks(range(n_cap))
    ax.set_yticklabels(yt)
    ax.set_xlabel(f"stress: {axis}")
    ax.set_ylabel("capacity (N/kv)")
    ax.set_title("AURC(Hopfield) - AURC(CLU)  (>0 = CLU abstains better)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # panel 4: accuracy-vs-stress lines + savings
    ax = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0, 0.85, n_cap))
    for i in range(n_cap):
        ax.plot(grid, clu_acc[i], "o-", color=colors[i], label=f"{caps[i]} CLU gate")
        ax.plot(grid, hop_acc[i], "s--", color=colors[i], alpha=0.6,
                label=f"{caps[i]} Hopfield")
        for j in range(n_str):
            if np.isfinite(savings[i, j]):
                ax.annotate(f"{savings[i, j]:.1f}x", (grid[j], clu_acc[i, j]),
                            fontsize=7, color=colors[i],
                            textcoords="offset points", xytext=(0, 6))
    ax.set_xlabel(f"stress: {axis}")
    ax.set_ylabel("accuracy")
    ax.set_title("accuracy vs stress (annot = CLU compute savings)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower left")

    plt.suptitle(
        f"V1: CLU-gate vs Hopfield regime map (stress axis: {axis})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved regime map to {save_path}")
def plot_v1_wormhole_cost_accuracy(summary: dict, arms: list, save_path: str):
    """Cost-vs-accuracy for the V1 wormhole-routing arms (the headline plot).

    One panel per lattice size N. x-axis = mean cost (unit-steps; a FLOP proxy =
    Verlet steps x active units), y-axis = retrieval accuracy. Each arm is a
    point (overall accuracy); the distant-only accuracy is shown as a lighter
    marker at the same cost (the routing story lives in the distant split).

    Args:
        summary: run summary dict (summary["by_N"][str(N)]["arms"][arm]).
        arms: ordered arm names.
        save_path: output path.
    """
    Ns = sorted(int(k) for k in summary["by_N"].keys())
    n = len(Ns)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)
    colors = {
        "local_only": "tab:gray",
        "gated": "tab:blue",
        "dense": "tab:red",
        "chain": "tab:green",
        "calibrated": "tab:purple",
        "router_mlp": "tab:orange",
    }
    for ci, N in enumerate(Ns):
        ax = axes[0, ci]
        ent = summary["by_N"][str(N)]["arms"]
        for a in arms:
            c = colors.get(a, "black")
            cost = ent[a]["cost_mean"]
            ax.errorbar(cost, ent[a]["acc_mean"], yerr=ent[a]["acc_std"],
                        fmt="o", color=c, ms=10, capsize=3, label=a, zorder=3)
            ax.scatter(cost, ent[a]["acc_distant_mean"], color=c, marker="x",
                       s=60, alpha=0.7, zorder=2)
            ax.annotate(a, (cost, ent[a]["acc_mean"]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("mean cost (unit-steps = Verlet steps x active units)",
                      fontsize=10)
        if ci == 0:
            ax.set_ylabel("retrieval accuracy\n(o = overall, x = distant-only)",
                          fontsize=10)
        auc = summary["by_N"][str(N)]["auroc_R0_distant_mean"]
        ax.set_title(f"N={N}  (AUROC R0->distant = {auc:.3f})", fontsize=11)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="lower right")
    plt.suptitle("V1 wormhole: cost vs accuracy — gated routing near dense "
                 "accuracy at a fraction of dense cost", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved wormhole cost-accuracy plot to {save_path}")


def plot_v1_wormhole_selectivity(runs: dict, summary: dict, n_units_values: list,
                                 save_path: str):
    """Gate-selectivity diagnostics for the V1 wormhole routing.

    Three panels (pooled over seeds of the FIRST N): (1) gate-open/closed vs
    answer distant/local confusion matrix; (2) residual R0 distribution split
    by local vs distant queries (the routing signal); (3) smooth gate g vs the
    z-normalized residual with the query type coloured (does the gate open
    selectively for distant-answer queries?).

    Args:
        runs: runs[N] = list of per-seed record dicts.
        summary: run summary (for the confusion matrix + AUROC).
        n_units_values: list of N (first is plotted).
        save_path: output path.
    """
    N = int(n_units_values[0])
    recs = runs[N]
    R0 = np.concatenate([r["R0"] for r in recs])
    z = np.concatenate([r["z"] for r in recs])
    g = np.concatenate([r["g_smooth"] for r in recs])
    dist = np.concatenate([r["is_distant"] for r in recs])
    cm = np.asarray(summary["by_N"][str(N)]["gate_confusion"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["distant", "local"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["gate open", "gate closed"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(int(v)), ha="center", va="center",
                color="black", fontsize=13)
    prec = summary["by_N"][str(N)]["gate_precision_distant"]
    rec = summary["by_N"][str(N)]["gate_recall_distant"]
    ax.set_title(f"gate selectivity (g>0.5)\nprecision={prec:.2f} recall={rec:.2f}",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    bins = np.linspace(min(R0.min(), 0), R0.max(), 30)
    ax.hist(R0[~dist], bins=bins, alpha=0.6, color="tab:gray", label="local")
    ax.hist(R0[dist], bins=bins, alpha=0.6, color="tab:blue", label="distant")
    ax.set_xlabel("local residual R0 = H0(settled) - floor0", fontsize=10)
    ax.set_ylabel("count", fontsize=10)
    auc = summary["by_N"][str(N)]["auroc_R0_distant_mean"]
    ax.set_title(f"routing signal (AUROC R0->distant = {auc:.3f})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.scatter(z[~dist], g[~dist], s=14, alpha=0.5, color="tab:gray",
               label="local")
    ax.scatter(z[dist], g[dist], s=14, alpha=0.5, color="tab:blue",
               label="distant")
    ax.axhline(0.5, color="k", ls=":", lw=1)
    ax.set_xlabel("z-normalized residual", fontsize=10)
    ax.set_ylabel("smooth gate g", fontsize=10)
    ax.set_title("gate opening vs residual", fontsize=10)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"V1 wormhole selectivity (N={N})", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved wormhole selectivity plot to {save_path}")


def plot_v1_wormhole_flops(summary: dict, arms: list, save_path: str):
    """FLOPs-vs-accuracy by workload mix (P9/V1.2): the energy-gated wormhole vs
    the parameter-matched learned router, priced in FLOPs (not unit-steps).

    Grid of panels [N x workload-mix]; x-axis = mean FLOPs/query (log), y-axis =
    accuracy. Each arm is a point with an accuracy error bar over seeds. The
    router-MLP (physics-free) is the baseline the energy gate must beat.
    """
    Ns = sorted(int(k) for k in summary["by_N"].keys())
    mix_labels = list(summary["by_N"][str(Ns[0])]["mixes"].keys())
    nrow, ncol = len(Ns), len(mix_labels)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.8 * nrow),
                             squeeze=False)
    colors = {
        "local_only": "tab:gray", "gated": "tab:blue", "dense": "tab:red",
        "chain": "tab:green", "calibrated": "tab:purple",
        "router_mlp": "tab:orange",
    }
    for ri, N in enumerate(Ns):
        for ciX, label in enumerate(mix_labels):
            ax = axes[ri][ciX]
            marms = summary["by_N"][str(N)]["mixes"][label]["arms"]
            for a in arms:
                c = colors.get(a, "black")
                fl = marms[a]["flops_mean"]
                ax.errorbar(fl, marms[a]["acc_mean"], yerr=marms[a]["acc_std"],
                            fmt="o", color=c, ms=9, capsize=3, label=a, zorder=3)
                ax.annotate(a, (fl, marms[a]["acc_mean"]), fontsize=7,
                            xytext=(4, 3), textcoords="offset points")
            ax.set_xscale("log")
            ax.set_title(f"N={N}  workload {label} (local/distant)", fontsize=10)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, alpha=0.3, which="both")
            if ciX == 0:
                ax.set_ylabel("accuracy", fontsize=10)
            if ri == nrow - 1:
                ax.set_xlabel("mean FLOPs / query (log)", fontsize=10)
    axes[0][0].legend(fontsize=7, loc="lower right")
    plt.suptitle("V1 wormhole: FLOPs vs accuracy — energy gate vs parameter-"
                 "matched learned router, by workload mix", fontsize=12,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved wormhole FLOPs-accuracy plot to {save_path}")
