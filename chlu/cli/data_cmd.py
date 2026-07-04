"""Data generation CLI commands."""

import math
from pathlib import Path
from rich.console import Console
import numpy as np

console = Console()


def setup_data_parser(subparsers):
    """Set up the 'data' subcommand parser."""
    data_parser = subparsers.add_parser(
        'data',
        help='Generate or prepare datasets'
    )
    
    data_subparsers = data_parser.add_subparsers(
        dest='dataset',
        help='Dataset type'
    )
    
    # data figure8
    fig8_parser = data_subparsers.add_parser(
        'figure8',
        help='Generate figure-8 trajectory data'
    )
    fig8_parser.add_argument('--steps', type=int, default=1000, help='Number of time steps')
    fig8_parser.add_argument('--dt', type=float, help='Time step size')
    fig8_parser.add_argument('--output', type=Path, help='Output file path (.npz)')
    fig8_parser.set_defaults(func=cmd_data_figure8)
    
    # data sine
    sine_parser = data_subparsers.add_parser(
        'sine',
        help='Generate sine wave data'
    )
    sine_parser.add_argument('--n-waves', type=int, default=100, help='Number of waves')
    sine_parser.add_argument('--steps', type=int, default=200, help='Steps per wave')
    sine_parser.add_argument('--dt', type=float, help='Time step size')
    sine_parser.add_argument('--output', type=Path, help='Output file path (.npz)')
    sine_parser.set_defaults(func=cmd_data_sine)
    
    # data mnist
    mnist_parser = data_subparsers.add_parser(
        'mnist',
        help='Download and prepare MNIST with PCA'
    )
    mnist_parser.add_argument('--pca-dim', type=int, default=32, help='PCA dimensions')
    mnist_parser.add_argument('--output-dir', type=Path, help='Output directory')
    mnist_parser.set_defaults(func=cmd_data_mnist)


def cmd_data_figure8(args):
    """Generate figure-8 data."""
    import jax

    from ..data.figure8 import generate_figure8
    from ..config import get_default_config

    config = get_default_config()
    dt = args.dt if args.dt else config.data.figure8_dt
    scale = config.data.figure8_scale
    seed = config.project.seed

    console.print("[bold cyan]Generating figure-8 data[/bold cyan]")
    console.print(f"  Steps: {args.steps}, dt: {dt}")

    try:
        # generate_figure8 works in whole cycles (period 2*pi); generate enough
        # cycles to cover the requested number of steps, then truncate.
        steps_per_cycle = int(2 * math.pi / dt)
        n_cycles = max(1, math.ceil(args.steps / max(steps_per_cycle, 1)))
        key = jax.random.PRNGKey(seed)

        trajectory = generate_figure8(key, n_cycles=n_cycles, dt=dt, scale=scale)
        trajectory = np.asarray(trajectory[: args.steps])  # (steps, 4) [x, y, vx, vy]

        # Split into generalized position q=[x, y] and momentum p=[vx, vy]
        q = trajectory[:, :2]
        p = trajectory[:, 2:]

        if args.output:
            output_path = args.output
        else:
            output_path = Path('figure8_data.npz')

        np.savez(output_path, q=q, p=p, trajectory=trajectory, dt=dt)
        console.print(f"✓ Saved to {output_path} (q={q.shape}, p={p.shape})", style="green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_data_sine(args):
    """Generate sine wave data."""
    import jax

    from ..data.sine_waves import generate_sine_waves
    from ..config import get_default_config

    config = get_default_config()
    dt = args.dt if args.dt else config.data.sine_dt
    seed = config.project.seed

    console.print("[bold cyan]Generating sine wave data[/bold cyan]")
    console.print(f"  Waves: {args.n_waves}, Steps: {args.steps}, dt: {dt}")

    try:
        key = jax.random.PRNGKey(seed)
        waves = generate_sine_waves(
            key,
            n_waves=args.n_waves,
            steps=args.steps,
            dt=dt,
        )
        waves = np.asarray(waves)  # (n_waves, steps, 2) [x, dx/dt]

        # Split state into generalized position q=[x] and momentum p=[dx/dt]
        q_data = waves[..., 0:1]
        p_data = waves[..., 1:2]

        if args.output:
            output_path = args.output
        else:
            output_path = Path('sine_data.npz')

        np.savez(output_path, q=q_data, p=p_data, waves=waves, dt=dt)
        console.print(
            f"✓ Saved to {output_path} (q={q_data.shape}, p={p_data.shape})", style="green"
        )
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_data_mnist(args):
    """Prepare MNIST data."""
    from ..data.mnist import load_mnist_pca
    from ..config import get_default_config
    
    config = get_default_config()
    pca_dim = args.pca_dim if args.pca_dim else config.data.mnist_pca_dim
    
    console.print("[bold cyan]Preparing MNIST data[/bold cyan]")
    console.print(f"  PCA dimensions: {pca_dim}")
    
    try:
        train_imgs, test_imgs, pca = load_mnist_pca(dim=pca_dim)
        
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = Path('.')
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            output_dir / 'mnist_pca.npz',
            train=train_imgs,
            test=test_imgs,
            pca_dim=pca_dim
        )
        console.print(f"✓ Saved to {output_dir / 'mnist_pca.npz'}", style="green")
        console.print(f"  Train samples: {len(train_imgs)}, Test samples: {len(test_imgs)}")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    
    return 0
