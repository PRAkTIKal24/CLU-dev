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

    # data shapes3d
    shapes3d_parser = data_subparsers.add_parser(
        'shapes3d',
        help='Fetch Shapes3D + build the Montero recomb-to-element split index'
    )
    shapes3d_parser.add_argument('--data-root', type=Path, help='Dataset directory')
    shapes3d_parser.add_argument('--download', action='store_true', help='Fetch if missing')
    shapes3d_parser.add_argument('--quick', action='store_true', help='Subsample train rows')
    shapes3d_parser.add_argument('--no-verify', action='store_true', help='Skip checksum check')
    shapes3d_parser.set_defaults(func=cmd_data_shapes3d)

    # data dsprites
    dsprites_parser = data_subparsers.add_parser(
        'dsprites',
        help='Fetch dSprites + build the Montero recomb-to-element split index'
    )
    dsprites_parser.add_argument('--data-root', type=Path, help='Dataset directory')
    dsprites_parser.add_argument('--download', action='store_true', help='Fetch if missing')
    dsprites_parser.add_argument('--quick', action='store_true', help='Subsample train rows')
    dsprites_parser.add_argument('--no-verify', action='store_true', help='Skip checksum check')
    dsprites_parser.set_defaults(func=cmd_data_dsprites)


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


# ---------------------------------------------------------------------------
# Generative-benchmark datasets (Shapes3D / dSprites) + the Montero splits.
# Both commands: fetch -> verify checksum -> build the exhaustive factor-grid
# index -> assert the held-out set is non-empty and of the realised size ->
# assert no train row satisfies the held-out predicate -> emit c_invalid.json.
# ---------------------------------------------------------------------------


def _report_gen_split(split, n_checked, extra=None):
    """Shared console report for the two generative-dataset commands."""
    console.print(f"  Factors: {list(split.grid.names)}")
    console.print(f"  Cardinalities: {list(split.grid.cardinalities)}")
    console.print(
        f"  n_total={split.n_total}  n_train={split.n_train}  n_test={split.n_test}"
    )
    console.print(
        f"✓ hold-out assertion passed over {n_checked} train combinations",
        style="green",
    )
    for line in extra or []:
        console.print(f"  {line}")


def _build_gen_artifacts(mod, split, cfg, out_dir):
    """Persist the factor-grid index and the C set; return their checksums."""
    from ..data.gen_splits import (
        build_c_invalid,
        save_index,
        validate_c_invalid,
        write_json_with_checksum,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path, index_sha = save_index(split, out_dir / 'factor_index.npz')
    n_per_kind = cfg.c_invalid_n_per_kind
    if n_per_kind is None:
        # |C| ~= |B|: two kinds, so ceil(n_test / 2) of each.
        n_per_kind = max(1, -(-split.n_test // 2))
    c_set = build_c_invalid(
        split.grid,
        n_per_kind=n_per_kind,
        seed=cfg.c_invalid_seed,
    )
    n_c = validate_c_invalid(c_set, split.grid)
    c_path, c_sha = write_json_with_checksum(c_set, out_dir / 'c_invalid.json')
    return [
        f"index:     {index_path}  sha256={index_sha}",
        f"c_invalid: {c_path}  sha256={c_sha}  ({n_c} records)",
    ]


def cmd_data_shapes3d(args):
    """Fetch Shapes3D and build the Montero recombination-to-element split."""
    from ..config import get_default_config
    from ..data import shapes3d as s3d

    cfg = get_default_config().gen_data
    root = args.data_root if args.data_root else (
        Path(cfg.data_root) if cfg.data_root else s3d.shapes3d_root()
    )
    verify = cfg.verify_checksum and not args.no_verify

    console.print("[bold cyan]Preparing Shapes3D (Montero recomb-to-element)[/bold cyan]")
    console.print(f"  Root: {root}")
    try:
        if args.download:
            prov = s3d.download_shapes3d(root, verify=verify)
            for name, rec in prov.items():
                console.print(f"  {name}: {rec['bytes']} bytes sha256={rec['sha256']}")
        result = s3d.load_shapes3d(
            root,
            quick=args.quick,
            quick_rows=cfg.quick_rows,
            seed=get_default_config().project.seed,
            holdout_assert=cfg.holdout_assert,
        )
        split = result['split']
        extra = _build_gen_artifacts(s3d, split, cfg, Path(root))
        _report_gen_split(split, result['n_checked'], extra)
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_data_dsprites(args):
    """Fetch dSprites and build the Montero recombination-to-element split."""
    from ..config import get_default_config
    from ..data import dsprites as dsp

    cfg = get_default_config().gen_data
    root = args.data_root if args.data_root else (
        Path(cfg.data_root) if cfg.data_root else dsp.dsprites_root()
    )
    verify = cfg.verify_checksum and not args.no_verify

    console.print("[bold cyan]Preparing dSprites (Montero recomb-to-element)[/bold cyan]")
    console.print(f"  Root: {root}")
    try:
        if args.download:
            prov = dsp.download_dsprites(
                root, with_images=cfg.dsprites_with_images, verify=verify
            )
            for name, rec in prov.items():
                console.print(f"  {name}: {rec['bytes']} bytes sha256={rec['sha256']}")
        result = dsp.load_dsprites(
            root,
            quick=args.quick,
            quick_rows=cfg.quick_rows,
            seed=get_default_config().project.seed,
            holdout_assert=cfg.holdout_assert,
        )
        split = result['split']
        extra = _build_gen_artifacts(dsp, split, cfg, Path(root))
        _report_gen_split(split, result['n_checked'], extra)
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0
