"""Experiment execution CLI commands."""

from pathlib import Path
from rich.console import Console
from ..project import ProjectManager
from ..experiments.exp_a_stability import run_experiment_a
from ..experiments.exp_b_noise import run_experiment_b
from ..experiments.exp_c_dreaming import run_experiment_c
from ..experiments.exp_d_goldstone import run_experiment_d
from ..experiments.exp_v1_calibration import (
    run_experiment_v1_calibration,
    run_v1_hopfield_regime_map,
)
from ..experiments.exp_v1_gate import run_experiment_v1_gate
from ..experiments.exp_lattice import run_experiment_lattice
from ..experiments.exp_s1_gamma_field import run_experiment_s1

console = Console()


def setup_experiment_parsers(subparsers):
    """Set up experiment subcommand parsers."""
    
    # exp-a
    exp_a_parser = subparsers.add_parser(
        'exp-a',
        help='Run Experiment A: Stability (100x extrapolation)'
    )
    exp_a_parser.add_argument('--project', help='Project name to use')
    exp_a_parser.add_argument('--seed', type=int, help='Random seed')
    exp_a_parser.add_argument('--quick', action='store_true', help='Quick mode (50 epochs)')
    exp_a_parser.set_defaults(func=cmd_exp_a)
    
    # exp-b
    exp_b_parser = subparsers.add_parser(
        'exp-b',
        help='Run Experiment B: Noise Rejection'
    )
    exp_b_parser.add_argument('--project', help='Project name to use')
    exp_b_parser.add_argument('--seed', type=int, help='Random seed')
    exp_b_parser.add_argument('--quick', action='store_true', help='Quick mode (50 epochs)')
    exp_b_parser.set_defaults(func=cmd_exp_b)
    
    # exp-c
    exp_c_parser = subparsers.add_parser(
        'exp-c',
        help='Run Experiment C: Dreaming/Generation'
    )
    exp_c_parser.add_argument('--project', help='Project name to use')
    exp_c_parser.add_argument('--seed', type=int, help='Random seed')
    exp_c_parser.add_argument('--quick', action='store_true', help='Quick mode (100 epochs)')
    exp_c_parser.add_argument('--init-mode', choices=['random', 'centroid'], 
                              help="Initialization mode: 'random' (default) or 'centroid' (dataset mean)")
    exp_c_parser.add_argument('--centroid-noise-scale', type=float,
                              help='Gaussian perturbation scale when using centroid init (default: 0.5)')
    exp_c_parser.set_defaults(func=cmd_exp_c)
    
    # exp-d
    exp_d_parser = subparsers.add_parser(
        'exp-d',
        help='Run Experiment D: SO(2) Goldstone Memory (V2)'
    )
    exp_d_parser.add_argument('--project', help='Project name to use')
    exp_d_parser.add_argument('--seed', type=int, help='Random seed')
    exp_d_parser.add_argument('--quick', action='store_true', help='Quick mode (100 epochs)')
    exp_d_parser.add_argument('--potential-type', choices=['so2_invariant', 'mlp'],
                              help="Potential: 'so2_invariant' (designed symmetry, default) or 'mlp' (emergent)")
    exp_d_parser.add_argument('--broken-isotropy', action='store_true',
                              help='Untie the channel inertial masses (F5 §4.1 falsifiable)')
    exp_d_parser.add_argument('--tilt-delta', type=float,
                              help='Explicit SO(2)-breaking amplitude delta (GMOR probe)')
    exp_d_parser.add_argument('--tilt-n', type=int, help='Tilt harmonic n')
    exp_d_parser.add_argument('--sleep-mode', choices=['on', 'off'],
                              help='Sleep phase: "on" (wake-sleep) or "off" '
                                   '(wake-only, data-pinned, no vacuum erosion)')
    exp_d_parser.set_defaults(func=cmd_exp_d)

    # exp-v1-gate
    exp_v1_parser = subparsers.add_parser(
        'exp-v1-gate',
        help='Run V1 L0 gate: boost-retry cascade on MQAR associative recall'
    )
    exp_v1_parser.add_argument('--project', help='Project name to use')
    exp_v1_parser.add_argument('--seed', type=int, help='Random seed')
    exp_v1_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (small grid, short training)')
    exp_v1_parser.set_defaults(func=cmd_exp_v1_gate)

    # exp-lattice
    exp_lattice_parser = subparsers.add_parser(
        'exp-lattice',
        help='Run CLU-lattice experiment: communication pricing + scaling (V3)'
    )
    exp_lattice_parser.add_argument('--project', help='Project name to use')
    exp_lattice_parser.add_argument('--seed', type=int, help='Random seed')
    exp_lattice_parser.add_argument('--quick', action='store_true',
                                    help='Quick mode (short sweep, 60 epochs)')
    exp_lattice_parser.add_argument('--skip-training', action='store_true',
                                    help='Skip the banded-vs-uniform training smoke')
    exp_lattice_parser.set_defaults(func=cmd_exp_lattice)

    # exp-v1-calib
    exp_v1c_parser = subparsers.add_parser(
        'exp-v1-calib',
        help='Run V1 pivot: learned calibration gate + compute allocation on MQAR'
    )
    exp_v1c_parser.add_argument('--project', help='Project name to use')
    exp_v1c_parser.add_argument('--seed', type=int,
                                help='Base random seed (replicates = seed + i)')
    exp_v1c_parser.add_argument('--quick', action='store_true',
                                help='Quick mode (small grid, short training)')
    exp_v1c_parser.set_defaults(func=cmd_exp_v1_calibration)

    # exp-v1-regime
    exp_v1r_parser = subparsers.add_parser(
        'exp-v1-regime',
        help='Run V1 CLU-vs-Hopfield regime map (capacity x stress sweep)'
    )
    exp_v1r_parser.add_argument('--project', help='Project name to use')
    exp_v1r_parser.add_argument('--seed', type=int,
                                help='Base random seed (replicates = seed + i)')
    exp_v1r_parser.add_argument('--quick', action='store_true',
                                help='Quick mode (small grid, short training)')
    exp_v1r_parser.set_defaults(func=cmd_exp_v1_regime)

    # exp-s1
    exp_s1_parser = subparsers.add_parser(
        'exp-s1',
        help='Run Experiment S1: Trash-Region Pareto (learned friction field)'
    )
    exp_s1_parser.add_argument('--project', help='Project name to use')
    exp_s1_parser.add_argument('--seed', type=int, help='Random seed')
    exp_s1_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (1 seed, 60 epochs, short eval)')
    exp_s1_parser.set_defaults(func=cmd_exp_s1)

    # all-experiments
    all_parser = subparsers.add_parser(
        'all-experiments',
        help='Run all experiments sequentially'
    )
    all_parser.add_argument('--project', help='Project name to use')
    all_parser.add_argument('--seed', type=int, help='Random seed')
    all_parser.add_argument('--quick', action='store_true', help='Quick mode')
    all_parser.set_defaults(func=cmd_all_experiments)


def _get_config_and_paths(args):
    """Get configuration and paths from project or defaults."""
    pm = ProjectManager()
    
    if args.project:
        try:
            config = pm.load(args.project)
            paths = pm.get_paths(args.project)
            pm.update_last_run(args.project)
        except ValueError as e:
            console.print(f"✗ Error loading project: {e}", style="bold red")
            return None, None
    else:
        from ..config import get_default_config
        config = get_default_config()
        # Use current results directory if no project specified
        paths = {
            'plots': Path('results'),
            'results': Path('results'),
            'models': Path('results')
        }
        paths['plots'].mkdir(exist_ok=True)
    
    # Override seed if provided
    if args.seed is not None:
        config.project.seed = args.seed
    
    # Adjust epochs for quick mode
    if args.quick:
        if hasattr(config, 'experiment_a'):
            config.experiment_a.train_epochs = 50
        if hasattr(config, 'experiment_b'):
            config.experiment_b.train_epochs = 50
        if hasattr(config, 'experiment_c'):
            config.experiment_c.train_epochs = 100
        if hasattr(config, 'experiment_d'):
            config.experiment_d.train_epochs = 100

    return config, paths


def cmd_exp_a(args):
    """Run Experiment A."""
    console.print("[bold cyan]Running Experiment A: Stability Test[/bold cyan]")
    
    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1
    
    # Set save directory in config
    config.project.save_dir = str(paths['plots'])
    
    try:
        run_experiment_a(config=config, models_dir=str(paths['models']))
        console.print("✓ Experiment A completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    
    return 0


def cmd_exp_b(args):
    """Run Experiment B."""
    console.print("[bold cyan]Running Experiment B: Noise Rejection[/bold cyan]")
    
    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1
    
    # Set save directory in config
    config.project.save_dir = str(paths['plots'])
    
    try:
        run_experiment_b(config=config, models_dir=str(paths['models']))
        console.print("✓ Experiment B completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    
    return 0


def cmd_exp_c(args):
    """Run Experiment C."""
    console.print("[bold cyan]Running Experiment C: Dreaming[/bold cyan]")
    
    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1
    
    # Set save directory in config
    config.project.save_dir = str(paths['plots'])
    
    # Extract CLI overrides
    kwargs = {'config': config, 'models_dir': str(paths['models'])}
    if hasattr(args, 'init_mode') and args.init_mode is not None:
        kwargs['init_mode'] = args.init_mode
    if hasattr(args, 'centroid_noise_scale') and args.centroid_noise_scale is not None:
        kwargs['centroid_noise_scale'] = args.centroid_noise_scale
    
    try:
        run_experiment_c(**kwargs)
        console.print("✓ Experiment C completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    
    return 0


def cmd_exp_d(args):
    """Run Experiment D."""
    console.print("[bold cyan]Running Experiment D: SO(2) Goldstone Memory[/bold cyan]")

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    # Set save directory in config
    config.project.save_dir = str(paths['plots'])

    # Extract CLI overrides (None = keep config value)
    kwargs = {'config': config, 'models_dir': str(paths['models'])}
    if getattr(args, 'potential_type', None) is not None:
        kwargs['potential_type'] = args.potential_type
    if getattr(args, 'broken_isotropy', False):
        kwargs['tie_channel_mass'] = False
    if getattr(args, 'tilt_delta', None) is not None:
        kwargs['tilt_delta'] = args.tilt_delta
    if getattr(args, 'tilt_n', None) is not None:
        kwargs['tilt_n'] = args.tilt_n
    if getattr(args, 'sleep_mode', None) is not None:
        kwargs['sleep_mode'] = args.sleep_mode

    try:
        run_experiment_d(**kwargs)
        console.print("✓ Experiment D completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_v1_gate(args):
    """Run the V1 L0 gate experiment."""
    console.print("[bold cyan]Running V1 L0 Gate: boost-retry cascade on MQAR[/bold cyan]")

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    try:
        run_experiment_v1_gate(
            config=config,
            models_dir=str(paths['models']),
            quick=bool(getattr(args, 'quick', False)),
        )
        console.print("✓ V1 gate experiment completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_lattice(args):
    """Run the CLU-lattice experiment."""
    console.print("[bold cyan]Running CLU-Lattice Experiment: communication pricing (V3)[/bold cyan]")

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    try:
        run_experiment_lattice(
            config=config,
            models_dir=str(paths['models']),
            quick=bool(getattr(args, 'quick', False)),
            skip_training=bool(getattr(args, 'skip_training', False)),
        )
        console.print("✓ Lattice experiment completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_v1_calibration(args):
    """Run the pivoted V1 calibration experiment."""
    console.print(
        "[bold cyan]Running V1 Calibration: learned energy gate on MQAR[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    try:
        run_experiment_v1_calibration(
            config=config,
            models_dir=str(paths['models']),
            quick=bool(getattr(args, 'quick', False)),
        )
        console.print("✓ V1 calibration experiment completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_v1_regime(args):
    """Run the V1 CLU-vs-Hopfield regime map."""
    console.print(
        "[bold cyan]Running V1 Regime Map: CLU gate vs Hopfield under stress[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    try:
        run_v1_hopfield_regime_map(
            config=config,
            models_dir=str(paths['models']),
            quick=bool(getattr(args, 'quick', False)),
        )
        console.print("✓ V1 regime map completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_s1(args):
    """Run the S1 trash-region Pareto pilot."""
    console.print(
        "[bold cyan]Running Experiment S1: Trash-Region Pareto (gamma-field)[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    if getattr(args, 'quick', False):
        config.experiment_s1.train_epochs = 60
        config.experiment_s1.seeds = config.experiment_s1.seeds[:1]
        config.experiment_s1.eval_clean_steps = 1000
        config.experiment_s1.n_injections = 8

    try:
        run_experiment_s1(config=config, models_dir=str(paths['models']))
        console.print("✓ Experiment S1 completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_all_experiments(args):
    """Run all experiments."""
    console.print("[bold cyan]Running All Experiments[/bold cyan]")
    
    experiments = [
        ('A: Stability', cmd_exp_a),
        ('B: Noise Rejection', cmd_exp_b),
        ('C: Dreaming', cmd_exp_c)
    ]
    
    for name, func in experiments:
        console.print(f"\n[bold]Starting {name}...[/bold]")
        result = func(args)
        if result != 0:
            console.print(f"✗ Failed at experiment {name}", style="bold red")
            return result
    
    console.print("\n✓ All experiments completed successfully!", style="bold green")
    return 0
