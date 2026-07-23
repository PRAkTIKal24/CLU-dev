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
from ..experiments.exp_v1_wormhole import run_experiment_v1_wormhole
from ..experiments.exp_lattice import run_experiment_lattice
from ..experiments.exp_s1_gamma_field import run_experiment_s1
from ..experiments.exp_dim_scaling import (
    run_experiment_dim_scaling,
    apply_quick as apply_dim_scaling_quick,
)
from ..experiments.exp_learned_memory import (
    run_experiment_learned_memory,
    apply_quick as apply_learned_memory_quick,
)
from ..experiments.exp_potential_class import (
    run_experiment_potential_class,
    apply_quick as apply_potential_class_quick,
)
from ..experiments.exp_designed_mechanism import (
    run_experiment_designed_mechanism,
    apply_quick as apply_designed_mechanism_quick,
)
from ..experiments.exp_retrieval import (
    run_experiment_retrieval,
    apply_quick as apply_retrieval_quick,
)
from ..experiments.exp_hopfield_capacity import (
    run_experiment_hopfield_capacity,
    apply_quick as apply_hopfield_capacity_quick,
)
from ..experiments.kt import KT_MODES

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
    exp_d_parser.add_argument('--anchor-lambda', type=float,
                              help='V(data)-energy anchor weight (sleep-erosion '
                                   'cure): pins mean V on the data manifold. '
                                   '0=off (default); 10-100 hold the SO(2) '
                                   'vacuum under wake-sleep CD.')
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
    exp_v1r_parser.add_argument('--train-epochs', type=int,
                                help='Per-cell PCD write epochs (compute-parity '
                                     'knob; anchor-robustness P14: 500->2000 '
                                     'closes the Hopfield gap). Overrides '
                                     'experiment_v1_gate.train_epochs.')
    exp_v1r_parser.set_defaults(func=cmd_exp_v1_regime)

    # exp-v1-wormhole
    exp_v1w_parser = subparsers.add_parser(
        'exp-v1-wormhole',
        help='Run V1 pillar 3: energy-gated sparse non-local routing (wormholes)'
    )
    exp_v1w_parser.add_argument('--project', help='Project name to use')
    exp_v1w_parser.add_argument('--seed', type=int,
                                help='Base random seed (replicates = seed + i)')
    exp_v1w_parser.add_argument('--quick', action='store_true',
                                help='Quick mode (N=4, 1 seed, short training)')
    exp_v1w_parser.set_defaults(func=cmd_exp_v1_wormhole)

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

    # exp-retrieval
    exp_ret_parser = subparsers.add_parser(
        'exp-retrieval',
        help='Run the hand-built write/address/retrieve loop (addressable memory, stage 1)'
    )
    exp_ret_parser.add_argument('--project', help='Project name to use')
    exp_ret_parser.add_argument('--seed', type=int, help='Random seed')
    exp_ret_parser.add_argument('--quick', action='store_true',
                                help='Quick mode (fewer queries/items/steps)')
    exp_ret_parser.set_defaults(func=cmd_exp_retrieval)

    # exp-dim-scaling
    exp_dim_parser = subparsers.add_parser(
        'exp-dim-scaling',
        help='Measure retrieval capacity K_max vs address-space dimension d'
    )
    exp_dim_parser.add_argument('--project', help='Project name to use')
    exp_dim_parser.add_argument('--seed', type=int, help='Random seed')
    exp_dim_parser.add_argument('--quick', action='store_true',
                                help='Quick mode (d<=3, K<=16, short rollouts)')
    exp_dim_parser.set_defaults(func=cmd_exp_dim_scaling)

    # exp-hopfield-capacity
    exp_hc_parser = subparsers.add_parser(
        'exp-hopfield-capacity',
        help='CLU vs modern-Hopfield/U-Hop SOTA on the associative-memory '
             'retrieval benchmark (designed register, w22)'
    )
    exp_hc_parser.add_argument('--project', help='Project name to use')
    exp_hc_parser.add_argument('--seed', type=int, help='Random seed')
    exp_hc_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (small load grid, short rollouts)')
    exp_hc_parser.add_argument('--dataset',
                               help='Override datasets (comma-separated)')
    exp_hc_parser.set_defaults(func=cmd_exp_hopfield_capacity)
    # exp-learned-memory
    exp_lm_parser = subparsers.add_parser(
        'exp-learned-memory',
        help='Does the write/address/read loop survive a LEARNED landscape? '
             '(design-freedom sweep, w20)'
    )
    exp_lm_parser.add_argument('--project', help='Project name to use')
    exp_lm_parser.add_argument('--seed', type=int, help='Random seed')
    exp_lm_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (2 rungs, K=2, short rollouts)')
    exp_lm_parser.set_defaults(func=cmd_exp_learned_memory)
    # exp-potential-class
    exp_pc_parser = subparsers.add_parser(
        'exp-potential-class',
        help='Is the learned-landscape failure EXPRESSIVITY or SUPPORT STRUCTURE? '
             '(MLP vs modern-Hopfield/attention vs atom dictionary, w21)'
    )
    exp_pc_parser.add_argument('--project', help='Project name to use')
    exp_pc_parser.add_argument('--seed', type=int, help='Random seed')
    exp_pc_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (3 arms, K=2, 1 seed, tiny sizes)')
    exp_pc_parser.add_argument('--classes', nargs='+',
                               help='Override the swept potential classes')
    exp_pc_parser.set_defaults(func=cmd_exp_potential_class)
    # exp-designed-mechanism
    exp_dm_parser = subparsers.add_parser(
        'exp-designed-mechanism',
        help='Is the K=8 wall GEOMETRY or LEARNING? K_learned vs d for a learned '
             'atom-dictionary mechanism, vs the designed 4*2^d ceiling (w22)'
    )
    exp_dm_parser.add_argument('--project', help='Project name to use')
    exp_dm_parser.add_argument('--seed', type=int, help='Random seed')
    exp_dm_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (d<=3, K<=8, 2 seeds, tiny writes)')
    exp_dm_parser.set_defaults(func=cmd_exp_designed_mechanism)
    # exp-primitive-harness
    exp_ph_parser = subparsers.add_parser(
        'exp-primitive-harness',
        help='Compare CLU vs MLP/GRU/SSM/attention in one drop-in slot at matched params'
    )
    exp_ph_parser.add_argument('--project', help='Project name to use')
    exp_ph_parser.add_argument('--seed', type=int, help='Random seed')
    exp_ph_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (30 steps, 1 lr, 1 seed, tiny budget)')
    exp_ph_parser.add_argument('--families', nargs='+',
                               choices=['mqar', 'adding', 'parity'],
                               help='Subset of task families to run')
    exp_ph_parser.add_argument('--steps', type=int, help='Override train_steps')
    exp_ph_parser.add_argument('--gamma-sweep', action='store_true',
                               help='Run ONLY the w21 CLU-internal gamma / read-mode / '
                                    'clu_steps sweep (baselines untouched)')
    exp_ph_parser.add_argument('--sweep-items', nargs='+',
                               choices=['gamma', 'read', 'steps'],
                               help='Subset of w21 sweep items (default: all three)')
    exp_ph_parser.set_defaults(func=cmd_exp_primitive_harness)

    # exp-gated-write
    exp_gw_parser = subparsers.add_parser(
        'exp-gated-write',
        help='w22 gated-write performance test: is gated CLU competitive, and '
             'is there a physics edge vs a matched gated GRU/SSM?'
    )
    exp_gw_parser.add_argument('--project', help='Project name to use')
    exp_gw_parser.add_argument('--seed', type=int, help='Random seed')
    exp_gw_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (tiny budget, 1 seed, smoke)')
    exp_gw_parser.add_argument('--items', nargs='+',
                               choices=['item1', '3a', '3b', '3c', 'cost'],
                               help='Subset of items to run (default: all)')
    exp_gw_parser.add_argument('--families', nargs='+',
                               choices=['adding', 'parity', 'mqar'],
                               help='Item 1 family subset (default: all three)')
    exp_gw_parser.add_argument('--out', default='gated_write.json',
                               help='Output JSON filename')
    exp_gw_parser.set_defaults(func=cmd_exp_gated_write)

    # exp-sequential-write
    exp_sw_parser = subparsers.add_parser(
        'exp-sequential-write',
        help='Sequential-write interference: does an MVC-0 admission gate stop '
             'new writes destroying stored items? (w21)'
    )
    exp_sw_parser.add_argument('--project', help='Project name to use')
    exp_sw_parser.add_argument('--seed', type=int, help='Random seed')
    exp_sw_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (2 seeds, 4 items, short writes)')
    exp_sw_parser.add_argument('--items', nargs='+', choices=['1', '2', '3', '4'],
                               help='Subset of items to run (default: all)')
    exp_sw_parser.set_defaults(func=cmd_exp_sequential_write)

    # exp-controller-mvp
    exp_cm_parser = subparsers.add_parser(
        'exp-controller-mvp',
        help='MVC-0 controller (admission+placement+eviction, no learning) on a '
             'designed store; the N75 rematch, per-admitted vs per-offered (w23)'
    )
    exp_cm_parser.add_argument('--project', help='Project name to use')
    exp_cm_parser.add_argument('--seed', type=int, help='Random seed')
    exp_cm_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (2 seeds, short K ladder)')
    exp_cm_parser.add_argument('--items', nargs='+', choices=['1', '2', '3'],
                               help='Subset of items to run (default: all)')
    exp_cm_parser.set_defaults(func=cmd_exp_controller_mvp)

    # exp-minus-physics
    exp_mp_parser = subparsers.add_parser(
        'exp-minus-physics',
        help='Run the CLU-minus-the-physics G2 controls (non-symplectic twins)'
    )
    exp_mp_parser.add_argument('--project', help='Project name to use')
    exp_mp_parser.add_argument('--seed', type=int, help='Base random seed')
    exp_mp_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (2 seeds, 60 epochs, short probes)')
    exp_mp_parser.set_defaults(func=cmd_exp_minus_physics)

    # exp-paid-access
    exp_pa_parser = subparsers.add_parser(
        'exp-paid-access',
        help='Run the w7 paid-access battery (reach/escape, wormhole vs squeeze)'
    )
    exp_pa_parser.add_argument('--project', help='Project name to use')
    exp_pa_parser.add_argument('--seed', type=int, help='Base random seed')
    exp_pa_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (2 seeds, distances straddling L)')
    exp_pa_parser.set_defaults(func=cmd_exp_paid_access)

    # exp-v1-hopfield-gate
    exp_v1hg_parser = subparsers.add_parser(
        'exp-v1-hopfield-gate',
        help='Run the V1.1 gate stack on a Hopfield memory (memory-agnostic check)'
    )
    exp_v1hg_parser.add_argument('--project', help='Project name to use')
    exp_v1hg_parser.add_argument('--seed', type=int, help='Base random seed')
    exp_v1hg_parser.add_argument('--quick', action='store_true', help='Quick mode')
    exp_v1hg_parser.add_argument('--rho', type=float, help='embedding correlation stress')
    exp_v1hg_parser.add_argument('--noise', type=float, help='cue eval-noise stress')
    exp_v1hg_parser.set_defaults(func=cmd_exp_v1_hopfield_gate)

    # exp-kt
    exp_kt_parser = subparsers.add_parser(
        'exp-kt',
        help='Run the Kosterlitz-Thouless memory-phase suite (Thread-10)'
    )
    exp_kt_parser.add_argument('--mode', choices=list(KT_MODES), default='winding1d',
                               help='winding1d (1-D CLU ring, GPU) | winding2d '
                                    '(2-D survival, CPU) | bridge (kill criterion, '
                                    'GPU) | reduced (phase diagram, CPU) | postproc')
    exp_kt_parser.add_argument('--project', help='Project name to use')
    exp_kt_parser.add_argument('--seed', type=int, help='Random seed (overrides the '
                                                        "mode's configured seed)")
    exp_kt_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (tiny L/N, few steps; same code path)')
    exp_kt_parser.add_argument('--out', help='Output directory for JSON/figures '
                                             '(default: <project>/results/kt)')
    exp_kt_parser.add_argument('--task-id', type=int,
                               help='Run ONE cell of the mode sweep grid '
                                    '(= $SLURM_ARRAY_TASK_ID); omit to run all')
    exp_kt_parser.add_argument('--no-figures', action='store_true',
                               help='postproc: write summary.json only')
    # Tranche overrides (the two soft exponents). None = keep config value.
    exp_kt_parser.add_argument('--tj', type=float,
                               help='winding1d: T/J for the 1-D ring run. Use '
                                    '<=0.2 (NOT the 0.5 originally scoped): at '
                                    'T/J>=0.5 the winding is barely metastable '
                                    '(E_wind/T~2-5) so the MSD saturates and the '
                                    'fitted exponent is an artifact. Pair with '
                                    '--msd-fit-max.')
    exp_kt_parser.add_argument('--n-values', type=int, nargs='+',
                               help='winding1d: ring sizes N')
    exp_kt_parser.add_argument('--l-values', type=int, nargs='+',
                               help='winding2d/reduced: linear sizes L (>=32 resolves '
                                    'the above-T_KT sign change)')
    exp_kt_parser.add_argument('--tj-values', type=float, nargs='+',
                               help='winding2d/bridge: T/J grid')
    exp_kt_parser.add_argument('--walkers', type=int,
                               help='winding1d/bridge: vmapped Langevin walkers')
    exp_kt_parser.add_argument('--chunks', type=int,
                               help='winding1d: measurement chunks (all N)')
    exp_kt_parser.add_argument('--chunk-steps', type=int,
                               help='winding1d: Langevin steps per chunk')
    exp_kt_parser.add_argument('--nwalk-2d', type=int,
                               help='winding2d: first-passage walkers per cell')
    exp_kt_parser.add_argument('--msd-fit-max', type=float,
                               help='winding1d: fit the MSD only over the '
                                    'DIFFUSIVE window MSD <= this (e.g. 0.3). '
                                    'Without it the full-range fit is '
                                    'saturation-dominated and the exponent is '
                                    'an artifact. Required for exponent runs.')
    exp_kt_parser.add_argument('--nmax-2d', type=int,
                               help='winding2d: first-passage censor (sweeps), '
                                    'applied to BOTH below/above T_KT')
    exp_kt_parser.set_defaults(func=cmd_exp_kt)

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
    if getattr(args, 'anchor_lambda', None) is not None:
        kwargs['anchor_lambda'] = args.anchor_lambda

    try:
        run_experiment_d(**kwargs)
        console.print("✓ Experiment D completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_minus_physics(args):
    """Run the CLU-minus-the-physics G2 controls."""
    console.print("[bold cyan]Running CLU minus the physics (G2 controls)[/bold cyan]")

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1
    config.project.save_dir = str(paths['plots'])

    from ..experiments.exp_minus_physics import run_experiment_minus_physics
    try:
        run_experiment_minus_physics(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            quick=bool(getattr(args, 'quick', False)),
        )
        console.print("✓ minus-the-physics completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    return 0


def cmd_exp_paid_access(args):
    """Run the w7 paid-access battery (intra-unit wormhole vs squeeze reach)."""
    console.print("[bold cyan]Running paid-access battery (reach/escape gate)[/bold cyan]")

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1
    config.project.save_dir = str(paths['plots'])

    from ..experiments.exp_paid_access import run_experiment_paid_access
    try:
        run_experiment_paid_access(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=int(getattr(args, 'seed', None) or 0),
            quick=bool(getattr(args, 'quick', False)),
        )
        console.print("✓ paid-access battery completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    return 0


def cmd_exp_v1_hopfield_gate(args):
    """Run the V1.1 gate stack on a Hopfield memory."""
    console.print("[bold cyan]Running V1.1: gate stack on a Hopfield memory[/bold cyan]")

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1
    config.project.save_dir = str(paths['plots'])
    if getattr(args, 'rho', None) is not None:
        config.experiment_v1_gate.hopfield_gate_correlation = args.rho
    if getattr(args, 'noise', None) is not None:
        config.experiment_v1_gate.hopfield_gate_eval_noise = args.noise

    from ..experiments.exp_v1_hopfield_gate import run_experiment_v1_hopfield_gate
    try:
        run_experiment_v1_hopfield_gate(
            config=config, save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            quick=bool(getattr(args, 'quick', False)),
        )
        console.print("✓ V1.1 hopfield-gate completed", style="bold green")
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
    if getattr(args, 'train_epochs', None) is not None:
        config.experiment_v1_gate.train_epochs = args.train_epochs

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


def cmd_exp_v1_wormhole(args):
    """Run the V1 wormhole-routing experiment."""
    console.print(
        "[bold cyan]Running V1 Wormhole: energy-gated sparse non-local routing"
        "[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    try:
        run_experiment_v1_wormhole(
            config=config,
            models_dir=str(paths['models']),
            quick=bool(getattr(args, 'quick', False)),
        )
        console.print("✓ V1 wormhole experiment completed", style="bold green")
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


def cmd_exp_primitive_harness(args):
    """Run the primitive harness (CLU vs MLP/GRU/SSM/attention, matched params)."""
    console.print(
        "[bold cyan]Running PRIMITIVE HARNESS: CLU vs MLP/GRU/SSM/attention "
        "in one drop-in slot[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])
    if getattr(args, 'steps', None):
        config.experiment_primitive_harness.train_steps = args.steps

    try:
        if getattr(args, 'gamma_sweep', False):
            from ..experiments.exp_primitive_harness import run_gamma_read_sweep

            run_gamma_read_sweep(
                config=config,
                save_dir=str(paths['plots']),
                families=getattr(args, 'families', None),
                items=getattr(args, 'sweep_items', None),
                quick=getattr(args, 'quick', False),
            )
            console.print("✓ w21 gamma/read sweep completed", style="bold green")
            return 0

        from ..experiments.exp_primitive_harness import run_primitive_harness

        run_primitive_harness(
            config=config,
            save_dir=str(paths['plots']),
            seed=getattr(args, 'seed', None),
            quick=getattr(args, 'quick', False),
            families=getattr(args, 'families', None),
        )
        console.print("✓ Primitive harness completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_gated_write(args):
    """Run the w22 gated-write performance test."""
    console.print(
        "[bold cyan]Running GATED-WRITE performance test: is gated CLU "
        "competitive, and is there a physics edge?[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    try:
        from ..experiments.exp_gated_write import run_gated_write

        run_gated_write(
            config=config,
            save_dir=str(paths['plots']),
            items=getattr(args, 'items', None),
            families=getattr(args, 'families', None),
            out_name=getattr(args, 'out', 'gated_write.json'),
            quick=bool(getattr(args, 'quick', False)),
        )
        console.print("✓ Gated-write performance test completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_sequential_write(args):
    """Run the sequential-write interference / admission-gate battery (w21)."""
    console.print(
        "[bold cyan]Running Experiment SEQUENTIAL-WRITE: gated vs ungated "
        "writes, the retention curve, and the cross-primitive comparison"
        "[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    try:
        from ..experiments.exp_sequential_write import (
            apply_quick as apply_seqwrite_quick,
        )
        from ..experiments.exp_sequential_write import (
            run_experiment_sequential_write,
        )

        if getattr(args, 'quick', False):
            apply_seqwrite_quick(config)

        res = run_experiment_sequential_write(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=getattr(args, 'seed', None),
            items=getattr(args, 'items', None),
        )
        console.print(
            f"✓ Experiment SEQUENTIAL-WRITE completed -> {res['metrics_path']}",
            style="bold green",
        )
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_controller_mvp(args):
    """Run the MVC-0 controller battery + the N75 rematch (w23)."""
    console.print(
        "[bold cyan]Running Experiment CONTROLLER-MVP: hand-coded controller on a "
        "designed store; retention-vs-K per-admitted and per-offered[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    try:
        from ..experiments.exp_controller_mvp import (
            apply_quick as apply_ctrl_quick,
        )
        from ..experiments.exp_controller_mvp import (
            run_experiment_controller_mvp,
        )

        if getattr(args, 'quick', False):
            apply_ctrl_quick(config)

        res = run_experiment_controller_mvp(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=getattr(args, 'seed', None),
            items=getattr(args, 'items', None),
        )
        console.print(
            f"✓ Experiment CONTROLLER-MVP completed -> {res['metrics_path']}",
            style="bold green",
        )
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_retrieval(args):
    """Run the hand-built write -> address -> retrieve battery."""
    console.print(
        "[bold cyan]Running Experiment RETRIEVAL: addressable memory loop "
        "(HAND-DESIGNED, not learned)[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    if getattr(args, 'quick', False):
        apply_retrieval_quick(config)

    try:
        res = run_experiment_retrieval(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=getattr(args, 'seed', None),
        )
        console.print(f"✓ Experiment RETRIEVAL completed -> {res['metrics_path']}",
                      style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_dim_scaling(args):
    """Run the address-space dimension-scaling capacity measurement."""
    console.print(
        "[bold cyan]Running Experiment DIM-SCALING: K_max vs address dimension (HAND-DESIGNED, not learned)[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    if getattr(args, 'quick', False):
        apply_dim_scaling_quick(config)

    try:
        res = run_experiment_dim_scaling(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=getattr(args, 'seed', None),
        )
        console.print(f"\u2713 Experiment DIM-SCALING completed -> {res['metrics_path']}",
                      style="bold green")
    except Exception as e:
        console.print(f"\u2717 Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_hopfield_capacity(args):
    """Run the Hopfield-capacity benchmark: CLU vs modern-Hopfield/U-Hop SOTA."""
    console.print(
        "[bold cyan]Running Experiment HOPFIELD-CAPACITY: CLU designed register "
        "vs modern-Hopfield/U-Hop SOTA (DESIGNED, not learned)[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    if getattr(args, 'quick', False):
        apply_hopfield_capacity_quick(config)
    if getattr(args, 'dataset', None):
        config.experiment_hopfield_capacity.datasets = args.dataset.split(',')

    try:
        res = run_experiment_hopfield_capacity(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=getattr(args, 'seed', None),
        )
        console.print(
            f"✓ Experiment HOPFIELD-CAPACITY completed -> {res['metrics_path']}",
            style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_learned_memory(args):
    """Run the LEARNED write -> address -> read loop + design-freedom sweep."""
    console.print(
        "[bold cyan]Running Experiment LEARNED-MEMORY: does the retrieval loop survive a LEARNED landscape?[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    if getattr(args, 'quick', False):
        apply_learned_memory_quick(config)

    try:
        res = run_experiment_learned_memory(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=getattr(args, 'seed', None),
        )
        console.print(f"\u2713 Experiment LEARNED-MEMORY completed -> {res['metrics_path']}",
                      style="bold green")
    except Exception as e:
        console.print(f"\u2717 Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_potential_class(args):
    """Run the potential FUNCTION-CLASS sweep (expressivity vs support structure)."""
    console.print(
        "[bold cyan]Running Experiment POTENTIAL-CLASS: expressivity or support structure?[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    if getattr(args, 'quick', False):
        apply_potential_class_quick(config)
    if getattr(args, 'classes', None):
        config.experiment_potential_class.potential_classes = list(args.classes)

    try:
        res = run_experiment_potential_class(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=getattr(args, 'seed', None),
        )
        console.print(f"✓ Experiment POTENTIAL-CLASS completed -> {res['metrics_path']}",
                      style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_designed_mechanism(args):
    """Run the K=8-wall discriminator: is it GEOMETRY or LEARNING?"""
    console.print(
        "[bold cyan]Running Experiment DESIGNED-MECHANISM: K_learned vs d (is the K=8 wall geometry or learning?)[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    if getattr(args, 'quick', False):
        apply_designed_mechanism_quick(config)

    try:
        res = run_experiment_designed_mechanism(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=getattr(args, 'seed', None),
        )
        console.print(
            f"✓ Experiment DESIGNED-MECHANISM completed -> {res['metrics_path']}",
            style="bold green",
        )
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_kt(args):
    """Run one mode of the Kosterlitz-Thouless memory-phase suite."""
    console.print(
        f"[bold cyan]Running KT suite: mode={args.mode}[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    kt = config.experiment_kt
    # CLI overrides (None = keep config value)
    if getattr(args, 'tj', None) is not None:
        kt.winding1d_tj = args.tj
    if getattr(args, 'n_values', None):
        kt.winding1d_n_values = list(args.n_values)
    if getattr(args, 'l_values', None):
        kt.winding2d_l_values = list(args.l_values)
        kt.reduced_l_values = list(args.l_values)
    if getattr(args, 'tj_values', None):
        kt.winding2d_tj_values = list(args.tj_values)
        kt.bridge_tj_values = list(args.tj_values)
    if getattr(args, 'walkers', None) is not None:
        kt.n_walkers = args.walkers
    if getattr(args, 'chunks', None) is not None:
        kt.winding1d_chunks = args.chunks
        kt.winding1d_chunks_large = args.chunks
    if getattr(args, 'chunk_steps', None) is not None:
        kt.winding1d_chunk_steps = args.chunk_steps
    if getattr(args, 'msd_fit_max', None) is not None:
        kt.winding1d_msd_fit_max = args.msd_fit_max
    if getattr(args, 'nwalk_2d', None) is not None:
        kt.winding2d_nwalk = args.nwalk_2d
    if getattr(args, 'nmax_2d', None) is not None:
        kt.winding2d_nmax_below = args.nmax_2d
        kt.winding2d_nmax_above = args.nmax_2d

    out_dir = args.out or str(Path(paths['results']) / 'kt')

    from ..experiments.kt import run_kt
    try:
        run_kt(
            config=config,
            mode=args.mode,
            out_dir=out_dir,
            seed=getattr(args, 'seed', None),
            quick=bool(getattr(args, 'quick', False)),
            task_id=getattr(args, 'task_id', None),
            make_figures=not bool(getattr(args, 'no_figures', False)),
            log=console.print,
        )
        console.print(f"✓ KT {args.mode} completed -> {out_dir}", style="bold green")
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
