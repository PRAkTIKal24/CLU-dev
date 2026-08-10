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
from ..experiments.exp_write_ceiling import (
    run_experiment_write_ceiling,
    apply_quick as apply_write_ceiling_quick,
)
from ..experiments.exp_well_lifecycle import (
    run_experiment_well_lifecycle,
)
from ..experiments.exp_capture_armB import (
    run_experiment_capture_armb,
)
# --- BEGIN c2w8p3-capture-strong-phi (additive) ---
from ..experiments.exp_capture_strong_phi import (
    run_experiment_capture_strong_phi,
)
# --- END c2w8p3-capture-strong-phi ---
# --- BEGIN c2w8p3-phi-geometry (additive) ---
from ..experiments.exp_phi_geometry import (
    run_experiment_phi_geometry,
)
# --- END c2w8p3-phi-geometry ---
from ..experiments.exp_sharded_store import (
    run_experiment_sharded_store,
    apply_quick as apply_sharded_store_quick,
)
from ..experiments.exp_retrieval import (
    run_experiment_retrieval,
    apply_quick as apply_retrieval_quick,
)
from ..experiments.exp_hopfield_capacity import (
    run_experiment_hopfield_capacity,
    apply_quick as apply_hopfield_capacity_quick,
)
from ..experiments.exp_phi_read_in import (
    run_experiment_phi_read_in,
    apply_quick as apply_phi_read_in_quick,
)
from ..experiments.exp_cl_entry import (
    run_experiment_cl_entry,
    apply_cifar10 as apply_cl_entry_cifar10,
    apply_quick as apply_cl_entry_quick,
)
from ..experiments.exp_phi_stream import (
    run_experiment_phi_stream,
    apply_quick as apply_phi_stream_quick,
)
from ..experiments.exp_retry_compute import (
    run_experiment_retry_compute,
    run_headroom_gate as run_retry_compute_headroom_gate,
    apply_ambiguity as apply_retry_compute_ambiguity,
    apply_quick as apply_retry_compute_quick,
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

    # exp-phi-read-in
    exp_phi_parser = subparsers.add_parser(
        'exp-phi-read-in',
        help='Learned read-in φ around a DESIGNED store: Hopfield protocol '
             're-fought in φ-space vs kNN-in-φ + laundering control (w23)'
    )
    exp_phi_parser.add_argument('--project', help='Project name to use')
    exp_phi_parser.add_argument('--seed', type=int, help='Random seed')
    exp_phi_parser.add_argument('--quick', action='store_true',
                                help='Quick mode (small sweeps, short AE/rollouts)')
    exp_phi_parser.add_argument('--dataset',
                                help='Override datasets (comma-separated)')
    exp_phi_parser.add_argument('--arms',
                                help='Override φ arms (comma-separated: pca,ae)')
    exp_phi_parser.set_defaults(func=cmd_exp_phi_read_in)

    # exp-phi-stream
    exp_ps_parser = subparsers.add_parser(
        'exp-phi-stream',
        help='φ stream discipline on a class-incremental stream: task-1-only '
             '(PRIMARY) vs generic-frozen (declared upper bound) φ, the '
             'cost-of-strictness curve + kNN-in-φ laundering control (w24)'
    )
    exp_ps_parser.add_argument('--project', help='Project name to use')
    exp_ps_parser.add_argument('--seed', type=int,
                               help='Single seed (overrides cfg.seeds)')
    exp_ps_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (1 seed, small store, short AE)')
    exp_ps_parser.add_argument('--regimes',
                               help='Override φ regimes (comma-separated: '
                                    'task1_only,generic_frozen)')
    exp_ps_parser.add_argument('--arms',
                               help='Override φ arms (comma-separated: pca,ae)')
    exp_ps_parser.set_defaults(func=cmd_exp_phi_stream)

    # exp-cl-entry
    exp_cl_parser = subparsers.add_parser(
        'exp-cl-entry',
        help='⭐ The continual-learning entry (w25): rehearsal-free Class-IL with a '
             'designed CLU store + task-1-only φ + MVC-0 controller, the mandatory '
             'baseline table (ER/iCaRL/GDumb/EWC/SI/LwF + kNN-in-φ launder), the '
             'R3-native retry ladder and the scheduled per-item retention demo'
    )
    exp_cl_parser.add_argument('--project', help='Project name to use')
    exp_cl_parser.add_argument('--seed', type=int,
                               help='Single seed (overrides cfg.seeds)')
    exp_cl_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (1 seed, tiny stream/store)')
    exp_cl_parser.add_argument('--dataset', choices=['mnist', 'cifar10'],
                               help='Override dataset (cifar10 also selects the CNN)')
    exp_cl_parser.add_argument(
        '--items',
        help='Comma-separated items: entry,retry,retention,frontier '
             '(frontier = the w26 matched-BYTES forgetting sweep)')
    exp_cl_parser.add_argument('--baselines',
                               help='Override the baseline list (comma-separated)')
    exp_cl_parser.add_argument(
        '--budgets',
        help='Comma-separated matched-BYTE budgets in floats (frontier item)')
    exp_cl_parser.set_defaults(func=cmd_exp_cl_entry)

    # exp-retry-compute
    exp_rc_parser = subparsers.add_parser(
        'exp-retry-compute',
        help='Accuracy-vs-compute curve for CLU retrieval: CLU-gated retry + 5 '
             'controls (ensemble/kick/ungated/feedforward/hopfield-k-steps, w23)'
    )
    exp_rc_parser.add_argument('--project', help='Project name to use')
    exp_rc_parser.add_argument('--seed', type=int, help='Random seed')
    exp_rc_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (small grid, short ladder/rollouts)')
    exp_rc_parser.add_argument('--dataset',
                               help='Override datasets (comma-separated)')
    exp_rc_parser.add_argument('--ambiguity', action='store_true',
                               help='w24 AMBIGUITY/headroom regimes (contiguous '
                                    'block occlusion + crowded store), 3 seeds')
    exp_rc_parser.add_argument('--headroom', action='store_true',
                               help='w24 Item 2 only: the cheap headroom gate '
                                    '(first-pass + NN floor), no ladder')
    exp_rc_parser.set_defaults(func=cmd_exp_retry_compute)
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
    # exp-write-ceiling
    exp_wc_parser = subparsers.add_parser(
        'exp-write-ceiling',
        help='Can any write operator (masked/sequential, scale-invariant, '
             'crowding-aware) break the d-independent K_ceiling~=32? (w24)'
    )
    exp_wc_parser.add_argument('--project', help='Project name to use')
    exp_wc_parser.add_argument('--seed', type=int, help='Random seed')
    exp_wc_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (d=2, K<=4, tiny writes)')
    exp_wc_parser.add_argument('--arms', nargs='+',
                               help='Override the swept write arms')
    exp_wc_parser.add_argument('--dims', nargs='+', type=int,
                               help='Override the swept address dimensions')
    exp_wc_parser.set_defaults(func=cmd_exp_write_ceiling)
    # exp-sharded-store
    exp_ssto_parser = subparsers.add_parser(
        'exp-sharded-store',
        help='The first N-unit sharded CLU store (Prop L2 / Theorem L1) and the '
             '2x2 discriminator: is the K~=32 write ceiling per-dig? (w25)'
    )
    exp_ssto_parser.add_argument('--project', help='Project name to use')
    exp_ssto_parser.add_argument('--seed', type=int, help='Random seed')
    exp_ssto_parser.add_argument('--quick', action='store_true',
                                 help='Quick mode (d=2, K=4, 1 seed, tiny writes)')
    exp_ssto_parser.add_argument('--cells', nargs='+',
                                 help='Override the swept cells (d:K:n_shards)')
    exp_ssto_parser.add_argument('--arms', nargs='+',
                                 help='Override the swept arms')
    exp_ssto_parser.add_argument('--items', nargs='+',
                                 help='Subset of items to run (1..6)')
    exp_ssto_parser.set_defaults(func=cmd_exp_sharded_store)
    # exp-well-lifecycle (C2W8 stage 1: the census that gates the lifecycle build)
    exp_wl_parser = subparsers.add_parser(
        'exp-well-lifecycle',
        help='C2W8 stage 1: the well census on the full CLU under an over-dug CL '
             'stream — is there anything to prune, anything to merge? (K1)'
    )
    exp_wl_parser.add_argument('--project', help='Project name to use')
    exp_wl_parser.add_argument('--seeds', help='Comma-separated seeds, e.g. 0,1,2')
    exp_wl_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (d=4, 1 seed, tiny stream)')
    exp_wl_parser.set_defaults(func=cmd_exp_well_lifecycle)
    # --- BEGIN c2w8p3-phi-geometry (additive) ---
    # exp-phi-geometry (C2W8 pass 3: the φ→addr map + the geometry it buys)
    exp_pg_parser = subparsers.add_parser(
        'exp-phi-geometry',
        help='C2W8 pass 3: build the phi_dim->addr_dim map the rig never had, then '
             'MEASURE whether strong phi separates at d in {8,12,16} on CIFAR-10 '
             '(sigma_q/spacing vs the PCA reference at matching d). Instrumentation.'
    )
    exp_pg_parser.add_argument('--project', help='Project name to use')
    exp_pg_parser.add_argument('--seeds', help='Comma-separated seeds, e.g. 0,1,2')
    exp_pg_parser.add_argument('--dims', help='Address dims, e.g. 8,12,16')
    exp_pg_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (tiny stream, 2 steps of phi)')
    exp_pg_parser.set_defaults(func=cmd_exp_phi_geometry)
    # --- END c2w8p3-phi-geometry ---
    # --- BEGIN c2w8p3-capture-strong-phi (additive) ---
    # exp-capture-strong-phi (C2W8 pass 3, THE SPINE: the completed gate at strong phi)
    exp_csp_parser = subparsers.add_parser(
        'exp-capture-strong-phi',
        help='C2W8 pass 3 SPINE: the frozen census + the COMPLETED gate '
             '(G-CAP/G-DEC/G-DRIFT/G-ADDR) on arm A co-scaled-width store over the '
             'strong-phi Split-CIFAR-10 rig, with an INTERNAL pca-phi reference at '
             'the same d. Both branches (daylight / no daylight) are reportable.'
    )
    exp_csp_parser.add_argument('--project', help='Project name to use')
    exp_csp_parser.add_argument('--seeds', help='Comma-separated seeds, e.g. 0,1,2')
    exp_csp_parser.add_argument('--arms', help='Comma-separated arms, '
                                               'e.g. randconv,simclr,pca')
    exp_csp_parser.add_argument('--addr-dim', type=int, dest='addr_dim',
                                help='Address dim (the joint dial with the atom '
                                     'budget); 12 => 32768 atoms. 16 is INERT.')
    exp_csp_parser.add_argument('--quick', action='store_true',
                                help='Quick mode (tiny stream, tiny phi, 1 seed)')
    exp_csp_parser.set_defaults(func=cmd_exp_capture_strong_phi)
    # --- END c2w8p3-capture-strong-phi ---
    # exp-capture-armb (C2W8 pass 2, ARM B: the emission head on pass 1's census)
    exp_armb_parser = subparsers.add_parser(
        'exp-capture-armb',
        help='C2W8 pass 2 ARM B: an MLP-class head on phi EMITS the well '
             'parameters (a forward pass instead of 300 gradient steps), scored '
             'on pass 1\'s census (G-CAP/G-DEC/G-DRIFT). NO_TIER_II_CLAIM.'
    )
    exp_armb_parser.add_argument('--project', help='Project name to use')
    exp_armb_parser.add_argument('--seeds', help='Comma-separated seeds, e.g. 0,1,2')
    exp_armb_parser.add_argument('--quick', action='store_true',
                                 help='Quick mode (d=4, 1 seed, tiny stream)')
    exp_armb_parser.set_defaults(func=cmd_exp_capture_armb)
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

    # exp-cat-test (C2W5, tier ii: the organization dividend)
    exp_ct_parser = subparsers.add_parser(
        'exp-cat-test',
        help='Run the cat test: the factored store scored on unseen combinations'
    )
    exp_ct_parser.add_argument('--project', help='Project name to use')
    exp_ct_parser.add_argument('--seed', type=int, default=0, help='Base seed')
    exp_ct_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (small family, 3 seeds, short write)')
    exp_ct_parser.add_argument(
        '--stages', nargs='+',
        choices=['family', 'calibrate', 'controls', 'arm', 'd_sweep', 'deletion'],
        help='Stages to run (default: all, in the pre-registered order)')
    exp_ct_parser.add_argument('--out-dir', help='Output directory for artifacts')
    exp_ct_parser.set_defaults(func=cmd_exp_cat_test)

    # ==================================================================
    # C2W11 (THE COMPOSITIONAL WAVE) -- all THREE subcommands are landed
    # together, by spoke A, in one commit. This is the wave's declared
    # conflict-elimination measure: `experiment_cmd.py` is a shared file and
    # three concurrent spokes would otherwise collide in it. Spokes B and C
    # own their MODULES, never this file.
    # ==================================================================
    # exp-c2w11-substrate (spoke A): the repaired substrate + K0-K8 + M1-M6
    exp_c11a_parser = subparsers.add_parser(
        'exp-c2w11-substrate',
        help='C2W11 spoke A: the repaired substrate (placing write, re-selected '
             'co-scaled widths, feature-factored launches) and every '
             'kill-condition K0-K8, run FIRST'
    )
    exp_c11a_parser.add_argument('--project', help='Project name to use')
    exp_c11a_parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2],
                                 help='Claim seeds (selection uses 100/101/102)')
    exp_c11a_parser.add_argument('--quick', action='store_true',
                                 help='Quick mode (small family, short settles)')
    exp_c11a_parser.add_argument(
        '--stages', nargs='+',
        choices=['k0', 'm6', 'width', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6',
                 'k7cap', 'k8', 'm4', 'm5', 'coverage', 'freeze'],
        help='Stages to run (default: all, in the PRE-REGISTERED order '
             'k0 -> m6 -> width -> k7cap/k6 -> k1 -> k2 -> k3 -> k4 -> k5 -> k8)')
    exp_c11a_parser.add_argument('--out-dir', help='Output directory for artifacts')
    exp_c11a_parser.set_defaults(func=cmd_exp_c2w11_substrate)

    # exp-c2w11-organizer (spoke B): the physics organizer + psi + the read
    exp_c11b_parser = subparsers.add_parser(
        'exp-c2w11-organizer',
        help='C2W11 spoke B: the physics organizer, the DeepSets psi read and '
             'the V-leg scores (GATED on spoke A freezing the interfaces)'
    )
    exp_c11b_parser.add_argument('--project', help='Project name to use')
    exp_c11b_parser.add_argument('--seeds', type=int, nargs='+',
                                 default=[0, 1, 2, 3, 4], help='Score seeds')
    exp_c11b_parser.add_argument('--quick', action='store_true', help='Quick mode')
    exp_c11b_parser.add_argument('--stages', nargs='+', help='Stages to run')
    exp_c11b_parser.add_argument('--out-dir', help='Output directory for artifacts')
    exp_c11b_parser.set_defaults(func=cmd_exp_c2w11_organizer)

    # exp-c2w11-nulls (spoke C): the matched-capacity organizer swap
    exp_c11c_parser = subparsers.add_parser(
        'exp-c2w11-nulls',
        help='C2W11 spoke C: the matched-capacity non-physics organizers on the '
             'frozen interfaces (GATED on spoke A freezing the interfaces)'
    )
    exp_c11c_parser.add_argument('--project', help='Project name to use')
    exp_c11c_parser.add_argument('--seeds', type=int, nargs='+',
                                 default=[0, 1, 2, 3, 4], help='Score seeds')
    exp_c11c_parser.add_argument('--quick', action='store_true', help='Quick mode')
    exp_c11c_parser.add_argument('--arms', nargs='+', help='Subset of null arms')
    exp_c11c_parser.add_argument('--stages', nargs='+', help='Stages to run')
    exp_c11c_parser.add_argument('--out-dir', help='Output directory for artifacts')
    exp_c11c_parser.set_defaults(func=cmd_exp_c2w11_nulls)

    # exp-null-arms (C2W5, the matched-capacity organizer audit: N1-N5)
    exp_na_parser = subparsers.add_parser(
        'exp-null-arms',
        help='Run the N1-N5 matched-capacity organizer arms against the cat test'
    )
    exp_na_parser.add_argument('--project', help='Project name to use')
    exp_na_parser.add_argument('--seed', type=int, default=0, help='Base seed')
    exp_na_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (tiny family, 1 tune seed, 2 score '
                                    'seeds, short grids)')
    exp_na_parser.add_argument(
        '--stages', nargs='+',
        choices=['grid', 'score', 'gridmax', 'mechanism', 'reader_audit',
                 'ceiling', 'oracle'],
        help='Stages to run (default: all, in the registered order; '
             '"reader_audit" re-scores the banked cells through the '
             'zero-parameter identity readers and is opt-in)')
    exp_na_parser.add_argument('--arms', nargs='+',
                               choices=['N1', 'N2', 'N3', 'N4', 'N5'],
                               help='Subset of arms (default: all five)')
    exp_na_parser.add_argument('--out-dir', help='Output directory for artifacts')
    exp_na_parser.set_defaults(func=cmd_exp_null_arms)

    # exp-tierii-read (C2W5/6, the read-protocol iteration: charter §A20.3)
    exp_tr_parser = subparsers.add_parser(
        'exp-tierii-read',
        help='Run the tier-ii multi-well read protocol + the organizer swap'
    )
    exp_tr_parser.add_argument('--seeds', type=int, nargs='+',
                               default=[0, 1, 2, 3, 4], help='Score seeds')
    exp_tr_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (tiny family, 1 seed, short settles)')
    exp_tr_parser.add_argument(
        '--stages', nargs='+',
        choices=['k0', 'arms', 'guards', 'consolidate', 'levers'],
        help='Stages to run (default: all; k0 is the blocking pre-condition)')
    exp_tr_parser.add_argument('--organize-steps', type=int, default=60,
                               help='Physics-organizer steps (through the settle)')
    exp_tr_parser.add_argument('--k-particles', type=int, default=12,
                               help='k (CAPACITY: ledgered, matched on every arm)')
    exp_tr_parser.add_argument('--out-dir', help='Output directory for artifacts')
    exp_tr_parser.set_defaults(func=cmd_exp_tierii_read)

    # exp-tierii-card (C2W7, the CARDINALITY iteration: charter §A21's C2W7 row)
    exp_tc_parser = subparsers.add_parser(
        'exp-tierii-card',
        help='Run the tier-ii multiplicity read (counting code + F-commitment)'
    )
    exp_tc_parser.add_argument('--seeds', type=int, nargs='+',
                               default=[0, 1, 2, 3, 4], help='Score seeds')
    exp_tc_parser.add_argument('--quick', action='store_true',
                               help='Quick mode (tiny family, 1 seed, short settles)')
    exp_tc_parser.add_argument(
        '--stages', nargs='+',
        choices=['k0', 'arms', 'guards', 'regularizer', 'levers', 'swap'],
        help='Stages (default: all; k0 is the blocking pre-condition, swap is gated)')
    exp_tc_parser.add_argument('--organize-steps', type=int, default=60,
                               help='Physics-organizer steps (through the settle)')
    exp_tc_parser.add_argument('--k-particles', type=int, default=12,
                               help='k (CAPACITY: ledgered, matched on every arm)')
    exp_tc_parser.add_argument('--lam-on', type=float, default=1.0,
                               help='Anti-collapse coefficient of the ON arm')
    exp_tc_parser.add_argument('--force-swap', action='store_true',
                               help='Run the swap even if the gate fails '
                                    '(LABELLED diagnostic, never a claim cell)')
    exp_tc_parser.add_argument('--out-dir', help='Output directory for artifacts')
    exp_tc_parser.set_defaults(func=cmd_exp_tierii_card)

    # exp-psi-residual (C2W5, the payload residual read-out lever: charter §A20.3(a))
    exp_pr_parser = subparsers.add_parser(
        'exp-psi-residual',
        help='Run the psi payload-residual ledger / trained tier (run-2 config evidence)'
    )
    exp_pr_parser.add_argument('--tier', choices=['ledger', 'trained'], default='ledger',
                               help='ledger (untrained per-stage spread) | trained')
    exp_pr_parser.add_argument('--cells', nargs='+',
                               help='Subset of the registered cells (default: all four)')
    exp_pr_parser.add_argument('--seeds', type=int, nargs='+',
                               help='Paired seeds (default: the module default)')
    exp_pr_parser.add_argument('--steps', type=int,
                               help='Outer training steps (trained tier only)')
    exp_pr_parser.add_argument('--eval-batches', type=int, default=4,
                               help='Eval batches per record')
    exp_pr_parser.add_argument('--out-dir', help='Output directory for artifacts')
    exp_pr_parser.add_argument('--tag', help='Artifact filename tag (default: the tier)')
    exp_pr_parser.set_defaults(func=cmd_exp_psi_residual)

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

    # exp-clu-system  (C2W1: the full-CLU synthesis harness)
    exp_cs_parser = subparsers.add_parser(
        'exp-clu-system',
        help='The FULL CLU with every lever live, staged, with all 13 '
             'anti-collapse monitors as loud runtime guards (C2W1)'
    )
    exp_cs_parser.add_argument('--project', help='Project name to use')
    exp_cs_parser.add_argument('--seed', type=int, help='Random seed')
    exp_cs_parser.add_argument('--quick', action='store_true',
                               help='Quick smoke mode (plumbing only, not a result)')
    exp_cs_parser.add_argument('--stages', nargs='+',
                               help='Run only these stages (default: all)')
    exp_cs_parser.add_argument('--offer', type=int,
                               help='How many items the write stream offers')
    exp_cs_parser.add_argument('--no-remediate', action='store_true',
                               help='Skip the remediation arms (restoring-verb table)')
    exp_cs_parser.set_defaults(func=cmd_exp_clu_system)

    # exp-memory-gym  (C2W1: Track 1, the dividend as the sole KPI)
    exp_gym_parser = subparsers.add_parser(
        'exp-memory-gym',
        help='The internal memory gym: four task families (one per structural '
             'opening), launder-native, dividend + byte ledger on every cell (C2W1)'
    )
    exp_gym_parser.add_argument('--project', help='Project name to use')
    exp_gym_parser.add_argument('--seed', type=int, help='Base seed offset')
    exp_gym_parser.add_argument('--quick', action='store_true',
                               help='Quick smoke mode (plumbing only, not a result)')
    exp_gym_parser.add_argument('--families', nargs='+',
                               help='Run only these families '
                                    '(overload|aggregate|recency|manifold)')
    exp_gym_parser.add_argument('--arms', nargs='+',
                               help='Run only these arms (base|tight|ridge|refN|...)')
    exp_gym_parser.add_argument('--seeds', nargs='+', type=int,
                               help='Override the per-cell seed list')
    exp_gym_parser.set_defaults(func=cmd_exp_memory_gym)

    # exp-traj-write  (C2W2 Route 1: the trajectory/path write objective)
    exp_tw_parser = subparsers.add_parser(
        'exp-traj-write',
        help='C2W2 Route 1: ask the WRITE to put information in the trajectory '
             '(lambda_traj / lambda_path, both default 0) and score the frozen '
             'race card'
    )
    exp_tw_parser.add_argument('--project', help='Project name to use')
    exp_tw_parser.add_argument('--seed', type=int, help='Base seed offset')
    exp_tw_parser.add_argument('--quick', action='store_true',
                               help='Quick smoke mode (plumbing only, not a result)')
    exp_tw_parser.add_argument('--families', nargs='+',
                               help='Run only these families '
                                    '(overload|aggregate|manifold)')
    exp_tw_parser.add_argument('--arms', nargs='+',
                               help='Run only these race arms (endpoint_write|'
                                    'traj_write|path_write|traj+path)')
    exp_tw_parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                               help='Seeds per cell (the gate needs {0,1,2})')
    exp_tw_parser.add_argument('--coeffs', nargs='+', type=float,
                               help='Coefficient grid (default: the PRE-REGISTERED '
                                    '0.03 0.3 3.0 30.0)')
    exp_tw_parser.set_defaults(func=cmd_exp_traj_write)

    # exp-route3-attribution  (C2W3 Route 3 stage 1 + C2W4 the C6 third-party probe)
    exp_r3_parser = subparsers.add_parser(
        'exp-route3-attribution',
        help='Route 3 stage 1: the per-slot store-attribution curve (the §A9.4 '
             'unlock bar + the §A9.5 per-slot table launder), and --part '
             'thirdparty for the C6 third-party attribution probe (delete a '
             'NON-selected item; a per-slot table gives exactly 0 by construction)'
    )
    exp_r3_parser.add_argument('--project', help='Project name to use')
    exp_r3_parser.add_argument('--seed', type=int, help='Base seed offset')
    exp_r3_parser.add_argument('--quick', action='store_true',
                               help='Quick smoke mode (plumbing only, not a result)')
    exp_r3_parser.add_argument('--part', choices=['curve', 'thirdparty'],
                               default='curve',
                               help='curve = the stage-1 attribution curve; '
                                    'thirdparty = the C6 probe across d/s')
    exp_r3_parser.add_argument('--families', nargs='+',
                               help='Run only these families '
                                    '(overload|aggregate|manifold); curve part only')
    exp_r3_parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2],
                               help='Seeds per cell (the bar needs {0,1,2})')
    exp_r3_parser.add_argument('--radii', nargs='+', type=float,
                               help='ball_radius sweep for --part thirdparty '
                                    '(default: the PRE-REGISTERED d/s grid)')
    exp_r3_parser.add_argument('--no-escalate', action='store_true',
                               help='Skip the one bounded write-budget escalation')
    exp_r3_parser.set_defaults(func=cmd_exp_route3_attribution)

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


def cmd_exp_cat_test(args):
    """Run the C2W5 cat test (tier ii's vehicle: the factored store)."""
    console.print("[bold cyan]Running the cat test "
                  "(factored store, unseen combinations)[/bold cyan]")

    from ..experiments.exp_cat_test import run_cat_test
    stages = getattr(args, 'stages', None)
    try:
        kw = {}
        if stages:
            kw['stages'] = tuple(stages)
        run_cat_test(
            project=getattr(args, 'project', None),
            seed=int(getattr(args, 'seed', 0) or 0),
            quick=bool(getattr(args, 'quick', False)),
            out_dir=getattr(args, 'out_dir', None),
            **kw,
        )
        console.print("✓ cat test completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    return 0


def cmd_exp_c2w11_substrate(args):
    """C2W11 spoke A: the repaired substrate and every kill-condition."""
    console.print("[bold cyan]Running C2W11 spoke A "
                  "(repaired substrate + K0-K8, kills FIRST)[/bold cyan]")

    from ..experiments.exp_c2w11_substrate import run_c2w11_substrate
    try:
        kw = {}
        if getattr(args, 'stages', None):
            kw['stages'] = tuple(args.stages)
        run_c2w11_substrate(
            project=getattr(args, 'project', None),
            seeds=tuple(getattr(args, 'seeds', (0, 1, 2)) or (0, 1, 2)),
            quick=bool(getattr(args, 'quick', False)),
            out_dir=getattr(args, 'out_dir', None),
            **kw,
        )
        console.print("✓ C2W11 substrate + kill-conditions completed",
                      style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    return 0


def _c2w11_gated(name: str, module: str, fn: str):
    """The two GATED C2W11 spokes share one stub body.

    ⛔ Spoke A lands these subcommands so that spokes B and C never touch this
    shared file. Until their modules exist the stub says so **by name** rather
    than dying in an import traceback — a missing spoke is a declared NOT-RUN,
    not a crash.
    """
    def run(args):
        console.print(f"[bold cyan]Running C2W11 {name}[/bold cyan]")
        try:
            mod = __import__(f"chlu.experiments.{module}", fromlist=[fn])
        except ImportError:
            console.print(
                f"✗ C2W11 {name} is NOT LANDED: chlu/experiments/{module}.py "
                "does not exist yet. The subcommand is registered by spoke A "
                "(the wave's conflict-elimination measure); its owner spoke "
                "supplies the module.", style="bold yellow")
            return 1
        kw = {}
        for opt in ('stages', 'arms'):
            if getattr(args, opt, None):
                kw[opt] = tuple(getattr(args, opt))
        try:
            getattr(mod, fn)(
                project=getattr(args, 'project', None),
                seeds=tuple(getattr(args, 'seeds', ()) or ()),
                quick=bool(getattr(args, 'quick', False)),
                out_dir=getattr(args, 'out_dir', None),
                **kw,
            )
        except Exception as e:  # pragma: no cover - owner spoke's territory
            console.print(f"✗ Error: {e}", style="bold red")
            return 1
        console.print(f"✓ C2W11 {name} completed", style="bold green")
        return 0

    return run


cmd_exp_c2w11_organizer = _c2w11_gated(
    "spoke B (physics organizer + psi)", "exp_c2w11_organizer",
    "run_c2w11_organizer")
cmd_exp_c2w11_nulls = _c2w11_gated(
    "spoke C (matched-capacity organizer swap)", "exp_c2w11_nulls",
    "run_c2w11_nulls")


def cmd_exp_null_arms(args):
    """Run the C2W5 matched-capacity organizer audit (null arms N1-N5)."""
    console.print("[bold cyan]Running the null-arm audit "
                  "(N1-N5, matched capacity, frozen phi)[/bold cyan]")

    from ..experiments.exp_null_arms import run_null_arms
    try:
        kw = {}
        if getattr(args, 'stages', None):
            kw['stages'] = tuple(args.stages)
        if getattr(args, 'arms', None):
            kw['arms'] = tuple(args.arms)
        run_null_arms(
            project=getattr(args, 'project', None),
            seed=int(getattr(args, 'seed', 0) or 0),
            quick=bool(getattr(args, 'quick', False)),
            out_dir=getattr(args, 'out_dir', None),
            **kw,
        )
        console.print("✓ null-arm audit completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    return 0


def cmd_exp_tierii_read(args):
    """Run the tier-ii read-fix iteration (multi-well read + organizer swap)."""
    console.print("[bold cyan]Running the tier-ii multi-well read protocol "
                  "(k-particle launch head, organizer swap)[/bold cyan]")

    from ..experiments.exp_tierii_read import run_tierii_read
    try:
        kw = {}
        if getattr(args, 'stages', None):
            kw['stages'] = tuple(args.stages)
        run_tierii_read(
            seeds=tuple(getattr(args, 'seeds', (0, 1, 2, 3, 4))),
            quick=bool(getattr(args, 'quick', False)),
            organize_steps=int(getattr(args, 'organize_steps', 60) or 60),
            k_particles=int(getattr(args, 'k_particles', 12) or 12),
            out_dir=getattr(args, 'out_dir', None),
            **kw,
        )
        console.print("✓ tier-ii read-fix completed", style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    return 0


def cmd_exp_tierii_card(args):
    """Run the tier-ii CARDINALITY iteration (multiplicity counting code)."""
    console.print("[bold cyan]Running the tier-ii multiplicity read "
                  "(counting code, F-commitment, launch-collapse monitor)"
                  "[/bold cyan]")

    from ..experiments.exp_tierii_cardinality import run_tierii_cardinality
    try:
        kw = {}
        if getattr(args, 'stages', None):
            kw['stages'] = tuple(args.stages)
        run_tierii_cardinality(
            seeds=tuple(getattr(args, 'seeds', (0, 1, 2, 3, 4))),
            quick=bool(getattr(args, 'quick', False)),
            organize_steps=int(getattr(args, 'organize_steps', 60) or 60),
            k_particles=int(getattr(args, 'k_particles', 12) or 12),
            lam_on=float(getattr(args, 'lam_on', 1.0)),
            force_swap=bool(getattr(args, 'force_swap', False)),
            out_dir=getattr(args, 'out_dir', None),
            **kw,
        )
        console.print("✓ tier-ii cardinality iteration completed",
                      style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1
    return 0


def cmd_exp_psi_residual(args):
    """Run the psi payload-residual tier (`exp_psi_residual`'s own argv contract)."""
    console.print("[bold cyan]Running the psi payload-residual ledger "
                  "(payload-only read-out residual)[/bold cyan]")

    from ..experiments.exp_psi_residual import main as psi_residual_main
    argv = ['--tier', str(getattr(args, 'tier', 'ledger') or 'ledger'),
            '--eval-batches', str(int(getattr(args, 'eval_batches', 4) or 4))]
    if getattr(args, 'cells', None):
        argv += ['--cells', *[str(c) for c in args.cells]]
    if getattr(args, 'seeds', None):
        argv += ['--seeds', *[str(int(s)) for s in args.seeds]]
    if getattr(args, 'steps', None):
        argv += ['--steps', str(int(args.steps))]
    if getattr(args, 'out_dir', None):
        argv += ['--out', str(args.out_dir)]
    if getattr(args, 'tag', None):
        argv += ['--tag', str(args.tag)]
    try:
        psi_residual_main(argv)
        console.print("✓ psi payload-residual completed", style="bold green")
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


def cmd_exp_phi_read_in(args):
    """Run the φ read-in benchmark: Hopfield protocol re-fought in φ-space."""
    console.print(
        "[bold cyan]Running Experiment PHI-READ-IN: learned φ around a DESIGNED "
        "store (phase doctrine flagship, w23)[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    if getattr(args, 'quick', False):
        apply_phi_read_in_quick(config)
    if getattr(args, 'dataset', None):
        config.experiment_phi_read_in.datasets = args.dataset.split(',')
    if getattr(args, 'arms', None):
        config.experiment_phi_read_in.phi_arms = args.arms.split(',')

    try:
        res = run_experiment_phi_read_in(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=getattr(args, 'seed', None),
        )
        console.print(
            f"✓ Experiment PHI-READ-IN completed -> {res['metrics_path']}",
            style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_phi_stream(args):
    """Run the φ stream-discipline study: cost of a task-1-only φ in Class-IL."""
    console.print(
        "[bold cyan]Running Experiment PHI-STREAM: task-1-only φ (PRIMARY) vs "
        "generic-frozen φ (declared upper bound) on a class-incremental stream "
        "(w24)[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    if getattr(args, 'quick', False):
        apply_phi_stream_quick(config)
    if getattr(args, 'regimes', None):
        config.experiment_phi_stream.phi_regimes = args.regimes.split(',')
    if getattr(args, 'arms', None):
        config.experiment_phi_stream.phi_arms = args.arms.split(',')

    try:
        res = run_experiment_phi_stream(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=getattr(args, 'seed', None),
        )
        console.print(
            f"✓ Experiment PHI-STREAM completed -> {res['metrics_path']}",
            style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_cl_entry(args):
    """Run the w25 continual-learning entry (Class-IL + retry + retention)."""
    console.print(
        "[bold cyan]Running Experiment CL-ENTRY: rehearsal-free Class-IL with a "
        "designed CLU store (w25) — entry + R3-native retry + scheduled per-item "
        "retention[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    if getattr(args, 'quick', False):
        apply_cl_entry_quick(config)
    if getattr(args, 'dataset', None):
        config.experiment_cl_entry.dataset = args.dataset
        if args.dataset == 'cifar10':
            apply_cl_entry_cifar10(config)
    if getattr(args, 'baselines', None):
        config.experiment_cl_entry.baselines = args.baselines.split(',')
    if getattr(args, 'budgets', None):
        config.experiment_cl_entry.frontier_budgets_floats = [
            int(b) for b in args.budgets.split(',')
        ]

    try:
        res = run_experiment_cl_entry(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=getattr(args, 'seed', None),
            items=(args.items.split(',') if getattr(args, 'items', None) else None),
        )
        console.print(
            f"✓ Experiment CL-ENTRY completed -> {res['metrics_path']}",
            style="bold green")
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_retry_compute(args):
    """Run the retry-compute study: accuracy-vs-compute curve, CLU-gated + 5 controls."""
    console.print(
        "[bold cyan]Running Experiment RETRY-COMPUTE: accuracy-vs-compute curve "
        "(CLU-gated retry + 5 controls, w23)[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    if getattr(args, 'quick', False):
        apply_retry_compute_quick(config)
    if getattr(args, 'ambiguity', False):
        apply_retry_compute_ambiguity(config)
    if getattr(args, 'dataset', None):
        config.experiment_retry_compute.datasets = args.dataset.split(',')

    try:
        if getattr(args, 'headroom', False):
            res = run_retry_compute_headroom_gate(
                config=config,
                save_dir=str(paths['plots']),
                seed=getattr(args, 'seed', None),
            )
            console.print(
                f"✓ Headroom gate completed ({res['n_passed']} cell(s) passed) "
                f"-> {res['metrics_path']}", style="bold green")
            return 0
        res = run_experiment_retry_compute(
            config=config,
            save_dir=str(paths['plots']),
            seed=getattr(args, 'seed', None),
        )
        console.print(
            f"✓ Experiment RETRY-COMPUTE completed -> {res['metrics_path']}",
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


def cmd_exp_sharded_store(args):
    """Run the w25 N-unit sharded store + the 2x2 additivity discriminator."""
    console.print(
        "[bold cyan]Running Experiment SHARDED-STORE: is the K~=32 write ceiling "
        "per-dig?[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    if getattr(args, 'quick', False):
        apply_sharded_store_quick(config)
    if getattr(args, 'cells', None):
        config.experiment_sharded_store.cells = list(args.cells)
    if getattr(args, 'arms', None):
        config.experiment_sharded_store.arms = list(args.arms)
    if getattr(args, 'items', None):
        ss = config.experiment_sharded_store
        want = set(args.items)
        ss.run_discriminator = '1' in want
        ss.run_read_parity = '2' in want
        ss.run_timing = '3' in want
        ss.run_allocator = '4' in want
        ss.run_init_ablation = '5' in want
        ss.run_deadband_sweep = '6' in want

    try:
        res = run_experiment_sharded_store(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=getattr(args, 'seed', None),
        )
        verdict = res.get('item1_verdict', {}).get('verdict', 'items subset')
        console.print(
            f"\u2713 Experiment SHARDED-STORE completed ({verdict}) -> "
            f"{res['metrics_path']}",
            style="bold green",
        )
    except Exception as e:
        console.print(f"\u2717 Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_well_lifecycle(args):
    """Run the C2W8 stage-1 well census (the mechanical stage-2 gate)."""
    console.print(
        "[bold cyan]Running Experiment WELL-LIFECYCLE (stage 1): the census — "
        "is there anything to prune, anything to merge?[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])
    seeds = ([int(s) for s in str(args.seeds).split(',')]
             if getattr(args, 'seeds', None) else None)

    try:
        res = run_experiment_well_lifecycle(
            config=config,
            save_dir=str(paths['plots']),
            seeds=seeds,
            quick=getattr(args, 'quick', False),
        )
        k1 = res['k1']
        console.print(
            f"✓ Experiment WELL-LIFECYCLE completed: P_mean={k1['P_mean']:.4f} "
            f"M_mean={k1['M_mean']:.4f} -> stage2_unlock={res['stage2_unlock']} "
            f"-> {res['census_json']}",
            style="bold green",
        )
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


# --- BEGIN c2w8p3-phi-geometry (additive) ---
def cmd_exp_phi_geometry(args):
    """C2W8 pass 3: the φ→addr map and the address geometry it buys."""
    console.print(
        "[bold cyan]Running Experiment PHI-GEOMETRY (C2W8 pass 3): the "
        "phi_dim->addr_dim map, then does strong phi actually separate?[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])
    if getattr(args, 'dims', None):
        config.experiment_phi_geometry.addr_dims = [
            int(x) for x in str(args.dims).split(',')
        ]
    seeds = ([int(s) for s in str(args.seeds).split(',')]
             if getattr(args, 'seeds', None) else None)

    try:
        res = run_experiment_phi_geometry(
            config=config,
            save_dir=str(paths['plots']),
            seeds=seeds,
            quick=getattr(args, 'quick', False),
        )
        console.print(
            f"✓ Experiment PHI-GEOMETRY completed: geometry_go={res['geometry_go']} "
            f"d_favoured={res['d_favoured_by_geometry']} -> {res['json_path']}",
            style="bold green",
        )
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0
# --- END c2w8p3-phi-geometry ---


# --- BEGIN c2w8p3-capture-strong-phi (additive) ---
def cmd_exp_capture_strong_phi(args):
    """C2W8 pass 3, THE SPINE: the completed gate at strong φ."""
    console.print(
        "[bold cyan]Running Experiment CAPTURE-STRONG-PHI (C2W8 pass 3, the "
        "SPINE): frozen census + the COMPLETED gate (G-CAP/G-DEC/G-DRIFT/G-ADDR) "
        "at strong phi on Split-CIFAR-10[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])
    if getattr(args, 'addr_dim', None):
        config.experiment_capture_strong_phi.addr_dim = int(args.addr_dim)
    seeds = ([int(s) for s in str(args.seeds).split(',')]
             if getattr(args, 'seeds', None) else None)
    arms = ([s.strip() for s in str(args.arms).split(',')]
            if getattr(args, 'arms', None) else None)

    try:
        res = run_experiment_capture_strong_phi(
            config=config,
            save_dir=str(paths['plots']),
            seeds=seeds,
            arms=arms,
            quick=getattr(args, 'quick', False),
        )
        console.print(
            f"✓ Experiment CAPTURE-STRONG-PHI completed: "
            f"branch={res['branch_by_arm']} gate={res['gate_pass_by_arm']} "
            f"-> {res['json_path']}",
            style="bold green",
        )
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0
# --- END c2w8p3-capture-strong-phi ---


def cmd_exp_capture_armb(args):
    """Run C2W8 pass 2 ARM B — the emission head, on pass 1's frozen census."""
    console.print(
        "[bold cyan]Running C2W8 pass 2 ARM B (emission head): a forward pass "
        "instead of 300 gradient steps[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])
    seeds = ([int(s) for s in str(args.seeds).split(',')]
             if getattr(args, 'seeds', None) else None)

    try:
        res = run_experiment_capture_armb(
            config=config,
            save_dir=str(paths['plots']),
            seeds=seeds,
            quick=getattr(args, 'quick', False),
        )
        g = res['gate']
        console.print(
            f"\u2713 ARM B completed: G-CAP {g['G_CAP_all_seeds']} \u00b7 "
            f"G-DEC {g['G_DEC_all_seeds']} \u00b7 G-DRIFT {g['G_DRIFT_all_seeds']} "
            f"-> all three = {g['ALL_THREE_ALL_SEEDS']} "
            f"[{res['tier_ii_status']}] -> {res['census_json']}",
            style="bold green",
        )
    except Exception as e:
        console.print(f"\u2717 Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_write_ceiling(args):
    """Run the w24 write-ceiling-break arms (locality / scale / crowding)."""
    console.print(
        "[bold cyan]Running Experiment WRITE-CEILING-BREAK: can any write break "
        "K_ceiling~=32?[/bold cyan]"
    )

    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1

    config.project.save_dir = str(paths['plots'])

    if getattr(args, 'quick', False):
        apply_write_ceiling_quick(config)
    if getattr(args, 'arms', None):
        config.experiment_write_ceiling.arms = list(args.arms)
    if getattr(args, 'dims', None):
        wc = config.experiment_write_ceiling
        keep = [(d, k) for d, k in zip(wc.dims, wc.k_start, strict=False)
                if d in args.dims]
        wc.dims = [d for d, _ in keep] or list(args.dims)
        wc.k_start = [k for _, k in keep] or [min(wc.k_ladder)] * len(wc.dims)

    try:
        res = run_experiment_write_ceiling(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=getattr(args, 'seed', None),
        )
        console.print(
            f"✓ Experiment WRITE-CEILING-BREAK completed "
            f"({res['item4_verdict']['verdict']}) -> {res['metrics_path']}",
            style="bold green",
        )
    except Exception as e:
        console.print(f"✗ Error: {e}", style="bold red")
        return 1

    return 0


def cmd_exp_clu_system(args):
    """Run the C2W1 full-CLU harness (staged lever activation + 13 monitors)."""
    console.print(
        "[bold cyan]Running Experiment CLU-SYSTEM: the full CLU, every lever live, "
        "staged; acceptance is 'does not collapse', not 'wins'[/bold cyan]"
    )
    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1
    config.project.save_dir = str(paths['plots'])
    try:
        from ..experiments.exp_clu_system import run_experiment_clu_system

        res = run_experiment_clu_system(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=getattr(args, 'seed', None),
            stages=getattr(args, 'stages', None),
            quick=getattr(args, 'quick', False),
            n_offer=getattr(args, 'offer', None),
            remediate=not getattr(args, 'no_remediate', False),
        )
        tripped = sorted(n for n, row in res['trip_table'].items() if row['ever_tripped'])
        console.print(f"✓ Experiment CLU-SYSTEM completed -> {res['metrics_path']}")
        console.print(f"  monitors that tripped at least once: {tripped or 'none'}")
        return 0
    except Exception as e:  # pragma: no cover - CLI plumbing
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return 1


def cmd_exp_memory_gym(args):
    """Run the C2W1 memory gym (four families, launder-native, dividend KPI)."""
    console.print(
        "[bold cyan]Running Experiment MEMORY-GYM: Track 1 development currency; "
        "the dividend is the sole KPI and <=0 at v0 is the honest start[/bold cyan]"
    )
    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1
    config.project.save_dir = str(paths['plots'])
    try:
        from ..experiments.exp_memory_gym import run_experiment_memory_gym

        res = run_experiment_memory_gym(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=getattr(args, 'seed', None),
            families=getattr(args, 'families', None),
            arms=getattr(args, 'arms', None),
            seeds=getattr(args, 'seeds', None),
            quick=getattr(args, 'quick', False),
        )
        console.print(f"✓ Experiment MEMORY-GYM completed -> {res['metrics_path']}")
        for key, row in res['aggregate'].items():
            console.print(
                f"  {key}: dividend {row['dividend']['mean']:+.4f} "
                f"± {row['dividend']['se']:.4f} ({row['sign']}, "
                f"n={row['dividend']['n']}) | bytes "
                f"{row['byte_ratio']['mean']:.1f}x matched={row['matched_bytes']}"
            )
        untested = sorted(n for n, r in res['monitor_table'].items() if r['untested'])
        console.print(f"  monitors still UNTESTED (never fired): {untested or 'none'}")
        return 0
    except Exception as e:  # pragma: no cover - CLI plumbing
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return 1


def cmd_exp_traj_write(args):
    """Run the C2W2 Route-1 race card (the trajectory/path write objective)."""
    console.print(
        "[bold cyan]Running Experiment TRAJ-WRITE (C2W2 Route 1): the write is "
        "asked to put information in the trajectory; the dividend is still the "
        "sole KPI and the gate's verdict is the Hub's[/bold cyan]"
    )
    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1
    config.project.save_dir = str(paths['plots'])
    try:
        from ..experiments.exp_traj_write import (
            COEFF_GRID,
            run_experiment_traj_write,
        )

        res = run_experiment_traj_write(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=getattr(args, 'seed', None),
            families=getattr(args, 'families', None),
            arms=getattr(args, 'arms', None),
            seeds=tuple(getattr(args, 'seeds', None) or (0, 1, 2)),
            coeffs=tuple(getattr(args, 'coeffs', None) or COEFF_GRID),
            quick=getattr(args, 'quick', False),
        )
        console.print(f"✓ Experiment TRAJ-WRITE completed ({res['n_cells']} cells)")
        for fam, row in res['coverage_per_family'].items():
            console.print(
                f"  admissible-cell coverage [{fam}]: "
                f"{row['n_admissible']}/{row['n_cells']} ({row['coverage']:.0%})"
                + (f"  reasons: {row['reasons']}" if row['reasons'] else "")
            )
        g = res['gate']
        console.print(f"  cleared 2 SE: {g['cleared_two_se'] or 'none'}")
        console.print(f"  <=0 votes:    {g['le_zero_votes'] or 'none'}")
        console.print(f"  abstained:    {g['abstained'] or 'none'}")
        console.print(f"  under-powered grids: {g['under_powered_grids'] or 'none'}")
        return 0
    except Exception as e:  # pragma: no cover - CLI plumbing
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return 1


def cmd_exp_route3_attribution(args):
    """Route 3 stage 1's attribution curve, or the C6 third-party probe.

    ⛔ Measurement only: no dividend is claimed here and §A9.5's kill (the
    per-slot table reproduces the slotted read) **stands**. ``--part thirdparty``
    measures the one coupling a per-slot table gives exactly 0 for by
    construction — audit-paper protocol evidence (§A14.1), not a revival.
    """
    part = getattr(args, 'part', 'curve')
    console.print(
        "[bold cyan]Running Experiment ROUTE3-ATTRIBUTION"
        + (": the C6 THIRD-PARTY probe (delete a NON-selected item; the "
           "per-slot table's Delta is exactly 0 by construction)"
           if part == 'thirdparty' else
           ": the per-slot store-attribution curve + the §A9.4 bar + the "
           "§A9.5 per-slot table launder")
        + "[/bold cyan]"
    )
    config, paths = _get_config_and_paths(args)
    if config is None:
        return 1
    config.project.save_dir = str(paths['plots'])
    try:
        from ..experiments.exp_route3_attribution import (
            BALL_RADIUS_GRID,
            run_experiment_route3_attribution,
            run_experiment_thirdparty,
        )

        if part == 'thirdparty':
            res = run_experiment_thirdparty(
                config=config,
                save_dir=str(paths['plots']),
                models_dir=str(paths['models']),
                seed=getattr(args, 'seed', None),
                seeds=tuple(getattr(args, 'seeds', None) or (0, 1, 2)),
                radii=tuple(getattr(args, 'radii', None) or BALL_RADIUS_GRID),
                quick=getattr(args, 'quick', False),
            )
            console.print("✓ C6 third-party probe completed "
                          f"({len(res['cells'])} cells)")
            for row in res['per_radius']:
                console.print(
                    f"  R={row['ball_radius']:<5} coverage "
                    f"{row['n_admissible']}/{row['n_cells']}  "
                    f"d/s(atom_width)={row['d_over_s_proxy']:.2f} "
                    f"d/s(fitted)={row['d_over_s_fitted']:.2f}  "
                    f"grad_ratio={row['grad_ratio']:.3e}  "
                    f"coupling_q(slot0)={row['coupling_slot0_q']:.3e}"
                )
            console.print("  ⛔ the per-slot table's third-party Delta: "
                          "0 by construction (Prop T5.4)")
            return 0

        res = run_experiment_route3_attribution(
            config=config,
            save_dir=str(paths['plots']),
            models_dir=str(paths['models']),
            seed=getattr(args, 'seed', None),
            families=getattr(args, 'families', None),
            seeds=tuple(getattr(args, 'seeds', None) or (0, 1, 2)),
            quick=getattr(args, 'quick', False),
            escalate=not getattr(args, 'no_escalate', False),
        )
        console.print("✓ Experiment ROUTE3-ATTRIBUTION completed")
        for fam, row in res['coverage_per_family'].items():
            console.print(
                f"  admissible-cell coverage [{fam}]: "
                f"{row['n_admissible']}/{row['n_cells']} ({row['coverage']:.0%})"
                f"  {row['verdict']}"
            )
        console.print(f"  unlock = {res['unlock']}")
        console.print(f"  §A9.5 fires on clearing slots = "
                      f"{res['a95_fires_on_clearing_slots']}")
        console.print(f"  {res['stage2_verdict']}")
        return 0
    except Exception as e:  # pragma: no cover - CLI plumbing
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return 1


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
