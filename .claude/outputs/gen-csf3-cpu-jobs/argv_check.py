"""Parse the jobscript's generated argv with each experiment module's OWN parser,
extracted by AST so nothing imports jax."""
import argparse, ast, pathlib, subprocess, os, sys

REPO = pathlib.Path("/Users/user/Desktop/CHLU-cpujobs")

def parser_for(mod_rel):
    src = (REPO / mod_rel).read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name == "build_arg_parser")
    # the parsers reference module constants; supply them from the module's AST
    ns = {"argparse": argparse, "PRETEXTS": ("recon", "cls")}  # gen_bridge.PRETEXTS
    for n in tree.body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            try:
                ns[n.targets[0].id] = ast.literal_eval(n.value)
            except Exception:
                pass
    fn.returns = None
    for a in fn.args.args:
        a.annotation = None
    exec(compile(ast.Module([fn], []), "<x>", "exec"), ns)
    return ns["build_arg_parser"]()

def dry_argv(env_over):
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "GEN_LOCAL": "1",
           "DRY": "1", "CHLU_SKIP_ARTEFACT_CHECK": "1"}
    env.update(env_over)
    out = subprocess.run(["bash", "scripts/csf3/job_cpu_gen.sh"], env=env,
                         capture_output=True, text=True, cwd=REPO).stdout
    argvs = []
    for line in out.splitlines():
        line = line.strip()
        for marker in ("[dry]   python3 -u -m ", "[dry] aggregate: python3 -u -m "):
            if line.startswith(marker):
                argvs.append(line[len(marker):].split()[0:1] + line[len(marker):].split()[1:])
    return [(a[0], a[1:]) for a in argvs]

cases = [
    ("chlu/experiments/exp_gen_atom_mean.py",
     {"EXP": "atom_mean", "DATASET": "shapes3d", "PRETEXT": "recon",
      "SEEDS": "0 1 2", "BUDGETS": "b1 b4",
      "EXTRA_ARGS": "--n-val 4096 --tune-n 2048 --allow-knn-proto-move"}),
    ("chlu/experiments/exp_gen_baselines.py",
     {"EXP": "baselines", "DATASET": "dsprites", "PRETEXT": "cls",
      "SEEDS": "0 7", "ARM": "knn", "QUICK": "1", "EXTRA_ARGS": "--no-plots --n-fit 64"}),
]
bad = 0
for mod, env in cases:
    p = parser_for(mod)
    for module_name, argv in dry_argv(env):
        try:
            ns = p.parse_args(argv)
            print(f"OK   {module_name}\n     {' '.join(argv)}\n     -> {vars(ns)}")
        except SystemExit:
            print(f"FAIL {module_name}: {' '.join(argv)}"); bad += 1
    print()
sys.exit(1 if bad else 0)
