# Hub workspace bundle — read this first

This directory **is** `$CHLU_ROOT`. Everything resolves from it; nothing is hardcoded.

```bash
cd <this directory>
source ./env_hub.sh            # sets CHLU_ROOT + every derived CHLU_* root
./scripts/verify_hub_workspace.sh     # re-checksums, OPENS every artefact, writes WORKSPACE-READY.json
```

⛔ **Do not run a hub until `WORKSPACE-READY.json` says `"status": "ready"`.**

## ⛔⛔ JAX HAS NO NATIVE WINDOWS SUPPORT — use WSL2

`jaxlib` publishes **no `win_amd64` wheel**. `pip install jax jaxlib` on native Windows
resolves nothing installable, and a hub imports `chlu` (hence `jax`) locally *even when its
compute is on GCloud*. ⇒ **On the second machine, run inside WSL2 (Ubuntu), not native Windows.**
Zed connects to WSL2 over its remote/SSH project support; the bundle lives on the WSL2 filesystem
(`/home/<user>/CHLU-hub`), ⛔ **not** under `/mnt/c/…` — the 9p bridge makes the 1.9 GB
`features.npy` read pathologically slowly and breaks `mmap_mode="r"` performance.

```bash
wsl --install -d Ubuntu            # PowerShell, admin, once
wsl                                # then, inside Ubuntu:
sudo apt-get update && sudo apt-get install -y python3.11 python3.11-venv git
python3.11 -m venv ~/chlu-venv && source ~/chlu-venv/bin/activate
pip install -e "$CHLU_ROOT"        # or: pip install -r <constraints>; JAX CPU wheels are manylinux
```

## The dataset is NOT in this bundle

It is 255 MiB and re-downloadable. Fetch and **assert the checksum before using it**:

```bash
python -m chlu.cli data shapes3d --download     # or: chlu data shapes3d --download
sha256sum "$CHLU_GEN_DATA_ROOT/shapes3d/3dshapes.h5"
# MUST equal:
# 0a0f6ed98baff276a50f3a081a7434d788da63cb135a98189b2a5b5769be1785
```

⛔ A different sha means a different grid, and every request index in the C set is then wrong.

Most hub work does **not** need the image blobs at all: the feature caches in
`.claude/outputs/gen-feature-bridge/runs/**` are pre-encoded.

## ⭐ THE RETURN PATH — a result that is not synced back does not exist

A hub writes only to `$CHLU_HUB_OUTPUTS/<hub-slug>/**` and `<hub-slug>.md`. To bring that home:

```bash
# on the SECOND machine, when the hub is done:
./scripts/bundle_hub_workspace.sh --return-pack <hub-slug> --return-out ~/<hub-slug>.return.tar.gz
# carry the tar back (scp / drive / whatever), then on the PRIMARY machine:
scripts/bundle_hub_workspace.sh --return-ingest /path/<hub-slug>.return.tar.gz
```

### …and the hub's CODE comes back by `git bundle`

The results tar carries `.claude/outputs/**` only. A hub **may modify the CLU** — that
work lives on its branch in this repository, and `git push` is forbidden to it.

```bash
# on the SECOND machine (the hub's branch, everything since the pin):
git -C "$CHLU_ROOT" bundle create ~/<hub-slug>.gitbundle agent/autonomous-hub/<hub-slug>
git -C "$CHLU_ROOT" bundle verify ~/<hub-slug>.gitbundle   # "records a complete history"
# on the PRIMARY machine — ⛔ fetch into a namespaced ref; never onto a local branch:
git fetch /path/<hub-slug>.gitbundle 'refs/heads/agent/autonomous-hub/<hub-slug>:refs/remotes/farside/<hub-slug>'
git diff --stat <pin> refs/remotes/farside/<hub-slug>      # the hub's whole code delta
```

⛔⛔ **`git bundle create … main..<branch>` DOES NOT WORK, and it fails only on arrival.**
A range bundle is *thin*: its prerequisite is this repository's own synthetic base
commit, which the primary machine has never seen, so the fetch dies with
`Repository lacks these prerequisite commits`. ⭐ Bundle the branch WHOLE. Measured:
~3 MB, self-contained. ⚠ Take the 3 MB.

⛔ `refs/remotes/farside/*` cannot fast-forward `main` or any local branch — the fetched
history shares no ancestor with it — so the worst case is an unreferenced ref, never a
lost commit. The Hub reviews, then cherry-picks or merges deliberately.

⭐ **The base commit of this repository holds the pin's tree verbatim** — `git log`'s
root commit. `verify_hub_workspace.sh` asserts its tree hash against the manifest, so
"the code is the pin" is checkable in one command, not a promise:

```bash
git -C "$CHLU_ROOT" rev-parse "$(git -C "$CHLU_ROOT" rev-list --max-parents=0 HEAD)^{tree}"
python -c "import json;print(json.load(open('BUNDLE_MANIFEST.json'))['code']['pin_tree'])"
```

```bash
git -C "$CHLU_ROOT" diff --stat "$(git -C "$CHLU_ROOT" rev-list --max-parents=0 HEAD)" HEAD
```
lists the **entire** deviation from the pin: three plumbing files, nothing else (plus
whatever the hub itself has since committed). ⛔ Not `git show --stat HEAD` — a rebuilt
bundle carries a second overlay commit and `show` would display only the last delta.

## Safety of the results ingest

`--return-ingest` **cannot overwrite anything**:
* every member path must be relative, `..`-free, not a symlink, and prefixed with `<hub-slug>`;
* every sha must match `RETURN_MANIFEST.json`, and a file in the tar that the manifest does not
  declare aborts the whole ingest;
* if `.claude/outputs/<hub-slug>` already exists the entire set is diverted to
  `<hub-slug>.returned-<utc>/` and the divert is printed. ⛔ Nothing existing is ever touched.

## What is deliberately NOT here

`.claude/scratch/` · every other slug's outputs · the campaign ledgers · the datasets.
The hub's own profile says: *"⛔ Read nothing else from the campaign archive."* This bundle
enforces that by construction.
