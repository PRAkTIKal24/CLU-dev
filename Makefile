# CHLU development helpers.
#
# make fix-env — repair the editable install on macOS (handover §7.12).
#
# Root cause: on this machine some sessions end up with the macOS UF_HIDDEN
# file flag set on freshly written files under .venv/**, including uv's
# editable-install hook `_editable_impl_chlu.pth`. Python >= 3.11 silently
# SKIPS hidden .pth files (site.addpackage checks st_flags & stat.UF_HIDDEN),
# so `import chlu` — and with it the `chlu` CLI — breaks with
# ModuleNotFoundError. The flag can reappear whenever uv rewrites the
# editable .pth (`uv run`'s implicit sync reinstalls the project after
# pyproject/branch changes) from an affected session, so a one-off
# `chflags nohidden` is not durable.
#
# Durable fix implemented by `fix-env`:
#   1. clear UF_HIDDEN recursively from .venv (heals the current state), and
#   2. drop an *unmanaged* path shim `zzz_chlu_dev.pth` into site-packages
#      containing this project root (the same content as the editable hook).
#      uv never rewrites unmanaged files, so the shim never gets re-flagged:
#      imports keep working even if the uv-managed .pth is hidden again.
#
# Re-run `make fix-env` after recreating the venv (rm -rf .venv && uv sync),
# and once inside each new git worktree (every worktree venv needs its own
# shim; run it from the worktree root).

VENV := .venv
SITE_PACKAGES = $(wildcard $(VENV)/lib/python3.*/site-packages)

.PHONY: fix-env
fix-env:
	@test -n "$(SITE_PACKAGES)" || { echo "error: no $(VENV) found — run 'uv sync' first" >&2; exit 1; }
	chflags -R nohidden $(VENV)
	printf '%s' "$(CURDIR)" > "$(SITE_PACKAGES)/zzz_chlu_dev.pth"
	@$(VENV)/bin/python -c "import chlu" \
		&& echo "OK: chlu importable; shim at $(SITE_PACKAGES)/zzz_chlu_dev.pth"
