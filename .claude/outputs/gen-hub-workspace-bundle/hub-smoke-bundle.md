# Hub task (STAND-IN) — `hub-smoke-bundle`

⛔ **This is not a real hub charter.** It exists only so
`gen-hub-workspace-bundle` could demonstrate acceptance #1 end-to-end: at the pin
`ae907466ba2b5fbec124e8995cf4d7c32a1ab30e` **no `.claude/tasks/hub-*.md` file exists yet**,
and `bundle_hub_workspace.sh` refuses to build a bundle for a hub with no task file.

⭐ When the Hub writes the three real charters
(`hub-generation-table.md` · `hub-composition.md` · `hub-refusal.md` · `hub-deletion.md`)
into `.claude/tasks/`, drop `--hub-task` and build with the slug alone:

```
scripts/bundle_hub_workspace.sh --hub hub-generation-table --out /path/to/bundle --tar
```

## What this stand-in declares (so the doc-anchor check has something real to read)
- **Target:** win or tie the generation table, through the CLU.
- **Pin:** `ae907466ba2b5fbec124e8995cf4d7c32a1ab30e`.
- **Floor:** every arm must beat the MATCHED-BYTES LOOKUP beyond seed spread.
- **Output path (the only one a hub may write):** `$CHLU_HUB_OUTPUTS/hub-smoke-bundle/**`.
