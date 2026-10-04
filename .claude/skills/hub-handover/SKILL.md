---
name: hub-handover
description: >-
  Flush the Hub's session state into the CLU program's living documents before the context window
  fills or the session ends. Use when the user says "handover", "flush state", "wrap up the session",
  or when a research-lead (Hub) session is ending with unrecorded progress. Updates
  .claude/handover_context.md (running log + affected sections), brainstorm_log.md statuses,
  research_roadmap.md deltas, and the cross-session memory index — so the next Hub agent can resume
  with zero context loss.
---

# Hub Handover Ritual

You are the Hub (research-lead) agent for the CLU program. Execute the full handover flush so a fresh Hub agent can resume seamlessly. Work through ALL steps; do not skip any because they "seem already covered."

## Steps

1. **Read the current state docs** (skim for staleness, don't re-read what you know): `.claude/handover_context.md`, `.claude/research_roadmap.md`, `.claude/brainstorm_log.md`.

2. **Handover doc (`.claude/handover_context.md`):**
   - Append a dated entry to the **§10 Running Log**: what happened this session — decisions by the Head, spoke waves launched/reviewed (with verdicts), merges, discoveries, corrections. Dense and factual; a new agent must be able to act on it.
   - Edit any **section made stale** by this session (Known Issues statuses, config defaults, architecture map, experiment provenance, agent roster/workflow). Never delete prior findings — strike and annotate if superseded.

3. **Brainstorm log (`.claude/brainstorm_log.md`):** append idea-level events only (new ideas, status transitions raw→shaped→adopted/parked/killed, Head decisions on threads). Attribute Head vs Hub. Append-only.

4. **Roadmap (`.claude/research_roadmap.md`):** update wave status tables, decisions, dates/freezes, and vertical scopes if they moved. Keep the binding-updates style: patch blocks, don't rewrite history.

5. **Task hygiene:** ensure every launched-but-unreviewed task file is listed as pending in the handover log's "Hub next" line; ensure staged tasks (e.g. date-gated scouts) carry their fire conditions.

6. **Cross-session memory** (`/Users/user/.claude/projects/-Users-user-Desktop-CHLU/memory/`): update only if something *durable* changed — operating model, Head preferences, environment gotchas that outlive the repo docs. Do not duplicate what the handover doc already records.

7. **Git state note:** record in the log entry — current branch, commits ahead of origin, any unmerged agent branches, uncommitted work.

8. **Close with a one-paragraph "Next Hub agent starts here"** at the end of the new log entry: the single most important thing to do next, pending Head decisions, and any time-critical dates (deadlines, gated scouts).

9. **End your reply to the Head with the resume command word** for the fresh thread, verbatim:
   > *"Act as my AI and physics research lead (Hub). Read `.claude/handover_context.md` and continue from the latest §10 Running Log entry."*
   Note for the fresh thread: if Fable is unavailable there, the Hub runs on Opus — the handover docs are written to be model-agnostic, so nothing else changes.

## Rules
- The handover doc is the single source of truth; the log entry must stand alone.
- Honest statuses only: partial work is recorded as partial, failed runs as failed.
- Do not launch new work during a handover flush; this skill records, it does not execute.
