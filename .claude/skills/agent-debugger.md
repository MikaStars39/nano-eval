---
name: agent-debugger
description: Two-phase adversarial bug audit — first find bugs, then challenge every finding to eliminate false positives. Only confirmed bugs survive.
user_invocable: true
---

# Agent Debugger — Adversarial Bug Audit

You are orchestrating a two-phase adversarial debugging process. The key insight: bug-finding agents produce many false positives. A second pass that tries to DISPROVE each finding dramatically improves precision.

## Phase 1: Bug Discovery

### Step 1 — Determine Scope

Ask the user what to scan using `AskUserQuestion`:

1. **Scope**: Full repo, specific directory, or recent changes only?
2. **Focus**: Any specific concern? (e.g., "concurrency issues", "data pipeline correctness", "API error handling") Or general scan?

If the user provides a scope argument (e.g., `/agent-debugger nanoeval/backend/`), use that as the scope without asking.

### Step 2 — Launch Parallel Bug Scanners

Launch **3-4 exploration agents in parallel** (using the `Agent` tool with `subagent_type: "Explore"`), each covering a different area:

**For full-repo scans**, split into:
1. **Core modules** (`nanoeval/`) — inference engines, scoring, utilities
2. **Recipes** (`recipes/`) — experiment scripts, task-specific code
3. **Entry point + tests** (`run.py`, `tests/`) — argument parsing, test coverage
4. **Cross-module consistency** — import chains, JSONL schema assumptions, function signature mismatches

**For targeted scans**, split the target into logical sub-areas.

Each agent should look for:
- Logic errors, off-by-one errors, incorrect conditions
- Unhandled exceptions at system boundaries
- Race conditions and concurrency issues
- Data flow bugs (JSONL field assumptions, missing fields)
- Dead code, unused imports
- Inconsistent behavior across code paths

### Step 3 — Collect Findings

Gather all bug reports from the parallel agents. Create a numbered list of every claimed bug with:
- File path and line number
- One-line description
- Claimed severity

## Phase 2: Adversarial Verification

### Step 4 — Challenge Every Finding

For EACH bug claim from Phase 1, apply these adversarial checks yourself (do NOT delegate this — you must read the code):

**4a. Read the actual code.** Don't trust the agent's code quotes. Read the file at the cited line number. Verify the code says what the report claims.

**4b. Trace the data flow upstream.** If the claim is "X can be None/missing":
- Grep for all callers of the function
- Check what the callers guarantee
- If every caller guarantees the value, it's not a bug

**4c. Check API contracts.** If the claim is "missing validation for field X":
- Is this a standardized benchmark format that guarantees the field?
- Is crashing on bad input actually correct fail-fast behavior?

**4d. Verify concurrency model.** If the claim is "race condition":
- asyncio = single-threaded, no true parallelism
- multiprocessing = separate memory, no shared state
- Each item having its own instance = no shared state

**4e. Grep before declaring "unused".** Search the entire repo, not just the file.

**4f. Check documentation.** If the claim is "silently ignored", check if the help text documents it.

**4g. Distinguish bug vs. preference.** "Could add `.get()` fallback" on a field that's always present is a style preference, not a bug.

### Step 5 — Classify Each Finding

For each claim, assign a verdict:
- **TRUE BUG**: The code is actually wrong and will produce incorrect behavior
- **CODE SMELL**: Not wrong, but could be improved (e.g., bare `except:`, misleading docstring)
- **FALSE ALARM**: Not a bug at all — agent misunderstood the code, data flow, or concurrency model

## Phase 3: Final Report

### Step 6 — Present Results

Output a clean summary:

```
## Audit Results

**Scope**: [what was scanned]
**Phase 1 findings**: N claims
**After verification**: X true bugs, Y code smells, Z false alarms

### Confirmed Bugs (action required)

| # | File | Line | Issue | Severity |
|---|------|------|-------|----------|
| 1 | ... | ... | ... | ... |

### Code Smells (optional cleanup)

| # | File | Line | Issue |
|---|------|------|-------|
| 1 | ... | ... | ... |

### False Alarms Eliminated

| # | Original Claim | Why It's Not a Bug |
|---|----------------|-------------------|
| 1 | ... | ... |
```

### Step 7 — Offer to Fix

Ask the user if they want to fix the confirmed bugs and/or code smells.

## Rules

- **Phase 2 is mandatory**. Never skip verification. The whole point of this skill is the adversarial second pass.
- **Read code yourself in Phase 2**. Do not delegate verification to another agent — you need the full context to make correct judgments.
- **Be honest about uncertainty**. If you can't fully disprove a claim, say so. Don't force false-alarm verdicts.
- **Respect project conventions**. Check CLAUDE.md before flagging "anti-patterns" — they might be project conventions (e.g., recipes/ code duplication is by design).
- **This machine has no GPU or project dependencies**. Do not run Python imports. Read and grep only.
