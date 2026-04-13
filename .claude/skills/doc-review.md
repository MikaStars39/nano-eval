---
name: doc-review
description: Review and restructure project documentation for AI-agent-friendly consumption (CLAUDE.md, README.md, module docs)
user_invocable: true
---

# Review Documentation for AI-Friendliness

You are reviewing and restructuring a project's documentation to optimize for **model context understanding efficiency** — not human reading experience. The primary consumer is an AI coding agent that reads files, understands codebases, and decides where to modify code.

## Design Philosophy

> Put it simply: write documentation as system prompts for models, not onboarding materials for new employees. The latter assumes the reader has common sense and can ask follow-up questions; the former requires making all assumptions explicit.

## Three-Layer Structure

Evaluate whether the project has all three layers, and whether each layer contains the right content.

### Layer 1: Global — `CLAUDE.md` (auto-loaded by Claude Code)

This is the most critical file. Must contain:

- **One-line description**: what the repo is and what problem it solves
- **Setup / test commands**: concrete commands, never "refer to official docs"
- **Tech stack and core conventions**: e.g. "all async functions use async/await, no callback style"
- **No-go zones**: which files/modules not to touch, and why
- **Entry map**: index of key files with one-line descriptions:

```
- src/core/engine.py    # Core scheduling logic entry point
- config/schema.yaml    # Source of truth for all config
- scripts/migrate.sh    # Data migration — don't touch lightly
```

### Layer 2: Module — subdirectory `README.md` or top-of-file docstrings

When the agent is working in a specific directory, it reads these. Core purpose is **boundary declaration**: what interfaces this module exposes, what internal details can be ignored, what external services it depends on.

### Layer 3: Task — `AGENTS.md` or `CONTRIBUTING.md`

Operational manuals for specific task types. E.g. "standard flow for adding a new API endpoint", "must write a failing test before fixing a bug". These serve as guardrails for agentic tasks.

## Six Core Principles

Apply these when reviewing each document:

### 1. Explicit over implicit

Models lack your team's tacit knowledge. "Keep code style consistent with existing code" is useless — the model might have read 5 files with inconsistent styles. Write specific rules.

### 2. Scope declaration

Every document's first paragraph should state "this document covers X, not Y". This makes the model's "should I keep reading this file" decision nearly free.

### 3. Counter-examples > examples

A `# Don't do this` section is more valuable than positive descriptions. Models tend to generate code in directions they consider "reasonable" — counter-examples correct this bias.

### 4. Machine-readable structure

Use consistent markdown heading hierarchy. Put key information in `code blocks` or `> blockquotes` for easy parsing. Avoid long prose paragraphs.

### 5. Decision log

Many architectural decisions look "weird" for historical reasons. Document "why we chose Y over X" — otherwise the model will "optimize away" your tech debt. Can go in `docs/decisions/` or inline comments.

### 6. Entry map

Maintain a key file index in `CLAUDE.md`. This directly saves massive tokens that the model would otherwise spend exploring the file system.

## Review Process

When invoked, do the following:

1. **Read `CLAUDE.md`** — check against Layer 1 requirements above. Identify what's missing or could be restructured.

2. **Scan for module-level docs** — `Glob` for `**/README.md` and check if key subdirectories have boundary declarations.

3. **Check for task-layer docs** — look for `AGENTS.md`, `CONTRIBUTING.md`, or equivalent.

4. **Apply the six principles** — for each document found, evaluate against each principle. Focus on:
   - Are there vague instructions that assume context? (violates #1)
   - Does each doc declare its scope? (#2)
   - Are anti-patterns documented? (#3)
   - Is the structure scannable by a model? (#4)
   - Are non-obvious decisions explained? (#5)
   - Is there a file index? (#6)

5. **Propose concrete edits** — don't just list problems. Show the specific changes (additions, restructuring, removals) with rationale.

## Output Format

Present findings as:

```
## Current State
Brief assessment of doc coverage and quality.

## Issues Found
For each issue:
- **File**: which doc
- **Principle violated**: which of the 6
- **Problem**: what's wrong
- **Fix**: concrete edit or addition

## Recommended Changes
Ordered list of edits, most impactful first.
```

## Rules

- Do NOT create documentation files unless the user approves the changes.
- Prefer editing existing files over creating new ones.
- Keep `CLAUDE.md` concise — it's loaded into every conversation. Under 100 lines is ideal.
- Don't duplicate information that can be derived from code or git history.
- Focus on information that saves the model exploration time or prevents wrong decisions.
