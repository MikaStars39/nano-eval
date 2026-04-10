#!/usr/bin/env python3
"""
simulator.py — Trajectory-based Tool Simulator

5-tier matching strategy:
1. Write Cache — files written by the model this session; returns cached content on Read
2. Exact Match — tool_name + arguments exact match; returns closest response before cut_position
3. Path Match — for Read/Write/Edit, matches by file_path
4. Name Match — same tool_name; returns closest response before cut_position
5. Smart Fallback — Bash(ls) returns known file listing, Glob returns matching paths
"""

import json
import os
import re
from collections import defaultdict
from fnmatch import fnmatch


FALLBACK_RESPONSES = {
    "Read": "File not found or empty.",
    "Write": "File written successfully.",
    "Edit": "Edit applied successfully.",
    "Bash": "Command executed successfully.",
    "Glob": "[]",
    "Grep": "No matches found.",
    "TodoWrite": "TODO updated.",
    "Skill": "Skill loaded.",
    "mcp__playwright__browser_navigate": '{"title": "Page", "content": "Page content loaded."}',
    "mcp__playwright__browser_install": "Browser installed.",
    "mcp__web-tools__search": '{"results": []}',
    "mcp__web-tools__browse": '{"content": "Page content."}',
}

# Write-operation tools, always return success
WRITE_TOOLS = {"Write", "Edit", "TodoWrite", "Skill"}

# File-path tools
FILE_PATH_TOOLS = {"Read", "Write", "Edit"}


class ToolSimulator:
    """
    Simulates tool calls based on original trajectory tool responses.

    Args:
        tool_responses: [{name, arguments, response, msg_index}, ...]
            Full set of tool call→response mappings extracted from the entire trajectory
        cut_position: prefix truncation position (msg index where the target subtask begins)
        known_file_paths: all file paths that appeared in the trajectory
    """

    def __init__(
        self,
        tool_responses: list[dict],
        cut_position: int = 0,
        known_file_paths: list[str] | None = None,
    ):
        self.cut_position = cut_position
        self.known_paths: set[str] = set(known_file_paths or [])

        # Index: key → [(msg_index, response), ...]
        self.exact_index: dict[str, list[tuple[int, str]]] = defaultdict(list)
        # Index: file_path → [(msg_index, response), ...] (Read tool responses only)
        self.path_index: dict[str, list[tuple[int, str]]] = defaultdict(list)
        # Index: tool_name → [(msg_index, response), ...]
        self.name_index: dict[str, list[tuple[int, str]]] = defaultdict(list)

        # Runtime write cache
        self.write_cache: dict[str, str] = {}

        # Stats
        self.cache_hits = 0
        self.exact_hits = 0
        self.path_hits = 0
        self.name_hits = 0
        self.fallback_hits = 0
        self.call_count = 0
        self.consecutive_fallbacks = 0

        # Build indices
        for tr in tool_responses:
            name = tr["name"]
            args = tr.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            response = tr["response"]
            msg_idx = tr.get("msg_index", 0)

            # Exact index
            key = self._make_key(name, args)
            self.exact_index[key].append((msg_idx, response))

            # Path index (only Read responses — file content to return to the model)
            if name == "Read" and isinstance(args, dict):
                file_path = args.get("file_path", "")
                if file_path:
                    self.path_index[file_path].append((msg_idx, response))
                    self.known_paths.add(file_path)

            # Collect all file paths (including Write/Edit)
            if name in FILE_PATH_TOOLS and isinstance(args, dict):
                fp = args.get("file_path", "")
                if fp:
                    self.known_paths.add(fp)

            # Name index
            self.name_index[name].append((msg_idx, response))

    @staticmethod
    def _make_key(name: str, args: dict) -> str:
        return f"{name}::{json.dumps(args, sort_keys=True, ensure_ascii=False)}"

    def _closest_before(self, entries: list[tuple[int, str]]) -> str | None:
        """Return the response with the largest msg_index <= cut_position; otherwise the nearest one."""
        if not entries:
            return None
        before = [(idx, resp) for idx, resp in entries if idx <= self.cut_position]
        if before:
            return max(before, key=lambda x: x[0])[1]
        return min(entries, key=lambda x: x[0])[1]

    def simulate(self, tool_name: str, arguments: dict) -> str:
        """Simulate a single tool call and return the response text."""
        self.call_count += 1
        file_path = arguments.get("file_path", "") if isinstance(arguments, dict) else ""

        # ── Tier 1: Write Cache ──
        if tool_name == "Read" and file_path and file_path in self.write_cache:
            self.cache_hits += 1
            self.consecutive_fallbacks = 0
            return self.write_cache[file_path]

        if tool_name == "Write" and file_path:
            content = arguments.get("content", "")
            self.write_cache[file_path] = content
            self.cache_hits += 1
            self.consecutive_fallbacks = 0
            return "File written successfully."

        if tool_name == "Edit" and file_path:
            old_str = arguments.get("old_string", "")
            new_str = arguments.get("new_string", "")
            # Seed cache: get current file content from path_index or existing cache
            if file_path not in self.write_cache:
                seed = self._closest_before(self.path_index.get(file_path, []))
                if seed:
                    self.write_cache[file_path] = seed
            if file_path in self.write_cache and old_str:
                self.write_cache[file_path] = self.write_cache[file_path].replace(old_str, new_str, 1)
            self.cache_hits += 1
            self.consecutive_fallbacks = 0
            return "Edit applied successfully."

        if tool_name == "TodoWrite":
            self.cache_hits += 1
            self.consecutive_fallbacks = 0
            return "TODO updated."

        # ── Tier 2: Exact Match ──
        key = self._make_key(tool_name, arguments)
        if key in self.exact_index:
            resp = self._closest_before(self.exact_index[key])
            if resp is not None:
                self.exact_hits += 1
                self.consecutive_fallbacks = 0
                return resp

        # ── Tier 3: Path Match (file tools only) ──
        if tool_name == "Read" and file_path and file_path in self.path_index:
            resp = self._closest_before(self.path_index[file_path])
            if resp is not None:
                self.path_hits += 1
                self.consecutive_fallbacks = 0
                return resp

        # ── Tier 4: Name Match (returns temporally closest, not first) ──
        if tool_name in self.name_index:
            resp = self._closest_before(self.name_index[tool_name])
            if resp is not None:
                self.name_hits += 1
                self.consecutive_fallbacks = 0
                return resp

        # ── Tier 5: Smart Fallback ──
        self.fallback_hits += 1
        self.consecutive_fallbacks += 1
        return self._smart_fallback(tool_name, arguments)

    def _smart_fallback(self, tool_name: str, arguments: dict) -> str:
        """Smart fallback: return meaningful responses for filesystem queries."""
        if tool_name == "Bash":
            command = arguments.get("command", "")
            return self._simulate_bash(command)

        if tool_name == "Glob":
            pattern = arguments.get("pattern", "")
            return self._simulate_glob(pattern)

        if tool_name == "Grep":
            return "No matches found."

        return FALLBACK_RESPONSES.get(tool_name, "OK")

    def _simulate_bash(self, command: str) -> str:
        """Simulate Bash commands: return known file listings for ls/find."""
        # ls command
        ls_match = re.search(r'\bls\b\s+([^\s|;]+)', command)
        if ls_match or command.strip().startswith("ls"):
            target_dir = ls_match.group(1) if ls_match else "."
            # Normalize path
            target_dir = target_dir.rstrip("/")
            files = []
            for p in sorted(self.known_paths):
                parent = os.path.dirname(p)
                if parent.rstrip("/") == target_dir or (target_dir == "." and "/" not in p):
                    files.append(os.path.basename(p))
            if files:
                return "\n".join(files)
            return ""

        # cat command → treat as Read
        cat_match = re.search(r'\bcat\b\s+([^\s|;]+)', command)
        if cat_match:
            file_path = cat_match.group(1)
            if file_path in self.write_cache:
                return self.write_cache[file_path]
            resp = self._closest_before(self.path_index.get(file_path, []))
            if resp:
                return resp

        return "Command executed successfully."

    def _simulate_glob(self, pattern: str) -> str:
        """Simulate Glob: filter known file paths using fnmatch."""
        matches = [p for p in sorted(self.known_paths) if fnmatch(p, pattern)]
        return json.dumps(matches)

    @property
    def match_summary(self) -> str:
        return (
            f"cache={self.cache_hits} exact={self.exact_hits} "
            f"path={self.path_hits} name={self.name_hits} fallback={self.fallback_hits}"
        )
