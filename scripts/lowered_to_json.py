#!/usr/bin/env python3
"""
scripts/lowered_to_json.py

Simple, forgiving parser that converts a Halide textual lowered-stmt (pipelines/.../lowered.txt)
into a JSON structure. It groups:
 - allocate/free
 - produce/consume blocks (per function)
 - nested loops (captures loop var + extent)
 - assignments (keeps RHS as raw strings)

Usage:
  python3 scripts/lowered_to_json.py pipelines/example/lowered.txt > pipelines/example/lowered.json
"""
import sys
import json
import re

def indent_level(line):
    # count leading spaces (2 per indent in typical Halide text, but we'll use spaces)
    return len(line) - len(line.lstrip(' '))

def parse_lowered(lines):
    root = {"functions": [], "output": None, "allocs": [], "frees": []}
    stack = []  # (type, name, container)
    current = {"type": "root", "container": root, "indent": -1}
    for raw in lines:
        line = raw.rstrip('\n')
        if not line.strip():
            continue
        ind = indent_level(line)
        s = line.strip()

        # allocate
        m = re.match(r'allocate\s+([^\[]+)\[', s)
        if m:
            name = m.group(1)
            entry = {"name": name, "line": s}
            root["allocs"].append(entry)
            continue

        # free
        m = re.match(r'free\s+(\S+)', s)
        if m:
            name = m.group(1)
            root["frees"].append(name)
            continue

        # produce start
        m = re.match(r'produce\s+(\S+)\s*\{', s)
        if m:
            fname = m.group(1)
            func = {"name": fname, "produce": {"loops": [], "stmts": []}, "consume": {"loops": [], "stmts": []}}
            root["functions"].append(func)
            stack.append(("produce", fname, func, ind))
            continue

        # consume start
        m = re.match(r'consume\s+(\S+)\s*\{', s)
        if m:
            fname = m.group(1)
            # find function entry
            func = None
            for f in root["functions"]:
                if f["name"] == fname:
                    func = f
                    break
            if func is None:
                func = {"name": fname, "produce": {"loops": [], "stmts": []}, "consume": {"loops": [], "stmts": []}}
                root["functions"].append(func)
            stack.append(("consume", fname, func, ind))
            continue

        # produce/consume end
        if s == '}' and stack:
            stack.pop()
            continue

        # loop line: for var, start, extent
        m = re.match(r'for\s+([^\:,.\s]+)[\.,\s]*[^\d]*,\s*0,\s*(.+)', s)
        if m:
            var = m.group(1)
            extent = m.group(2).strip()
            if stack:
                top = stack[-1]
                typ, fname, func, fint = top
                target = func["produce"] if typ == "produce" else func["consume"]
                target["loops"].append({"var": var, "extent": extent, "line": s})
            else:
                # top-level loop -> output?
                if root.get("output") is None:
                    root["output"] = {"loops": [], "stmts": []}
                root["output"]["loops"].append({"var": var, "extent": extent, "line": s})
            continue

        # assignment or expression (very permissive)
        m = re.match(r'(\S+)\s*=\s*(.+)', s)
        if m:
            lhs = m.group(1)
            rhs = m.group(2)
            entry = {"lhs": lhs, "rhs": rhs, "line": s}
            if stack:
                typ, fname, func, fint = stack[-1]
                target = func["produce"] if typ == "produce" else func["consume"]
                target["stmts"].append(entry)
            else:
                if root.get("output") is None:
                    root["output"] = {"loops": [], "stmts": []}
                root["output"]["stmts"].append(entry)
            continue

        # produce output block start
        m = re.match(r'produce\s+output\s*\{', s)
        if m:
            if root.get("output") is None:
                root["output"] = {"loops": [], "stmts": []}
            stack.append(("produce_output", "output", root["output"], ind))
            continue

        # any other lines: stash as raw info in current context
        if stack:
            typ, fname, func, fint = stack[-1]
            target = func["produce"] if typ == "produce" else func["consume"]
            target.setdefault("raw", []).append(s)
        else:
            root.setdefault("raw", []).append(s)

    return root


def main():
    if len(sys.argv) != 2:
        print("Usage: lowered_to_json.py <lowered.txt>", file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    with open(path, 'r') as f:
        lines = f.readlines()
    parsed = parse_lowered(lines)
    print(json.dumps(parsed, indent=2))

if __name__ == "__main__":
    main()
