#!/usr/bin/env python3
"""Serve info: records which vLLM an eval run actually talked to (version, image, GPU budget, free memory).

Why: docker/docker-compose.yml pulled an unpinned `vllm/vllm-openai:latest` until 2026-11-06. That tag
moved from v0.21.0 to v0.30.0 on 2026-09-23 without anyone noticing. A measuring instrument must not
change version silently (v0.30 also brings new parsers and tool-call grammars; with tool_choice=auto they
only apply when a tool carries "strict": true). No run had recorded the version, and the GPU-util
overrides only showed up in run.log.

The eval scripts call this right after a server reports healthy:

    serve_info.py record --file <run>/serve_info.json --role agent --port 8000 \
        --container text2sql_vllm_teacher --gpu-util 0.22 [--gate]

The JSON file holds:
  * one identity per role (`agent`, `user_sim`, `server`, ...): {vllm_version, image_id, image_ref},
    fixed at the first start;
  * `events`: one entry per server start: {utc, role, vllm_version, image_id, gpu_util,
    mem_avail_gib[, mismatch]}. A resumed run therefore shows its whole start history.

Identity check:
  * image_id always counts;
  * vllm_version counts only when both sides know it.

  --gate   a start whose identity differs from the recorded one is refused (exit 3) and nothing is
           written: one label = one vLLM version, so a resume cannot mix versions.
           ALLOW_VLLM_CHANGE=1 overrides that on purpose (the event then carries `mismatch`).
  no gate  a mismatch is only a warning. That is enough for the DB heldout and BFCL, which do not depend
           on the version.

Exit codes:
  0  recorded
  2  nothing measurable: no /version answer and no container
  3  gate refused

Stdlib only, so it runs under any of the repo venvs. /version is fetched with proxies disabled, so an
http(s)_proxy in the environment (as on the Atos H100) cannot hijack the localhost call. Without docker
(a plain `vllm serve`), image_id stays None and the identity rests on the version alone.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

IDENTITY_KEYS = ("vllm_version", "image_id", "image_ref")


def mem_available_gib(meminfo_text: str | None = None) -> float | None:
    """MemAvailable in GiB (what vLLM itself compares against on unified-memory boards like the GB10)."""
    if meminfo_text is None:
        try:
            meminfo_text = Path("/proc/meminfo").read_text()
        except OSError:
            return None
    for line in meminfo_text.splitlines():
        if line.startswith("MemAvailable:"):
            return round(int(line.split()[1]) / 1024 ** 2, 2)
    return None


def fetch_version(port: int, timeout: float = 5.0) -> str | None:
    """GET http://localhost:<port>/version without any proxy; None when there is no answer."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://localhost:{port}/version", timeout=timeout) as resp:
            return json.loads(resp.read().decode()).get("version")
    except Exception:  # noqa: BLE001 — any failure means "not measurable", the caller decides
        return None


def docker_image(container: str) -> tuple[str | None, str | None]:
    """(image_id, image_ref) of a running container; (None, None) without docker or container."""
    try:
        out = subprocess.run(["docker", "inspect", "--format", "{{.Image}}|{{.Config.Image}}", container],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    if out.returncode != 0 or "|" not in out.stdout:
        return None, None
    image_id, image_ref = out.stdout.strip().split("|", 1)
    return image_id or None, image_ref or None


def identity_diff(old: dict, new: dict) -> list[str]:
    """Identity fields in which two starts disagree (see module docstring)."""
    diffs = []
    if old.get("image_id") != new.get("image_id"):
        diffs.append("image_id")
    if old.get("vllm_version") and new.get("vllm_version") and old["vllm_version"] != new["vllm_version"]:
        diffs.append("vllm_version")
    return diffs


def _write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def record(path: Path, role: str, ident: dict, event: dict, gate: bool,
           allow_change: bool = False) -> tuple[bool, str]:
    """Add one server start to the file. Returns (ok, message); ok=False means refused, nothing written."""
    data = json.loads(path.read_text()) if path.exists() else {}
    old = data.get(role)
    mismatch = identity_diff(old, ident) if old is not None else []
    msg = ""
    if mismatch:
        detail = ", ".join(f"{k} {old.get(k)} -> {ident.get(k)}" for k in mismatch)
        if gate and not allow_change:
            return False, (f"{role}: vLLM weicht vom bisherigen Lauf ab ({detail}) — abgelehnt; ein Label = "
                           "eine vLLM-Version (ALLOW_VLLM_CHANGE=1 uebersteuert bewusst)")
        msg = f"WARN {role}: vLLM weicht vom bisher erfassten ab ({detail})"
    if old is None:
        data[role] = {k: ident.get(k) for k in IDENTITY_KEYS}
    entry = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "role": role,
             "vllm_version": ident.get("vllm_version"), "image_id": ident.get("image_id"), **event}
    if mismatch:
        entry["mismatch"] = mismatch
    data.setdefault("events", []).append(entry)
    _write_atomic(path, data)
    return True, msg


def _short(image_id: str | None) -> str:
    return image_id.split(":", 1)[-1][:12] if image_id else "?"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="record one server start (call right after /health is OK)")
    r.add_argument("--file", type=Path, required=True)
    r.add_argument("--role", required=True)
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--container", required=True)
    r.add_argument("--gpu-util", type=float)
    r.add_argument("--gate", action="store_true")
    a = ap.parse_args(argv)

    version = fetch_version(a.port)
    image_id, image_ref = docker_image(a.container)
    if version is None and image_id is None:
        print(f"== vLLM {a.role}: nichts messbar (kein /version auf :{a.port}, kein Container "
              f"{a.container}) — nichts geschrieben", file=sys.stderr)
        return 2
    mem = mem_available_gib()
    ident = {"vllm_version": version, "image_id": image_id, "image_ref": image_ref}
    ok, msg = record(a.file, a.role, ident, {"gpu_util": a.gpu_util, "mem_avail_gib": mem}, a.gate,
                     allow_change=os.environ.get("ALLOW_VLLM_CHANGE") == "1")
    if not ok:
        print(f"== vLLM-GATE: {msg}")
        return 3
    if msg:
        print(f"== {msg}")
    util = "?" if a.gpu_util is None else a.gpu_util
    print(f"== vLLM {a.role}: {version or '?'} | image {_short(image_id)} ({image_ref or '?'}) | "
          f"util {util} | verfuegbar {mem if mem is not None else '?'} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
