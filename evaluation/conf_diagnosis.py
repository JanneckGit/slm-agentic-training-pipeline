#!/usr/bin/env python3
"""Confidence diagnosis — decides whether r3 trains with the confidence-weighted loss
(training_pipeline/grpo_conf_loss.py, YAML train.conf_weight).

The question: when the model gets an arrival time wrong (t_info_ankunft*: scheduled arrival + delay,
carried over the hour wrongly), was it UNSURE at the wrong digit — a dent in an otherwise confident
attempt — or confidently wrong? Only in the first case does "keep the full negative gradient for the
uncertain tokens, scale it down for the confident ones" aim at the mistake.

  measure  (grpo image, GPU; torch + transformers)
      teacher-forced forward of the model over its own K=8 heldout traces at T = 1.0 (the RL sampling
      temperature). Population = every attempt of every task with >= 1 attempt that eval_report does not
      accept. Rendered exactly like the SFT collator (asserted token-for-token against
      collator_multiturn.build_masked_labels); scored = the assistant tokens it trains on. Per token:
      p, entropy, rank, segment (think / text / tool_call / answer / end). Wrong arrival times: the first
      occurrence of the wrong time in the final turn's think block (else its answer), first digit that
      differs from the correct time -> p of the written digit (p_err), p of the correct digit in the same
      distribution, percentile of p_err among the attempt's tokens; siblings: p of the correct digit
      where accepted attempts of the task write the correct time.
      Fidelity: the traces were sampled with top_k 20, so a re-rendered token with rank > 20 was not
      what the model emitted — a segment with > 0.5 % of those is not trustworthy. Logits come from the
      bf16 lm_head like the verl actor's old_log_prob (the quantity tau is compared against in training);
      log-softmax in fp32.
      Not counted: (1) "wrong" times that are only the 24-h wrap of an observed time (GTFS 33:11 written
      as 09:11 — correct, but the verifier's grounding corpus lacks the wrap; reported as
      `time_artefacts`), (2) the token statistics of truncated attempts — the chat template injects an
      empty think block before an unclosed one, so their runaway think would count as `answer`
      (reported as `truncated_excluded`).
  decide   (host, plain python) — the rule fixed BEFORE the measurement (28.10.2026):
      A  median p_err <= 0.6 and median percentile <= 5 %  -> tau = smallest of 0.5..0.9 with >= 80 % of
         the error tokens below it; on only if at most 20 % of all assistant tokens of failed attempts
         lie below tau (selectivity; from --corridor when given = T 1.0 training traces), else off
      B  median p_err >= 0.9 (confidently wrong)          -> off
      C  anything else                                     -> off
      The decision rests on the init model (ep3, --ep3); --s24 is reported beside it.

    docker compose -f docker/docker-compose.yml --profile grpo run --rm -T grpo evaluation/conf_diagnosis.py \
        measure --model /app/data/final/checkpoints/db_bahn_traj_merged_qwen3-4b/ep3 \
        --traces /app/data/generated/eval/db_traces_heldout_eval_after_qwen3-4b_ep3_k8.jsonl --label ep3 \
        --out /app/data/generated/eval/conf_diagnosis/ep3_heldout.json
    python3 evaluation/conf_diagnosis.py decide --ep3 data/generated/eval/conf_diagnosis/ep3_heldout.json \
        [--s24 …/s24_heldout.json] [--corridor …/ep3_corridor.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TAUS = (0.5, 0.6, 0.7, 0.8, 0.9)
SEGMENTS = ("think", "text", "tool_call", "answer", "end")
RANK_LIMIT = 20                 # top_k of the trace sampling
UNRELIABLE_RANK_SHARE = 0.005   # > 0.5 % tokens with rank > 20 in a segment -> not trustworthy
_TIME = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?!\d)")
_ASSISTANT_HDR = "<|im_start|>assistant\n"
_IM_END = "<|im_end|>"


# --- pure helpers (tested in tests/test_conf_diagnosis.py) ------------------------------------------
def _hm(t: str) -> tuple[int, int]:
    h, m = t.split(":")
    return int(h), int(m)


def _mins(t: str) -> int:
    h, m = _hm(t)
    return (h * 60 + m) % 1440


def time_verdict(rec: dict) -> dict | None:
    """A failed t_info_ankunft* attempt whose final answer carries ungrounded times:
    {"kind": "wrong", "written", "want", "wrap"} when one of them is neither scheduled arrival + delay nor an
    observed time (`wrap`: observed times >= 24:00 written mod 24 h — the reference when the model was
    converting the planned time, t_info_ankunft_suche) (all compared as minutes mod 24 h), {"kind": "artefact", "written"} when every one is
    such a correct value the verifier could not ground (the 24-h wrap: tool 33:11 written as 09:11), else
    None. Otherwise the logic of the K=8 comparison (logs/k8/compare_trained_k8.py, grounding section)."""
    from evaluation.trajectory_reward import _final_answer, _grounding_corpus, _grounding_pass
    comps = (rec.get("score") or {}).get("components") or {}
    if (rec.get("score") or {}).get("score", 0) >= 1 or comps.get("grounding", True) \
            or not rec.get("template", "").startswith("t_info_ankunft"):
        return None
    msgs = rec["messages"]
    ticket = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    _, missing = _grounding_pass(_final_answer(msgs), _grounding_corpus(msgs, ticket))
    raw = "\n".join(m.get("content") or "" for m in msgs if m.get("role") == "tool")
    want, observed, wrap = set(), set(), set()
    for h, mi in re.findall(r'"ankunft":\s*"(\d{1,2}):(\d{2})"', raw):
        observed.add((int(h) * 60 + int(mi)) % 1440)
        if int(h) >= 24:                                   # the planned time as it should be written (33:11 -> 09:11)
            wrap.add(f"{int(h) % 24:02d}:{mi}")
        for d in (int(x) for x in re.findall(r'"verspaetung_minuten":\s*(\d+)', raw)):
            tot = int(h) * 60 + int(mi) + d
            want.add(f"{(tot // 60) % 24:02d}:{tot % 60:02d}")
    times = [t for t in missing if re.fullmatch(r"\d{1,2}:\d{2}", t)]
    if not times or not want:
        return None
    ok = {_mins(w) for w in want} | observed
    bad = next((t for t in times if _mins(t) not in ok), None)
    if not bad:
        return {"kind": "artefact", "written": times[0]}
    return {"kind": "wrong", "written": bad, "want": want, "wrap": wrap}


def wrong_time(rec: dict) -> tuple[str, set[str]] | None:
    """(wrong time as written, correct times) for a real arithmetic error, else None — see time_verdict."""
    v = time_verdict(rec)
    return (v["written"], v["want"]) if v and v["kind"] == "wrong" else None


def is_truncated(rec: dict) -> bool:
    """The attempt ran out of budget: flagged by the harness, or its last assistant turn opens a think block that
    never closes."""
    last = next((m.get("content") or "" for m in reversed(rec["messages"]) if m.get("role") == "assistant"), "")
    return bool(rec.get("truncated")) or ("<think>" in last and "</think>" not in last)


def pick_correct(written: str, want: set[str]) -> str:
    """The correct time closest to the written one (circular minutes), formatted like it (hour padding)."""
    h, m = _hm(written)
    best = min(want, key=lambda w: (min((abs((_hm(w)[0] * 60 + _hm(w)[1]) - (h * 60 + m))) % 1440,
                                        1440 - abs((_hm(w)[0] * 60 + _hm(w)[1]) - (h * 60 + m)) % 1440), w))
    bh, bm = _hm(best)
    hour = f"{bh:02d}" if len(written.split(":")[0]) == 2 else str(bh)
    return f"{hour}:{bm:02d}"


def first_diff(a: str, b: str) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def assistant_segments(text: str) -> list[tuple[int, int, str, int]]:
    """Char spans of the rendered conversation -> [(start, end, segment, turn)] over every assistant block
    (from after its header through its <|im_end|>). The last block's free text is the `answer`."""
    blocks = []
    pos = 0
    while (i := text.find(_ASSISTANT_HDR, pos)) >= 0:
        s = i + len(_ASSISTANT_HDR)
        e = text.find(_IM_END, s)
        e = len(text) if e < 0 else e
        blocks.append((s, e))
        pos = e
    spans = []
    for turn, (s, e) in enumerate(blocks):
        free = "answer" if turn == len(blocks) - 1 else "text"
        marks = sorted([(m.start(), m.end(), "think") for m in re.finditer(r"<think>.*?</think>", text[s:e], re.DOTALL)]
                       + [(m.start(), m.end(), "tool_call") for m in re.finditer(r"<tool_call>.*?</tool_call>", text[s:e], re.DOTALL)])
        cur = s
        for ms, me, seg in marks:
            if s + ms > cur:
                spans.append((cur, s + ms, free, turn))
            spans.append((s + ms, s + me, seg, turn))
            cur = s + me
        if e > cur:
            spans.append((cur, e, free, turn))
        if e < len(text):
            spans.append((e, e + len(_IM_END), "end", turn))
    return spans


def locate_time(text: str, spans, hm: tuple[int, int], turn: int, segment: str) -> tuple[int, str] | None:
    """First occurrence of time hm inside the given segment of the given turn -> (char index, as written)."""
    for s, e, seg, t in spans:
        if t == turn and seg == segment:
            for m in _TIME.finditer(text, s, e):
                if (int(m.group(1)), int(m.group(2))) == hm:
                    return m.start(), m.group(0)
    return None


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def decide(ep3: dict, s24: dict | None = None, corridor: dict | None = None) -> dict:
    """The pre-registered rule (module docstring). Returns the verdict + the numbers it rests on."""
    errs = [e for e in ep3["errors"] if e.get("p_err") is not None]
    med_p = _median([e["p_err"] for e in errs])
    med_pct = _median([e["percentile"] for e in errs])
    seg_err = {e["segment"] for e in errs}
    unreliable = sorted(s for s, c in ep3["segments"].items()
                        if c["n"] and c["rank_gt20"] / c["n"] > UNRELIABLE_RANK_SHARE)
    out = {"n_errors": len(errs), "n_tasks": len({e["task_id"] for e in errs}),
           "time_artefacts": ep3.get("population", {}).get("time_artefacts"),
           "unlocated_errors": sum(e.get("p_err") is None for e in ep3["errors"]), "median_p_err": med_p,
           "median_percentile": med_pct, "unreliable_segments": unreliable, "verdict": None, "enabled": False,
           "tau": None, "factor": 0.2, "selectivity": None, "selectivity_source": None, "reason": None,
           "s24": None}
    if s24:
        e24 = [e for e in s24["errors"] if e.get("p_err") is not None]
        out["s24"] = {"n_errors": len(e24), "median_p_err": _median([e["p_err"] for e in e24]),
                      "median_percentile": _median([e["percentile"] for e in e24])}
    if not errs:
        out.update(verdict="C", reason="no wrong-time error token found — nothing to aim at")
        return out
    if seg_err & set(unreliable):
        out.update(verdict="unreliable", reason=f"error tokens lie in segments with > {UNRELIABLE_RANK_SHARE:.1%} rank>{RANK_LIMIT} "
                                                f"tokens {sorted(seg_err & set(unreliable))} — re-render does not match the sampled text; ask")
        return out
    if med_p <= 0.6 and med_pct <= 0.05:
        tau = next((t for t in TAUS if sum(e["p_err"] < t for e in errs) >= 0.8 * len(errs)), None)
        sel_src = corridor or ep3
        if tau is not None:
            c = sel_src["failed_tokens"]
            out["selectivity"] = c["below"][str(tau)] / max(1, c["n"])
            out["selectivity_source"] = sel_src["label"]
        if tau is None:
            out.update(verdict="A", reason="dent, but no tau in 0.5..0.9 keeps >= 80 % of the error tokens below it")
        elif out["selectivity"] > 0.20:
            out.update(verdict="A", tau=tau, reason=f"dent, but not selective: {out['selectivity']:.1%} of the failed "
                                                    f"attempts' tokens lie below tau {tau} (> 20 %)")
        else:
            out.update(verdict="A", enabled=True, tau=tau,
                       reason=f"dent: median p_err {med_p:.3f}, median percentile {med_pct:.1%}; tau {tau} keeps "
                              f"{sum(e['p_err'] < tau for e in errs)}/{len(errs)} error tokens at full weight and down-weights "
                              f"{1 - out['selectivity']:.1%} of the failed attempts' tokens")
        return out
    if med_p >= 0.9:
        out.update(verdict="B", reason=f"confidently wrong: median p_err {med_p:.3f} — the weighting would spare the mistake")
        return out
    out.update(verdict="C", reason=f"neither dent nor confidently wrong: median p_err {med_p:.3f}, median percentile {med_pct:.1%}")
    return out


# --- measurement (torch) -----------------------------------------------------------------------------
def _sha16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()[:16]


def measure(model_dir: Path, traces: Path, label: str, out: Path, tasks_limit: int | None = None,
            task_regex: str | None = None, block: int = 1024) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from evaluation.eval_report import accepted
    from training_pipeline.collator_multiturn import build_masked_labels
    from training_pipeline.grpo_preflight import weights_fingerprint

    t0 = time.time()
    rows = [json.loads(line) for line in open(traces)]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
    pop = [t for t in sorted(by_task) if any(not accepted(r) for r in by_task[t])]
    if task_regex:                                    # test runs only — the diagnosis uses the whole population
        pop = [t for t in pop if re.search(task_regex, t)]
    if tasks_limit:
        pop = pop[:tasks_limit]
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to("cuda").eval()
    assert all(len(tok.encode(str(d), add_special_tokens=False)) == 1 for d in range(10)), "digits must be single tokens"
    digit_cols = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]

    seg = {s: {"n": 0, "rank_gt20": 0, "n_failed": 0, "below_failed": {str(t): 0 for t in TAUS}} for s in SEGMENTS}
    failed_tokens = {"n": 0, "below": {str(t): 0 for t in TAUS}}
    errors, attempts = [], []
    artefacts = truncated = 0

    @torch.no_grad()
    def score(rec):
        text = tok.apply_chat_template(rec["messages"], tokenize=False, add_generation_prompt=False)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        ref_ids, labels = build_masked_labels(tok, rec["messages"])
        assert ids == ref_ids, f"render mismatch vs collator for {rec['task_id']}#{rec.get('sample_idx')}"
        pos = [i for i, lab in enumerate(labels) if lab != -100]
        x = torch.tensor([ids], device="cuda")
        h = model.model(input_ids=x).last_hidden_state[0]
        tgt = torch.tensor([ids[i] for i in pos], device="cuda")
        prev = torch.tensor([i - 1 for i in pos], device="cuda")
        p, ent, rank, dig = [], [], [], []
        for b in range(0, len(pos), block):
            lg = model.lm_head(h[prev[b:b + block]]).float()
            lp = torch.log_softmax(lg, dim=-1)
            tl = lg.gather(1, tgt[b:b + block, None])
            p.append(lp.gather(1, tgt[b:b + block, None]).exp().squeeze(1))
            ent.append(-(lp.exp() * lp).sum(-1))
            rank.append((lg > tl).sum(-1) + 1)
            dig.append(lp[:, digit_cols].exp())                 # p of "0".."9" at every scored position
        spans = assistant_segments(text)
        segs = [next((sg for s, e, sg, _ in spans if s <= offs[i][0] < e), "end") for i in pos]
        return {"text": text, "ids": ids, "offs": offs, "pos": pos, "p": torch.cat(p).tolist(),
                "ent": torch.cat(ent).tolist(), "rank": torch.cat(rank).tolist(), "seg": segs, "spans": spans,
                "digits": torch.cat(dig).cpu().tolist()}

    def token_at(sc, char):
        for k, i in enumerate(sc["pos"]):
            s, e = sc["offs"][i]
            if s <= char < e:
                return k, i
        return None, None

    for ti, task in enumerate(pop):
        scored = []
        for rec in by_task[task]:
            sc = score(rec)
            ok = accepted(rec)
            trunc = is_truncated(rec)
            truncated += trunc
            for k in range(len(sc["pos"]) if not trunc else 0):   # truncated: template-injected think, see docstring
                c = seg[sc["seg"][k]]
                c["n"] += 1
                c["rank_gt20"] += sc["rank"][k] > RANK_LIMIT
                if not ok:
                    c["n_failed"] += 1
                    failed_tokens["n"] += 1
                    for t in TAUS:
                        below = sc["p"][k] < t
                        c["below_failed"][str(t)] += below
                        failed_tokens["below"][str(t)] += below
            attempts.append({"task_id": task, "sample_idx": rec.get("sample_idx"), "accepted": ok, "truncated": trunc,
                             "n_tokens": len(sc["pos"]), "mean_p": sum(sc["p"]) / max(1, len(sc["p"])),
                             "rank_gt20": sum(r > RANK_LIMIT for r in sc["rank"])})
            scored.append((rec, sc, ok))
        last_turn = lambda sc: max((t for *_, t in sc["spans"]), default=0)  # noqa: E731
        for rec, sc, ok in scored:
            tv = time_verdict(rec)
            artefacts += bool(tv) and tv["kind"] == "artefact"
            if not tv or tv["kind"] != "wrong":
                continue
            bad, want = tv["written"], tv["want"] | tv["wrap"]   # reference: the sum or the wrapped planned time
            hit, where = None, None
            for segment in ("think", "answer"):
                hit = locate_time(sc["text"], sc["spans"], _hm(bad), last_turn(sc), segment)
                if hit:
                    where = segment
                    break
            err = {"task_id": task, "sample_idx": rec.get("sample_idx"), "template": rec["template"], "wrong": bad,
                   "correct_times": sorted(want), "segment": where, "p_err": None}
            if hit:
                char, written = hit
                corr = pick_correct(written, want)
                d = first_diff(written, corr)
                err.update(written=written, correct=corr, digit_index=d)
                if d is not None:
                    k, i = token_at(sc, char + d)
                    if k is not None:
                        p_err = sc["p"][k]
                        err.update(token=tok.decode([sc["ids"][i]]), p_err=p_err, rank=sc["rank"][k],
                                   entropy=sc["ent"][k], percentile=sum(x <= p_err for x in sc["p"]) / len(sc["p"]),
                                   p_correct=sc["digits"][k][int(corr[d])] if d < len(corr) and corr[d].isdigit() else None)
                        # siblings: accepted attempts of the task that write the correct time — p of the same
                        # digit there (aligned from the right: hour padding may differ, the minutes do not)
                        sib = []
                        for r2, sc2, ok2 in scored:
                            if r2 is rec or not ok2:
                                continue
                            h2 = (locate_time(sc2["text"], sc2["spans"], _hm(corr), last_turn(sc2), "think")
                                  or locate_time(sc2["text"], sc2["spans"], _hm(corr), last_turn(sc2), "answer"))
                            if h2 and 0 <= d + len(h2[1]) - len(corr) < len(h2[1]):
                                k2, _ = token_at(sc2, h2[0] + d + len(h2[1]) - len(corr))
                                if k2 is not None:
                                    sib.append(sc2["p"][k2])
                        err["siblings_p_correct"] = sib
            errors.append(err)
        print(f"  [{ti + 1}/{len(pop)}] {task}: {len(scored)} attempts, errors so far {len(errors)}", flush=True)

    res = {"label": label, "model": str(model_dir), "model_weights_fp": weights_fingerprint(Path(model_dir)),
           "traces": str(traces), "traces_sha256_16": _sha16(traces), "temperature": 1.0, "taus": list(TAUS),
           "population": {"tasks": len(pop), "attempts": len(attempts),
                          "failed_attempts": sum(not a["accepted"] for a in attempts),
                          "truncated_excluded": truncated, "time_artefacts": artefacts},
           "segments": seg, "failed_tokens": failed_tokens, "errors": errors, "attempts": attempts,
           "render_asserted": len(attempts), "seconds": round(time.time() - t0)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res


def _print_measure(res: dict) -> None:
    pop = res["population"]
    print(f"\n{res['label']}: {pop['tasks']} tasks, {pop['attempts']} attempts ({pop['failed_attempts']} failed; "
          f"{pop.get('truncated_excluded', 0)} truncated left out of the token counts; {pop.get('time_artefacts', 0)} "
          f"wrap artefacts not counted as errors), "
          f"render == collator for all, {res['seconds']}s")
    for s, c in res["segments"].items():
        if c["n"]:
            print(f"  {s:9s} tokens {c['n']:7d}  rank>{RANK_LIMIT} {c['rank_gt20'] / c['n']:.3%}  "
                  + " ".join(f"p<{t}: {c['below_failed'][str(t)] / max(1, c['n_failed']):.1%}" for t in TAUS) + " (failed attempts)")
    for e in res["errors"]:
        print(f"  error {e['task_id']}#{e['sample_idx']}: {e.get('written')} vs {e.get('correct')} in {e['segment']}: "
              f"p_err {e['p_err']}, p_correct {e.get('p_correct')}, rank {e.get('rank')}, pct {e.get('percentile')}")


def main() -> None:
    ap = argparse.ArgumentParser(description="confidence diagnosis: measure (GPU) / decide (host)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--model", type=Path, required=True)
    m.add_argument("--traces", type=Path, required=True)
    m.add_argument("--label", required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--tasks-limit", type=int, default=None, help="test runs: first N population tasks")
    m.add_argument("--task-regex", default=None, help="test runs: population tasks matching this regex")
    d = sub.add_parser("decide")
    d.add_argument("--ep3", type=Path, required=True)
    d.add_argument("--s24", type=Path, default=None)
    d.add_argument("--corridor", type=Path, default=None)
    d.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "measure":
        _print_measure(measure(a.model, a.traces, a.label, a.out, a.tasks_limit, a.task_regex))
        return
    load = lambda p: json.loads(p.read_text()) if p else None  # noqa: E731
    res = decide(load(a.ep3), load(a.s24), load(a.corridor))
    print(json.dumps(res, indent=1))
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
