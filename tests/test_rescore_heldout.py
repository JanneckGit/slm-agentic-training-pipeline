"""
tests/test_rescore_heldout.py
=============================
evaluation/rescore_heldout.py on a two-record scratch file (GPU-free, tau2 venv + db_bahn pool like
tests/test_trajectory_reward.py): a stale score is replaced, an unchanged line stays byte-identical,
messages are untouched, the original lands in the backup dir and the sidecar records the rescore outside
`identity`. A second run finds nothing to change and writes nothing.
"""
import json

from evaluation import rescore_heldout as rh
from evaluation.trajectory_reward import score_trajectory
from tests.db_bahn_helpers import copy, oracle_trace


def test_rescore_replaces_only_stale_scores(dom, tasks, keys, pick, tmp_path):
    tid = pick("t_info_crew")
    task, key = tasks[tid], keys[tid]
    comm = task["evaluation_criteria"]["communicate_info"]
    assert any(" " in c for c in comm), "case needs a communicate item with a space"
    good_msgs = oracle_trace(dom, task, key) + [{"role": "assistant", "content": "Ergebnis: " + "; ".join(comm) + "."}]
    score = score_trajectory(task, good_msgs, key)
    assert score["score"] == 1.0, f"precondition: the oracle answer passes: {score}"
    typo_msgs = copy(good_msgs)
    typo_msgs[-1]["content"] = typo_msgs[-1]["content"].replace(" ", " ")
    stale = json.loads(json.dumps(score))   # what the pre-fix verifier said about the U+202F answer
    stale.update(score=0.0, task_solved=0.0)
    stale["components"]["communicate"] = False

    rec = {"task_id": tid, "template": key["template"], "truncated": False, "degen": {}}
    lines = [json.dumps({**rec, "sample_idx": 0, "score": score, "messages": good_msgs}, ensure_ascii=False) + "\n",
             json.dumps({**rec, "sample_idx": 1, "score": stale, "messages": typo_msgs}, ensure_ascii=False) + "\n"]
    f = tmp_path / "db_traces_heldout_eval_x_k8.jsonl"
    f.write_text("".join(lines))
    meta_p = tmp_path / (f.name + ".meta.json")
    meta_p.write_text(json.dumps({"identity": {"label": "x", "k": 8}, "info": {}}))
    args = [str(f), "--reason", "test", "--backup-dir", str(tmp_path / "bak"), "--workers", "1"]

    assert rh.main(args + ["--dry-run"]) == 0
    assert f.read_text() == "".join(lines), "--dry-run must not write"

    assert rh.main(args) == 0
    out = f.read_text().splitlines(keepends=True)
    assert out[0] == lines[0], "an unchanged record must stay byte-identical"
    fixed = json.loads(out[1])
    assert fixed["score"]["score"] == 1.0 and fixed["score"]["components"]["communicate"] is True
    assert fixed["messages"] == typo_msgs, "messages must stay exactly as generated"
    assert (tmp_path / "bak" / f.name).read_text() == "".join(lines), "backup = the original file"
    meta = json.loads(meta_p.read_text())
    assert meta["identity"] == {"label": "x", "k": 8}, "identity (resume check) must stay untouched"
    entry = meta["rescored"][0]
    assert (entry["reason"], entry["changed_records"], entry["accepted_before"], entry["accepted_after"]) \
        == ("test", 1, 1, 2), entry

    assert rh.main(args) == 0, "second run: nothing stale left"
    assert len(json.loads(meta_p.read_text())["rescored"]) == 1, "no change -> no new sidecar entry"
