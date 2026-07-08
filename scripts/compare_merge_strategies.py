"""Compare 3 merge strategies for combining retrieval sources.

Loads the actual pipeline prediction file, then for each conversation:
- Checks what BM25+BERT found (existing results)
- Runs i2i + artist shortcut independently
- Tests 3 merge strategies:
  A: Pool expansion (append new, don't reorder existing)
  B: Rule-based routing (activate sources based on conversation signals)
  C: Weighted pool (BM25+BERT primary, new sources secondary with low weight)

All share the same loaded data — no extra GPU/memory cost.
"""
import ast, json, math, hashlib
from pathlib import Path
from collections import Counter, defaultdict

import torch
from datasets import load_dataset, concatenate_datasets


def fmt(v):
    if v is None: return ""
    if isinstance(v, (list, tuple, set)): return ", ".join(str(i) for i in v)
    s = str(v)
    if s.startswith("[") and s.endswith("]"):
        try:
            p = ast.literal_eval(s)
            if isinstance(p, list): return ", ".join(str(i) for i in p)
        except: pass
    return s

def norm(v): return str(v).strip() if v else ""


def ndcg_at_k(predicted, target, k):
    for idx, tid in enumerate(predicted[:k]):
        if tid == target:
            return 1.0 / math.log2(idx + 2)
    return 0.0


def recall_at_k(predicted, target, k):
    return 1.0 if target in predicted[:k] else 0.0


def i2i(anchors, edata, topk=100):
    vecs = [edata["emb"][norm(a)] for a in anchors if norm(a) in edata["emb"]]
    if not vecs: return []
    q = torch.stack(vecs).mean(0)
    n = torch.linalg.norm(q)
    if n > 0: q = q / n
    sc = torch.matmul(edata["mat"], q)
    ti = torch.topk(sc, k=min(topk, len(edata["tids"]))).indices.tolist()
    skip = {norm(a) for a in anchors}
    return [edata["tids"][i] for i in ti if edata["tids"][i] not in skip][:topk]


FOLLOWUP_KW = {"more", "similar", "another", "like this", "same vibe", "same kind",
               "same feel", "same style", "same energy", "that kind", "like that"}
PIVOT_KW = {"different", "new artist", "branch out", "other artist", "other band",
            "switch", "something else", "explore", "discover"}


def detect_intent(user_query, anchors_share_artist):
    q = user_query.lower()
    is_followup = any(kw in q for kw in FOLLOWUP_KW)
    is_pivot = any(kw in q for kw in PIVOT_KW)
    return {
        "same_artist": anchors_share_artist,
        "followup": is_followup,
        "pivot": is_pivot,
    }


def main():
    # Load predictions from actual pipeline run
    pred_file = Path("exp/inference/devset/qwen3_8b_multi_source_devset.json")
    predictions = json.loads(pred_file.read_text())
    pred_map = {}
    for row in predictions:
        key = (row["session_id"], int(row["turn_number"]))
        pred_map[key] = row.get("predicted_track_ids", [])

    mini_ids = set(json.loads(Path("config/mini_devset_session_ids.json").read_text()))
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")
    mds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    tm = {}
    for s in mds.keys():
        for r in mds[s]: tm[r["track_id"]] = r

    # Artist index
    ai = defaultdict(list)
    for tid, m in tm.items():
        a = fmt(m.get("artist_name", "")).lower().strip()
        if a: ai[a].append(tid)

    print("Loading embeddings...")
    eds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")
    emg = concatenate_datasets([eds[s] for s in eds.keys()])
    etypes = ["image-siglip2", "cf-bpr"]
    edata = {e: {"emb": {}, "tids": [], "mat": None} for e in etypes}
    for row in emg:
        tid = norm(row["track_id"])
        for e in etypes:
            v = row.get(e)
            if v and len(v) > 0:
                t = torch.tensor(v, dtype=torch.float32)
                n = torch.linalg.norm(t)
                if n > 0: t = t / n
                edata[e]["emb"][tid] = t
    for e in etypes:
        tids = list(edata[e]["emb"].keys())
        edata[e]["tids"] = tids
        edata[e]["mat"] = torch.stack([edata[e]["emb"][t] for t in tids]) if tids else torch.empty(0, 0)
        print(f"  {e}: {len(tids)}")

    # Evaluate all strategies
    strategies = {
        "baseline": [],      # current pipeline as-is
        "A_pool_expand": [], # append new sources, don't reorder
        "B_routed": [],      # rule-based routing
        "C_weighted": [],    # low-weight secondary merge
    }

    for item in ds:
        if item["session_id"] not in mini_ids: continue
        convs = item["conversations"]
        goal = item.get("conversation_goal") or {}
        spec = str(goal.get("specificity", "")).strip().upper()

        # Find last music turn
        lm = None
        for t in reversed(convs):
            if t["role"] == "music": lm = t; break
        if not lm: continue

        target = norm(lm["content"])
        ttn = int(lm["turn_number"])
        ctx = [t for t in convs if int(t["turn_number"]) < ttn]

        user_query = ""
        for t in reversed(ctx):
            if t["role"] == "user": user_query = t["content"]; break

        # Get anchors
        anchors = []
        for t in reversed(ctx):
            if t["role"] == "music":
                tid = norm(t["content"])
                if tid in tm and tid not in anchors: anchors.append(tid)
                if len(anchors) >= 3: break

        # Current pipeline results (from prediction file)
        baseline_results = pred_map.get((item["session_id"], ttn), [])
        tname = fmt(tm.get(target, {}).get("track_name", "")) + " by " + fmt(tm.get(target, {}).get("artist_name", ""))

        # Get anchor artists
        anchor_artists = [fmt(tm.get(a, {}).get("artist_name", "")).lower().strip() for a in anchors]
        ac = Counter(a for a in anchor_artists if a)
        anchors_share_artist = ac and ac.most_common(1)[0][1] >= 2
        dominant_artist = ac.most_common(1)[0][0] if anchors_share_artist else None

        # Get new source candidates
        artist_pool = []
        if anchors_share_artist and dominant_artist:
            artist_pool = [t for t in ai.get(dominant_artist, []) if t not in set(anchors)][:100]

        i2i_image = i2i(anchors, edata["image-siglip2"], topk=100)
        i2i_cfbpr = i2i(anchors, edata["cf-bpr"], topk=100)

        intent = detect_intent(user_query, anchors_share_artist)

        # === Strategy: Baseline (current pipeline) ===
        strategies["baseline"].append({
            "results": baseline_results, "target": target, "spec": spec, "tname": tname
        })

        # === Strategy A: Pool expansion ===
        pool_a = list(baseline_results)  # preserve order
        seen = set(pool_a)
        # Append artist shortcut first (high priority when applicable)
        for tid in artist_pool:
            if tid not in seen:
                pool_a.append(tid)
                seen.add(tid)
        # Then i2i candidates
        for tid in i2i_image:
            if tid not in seen:
                pool_a.append(tid)
                seen.add(tid)
        for tid in i2i_cfbpr:
            if tid not in seen:
                pool_a.append(tid)
                seen.add(tid)
        strategies["A_pool_expand"].append({
            "results": pool_a, "target": target, "spec": spec, "tname": tname
        })

        # === Strategy B: Rule-based routing ===
        pool_b = list(baseline_results)
        seen_b = set(pool_b)
        if intent["same_artist"] and artist_pool:
            # Same artist detected: insert artist pool near top
            # Insert after position 20 (keep BM25 top-20 intact)
            insert_pos = min(20, len(pool_b))
            for tid in artist_pool:
                if tid not in seen_b:
                    pool_b.insert(insert_pos, tid)
                    seen_b.add(tid)
                    insert_pos += 1
        if intent["followup"] and not intent["pivot"]:
            # "More like this" → image-siglip2 (visual/vibe similarity)
            for tid in i2i_image:
                if tid not in seen_b:
                    pool_b.append(tid)
                    seen_b.add(tid)
        if intent["pivot"]:
            # "Different artist" → cf-bpr (collaborative filtering)
            for tid in i2i_cfbpr:
                if tid not in seen_b:
                    pool_b.append(tid)
                    seen_b.add(tid)
        if not intent["same_artist"] and not intent["followup"] and not intent["pivot"]:
            # Generic: add both i2i at end
            for tid in i2i_image + i2i_cfbpr:
                if tid not in seen_b:
                    pool_b.append(tid)
                    seen_b.add(tid)
        strategies["B_routed"].append({
            "results": pool_b, "target": target, "spec": spec, "tname": tname
        })

        # === Strategy C: Weighted secondary merge ===
        # BM25+BERT get score 1.0, new sources get 0.3 (secondary)
        rrf_k = 60
        scores = {}
        for rank, tid in enumerate(baseline_results):
            scores[tid] = scores.get(tid, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, tid in enumerate(artist_pool):
            scores[tid] = scores.get(tid, 0.0) + 0.5 / (rrf_k + rank + 1)
        for rank, tid in enumerate(i2i_image):
            scores[tid] = scores.get(tid, 0.0) + 0.2 / (rrf_k + rank + 1)
        for rank, tid in enumerate(i2i_cfbpr):
            scores[tid] = scores.get(tid, 0.0) + 0.15 / (rrf_k + rank + 1)
        ranked_c = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        pool_c = [tid for tid, _ in ranked_c[:400]]
        strategies["C_weighted"].append({
            "results": pool_c, "target": target, "spec": spec, "tname": tname
        })

    # === Report ===
    print("\n" + "=" * 80)
    print("MERGE STRATEGY COMPARISON")
    print("=" * 80)

    for sname in ["baseline", "A_pool_expand", "B_routed", "C_weighted"]:
        data = strategies[sname]
        n = len(data)
        print(f"\n## {sname}")
        for k in [1, 5, 10, 20, 100, 200]:
            r = sum(recall_at_k(d["results"], d["target"], k) for d in data) / n
            nd = sum(ndcg_at_k(d["results"], d["target"], k) for d in data) / n
            print(f"  recall@{k:<4} {r:.3f}   nDCG@{k:<4} {nd:.3f}")

        # By specificity
        for sp in ["HH", "HL", "LH", "LL"]:
            sp_data = [d for d in data if d["spec"] == sp]
            if not sp_data: continue
            r20 = sum(recall_at_k(d["results"], d["target"], 20) for d in sp_data) / len(sp_data)
            r200 = sum(recall_at_k(d["results"], d["target"], 200) for d in sp_data) / len(sp_data)
            nd20 = sum(ndcg_at_k(d["results"], d["target"], 20) for d in sp_data) / len(sp_data)
            print(f"  [{sp}] recall@20={r20:.3f} recall@200={r200:.3f} nDCG@20={nd20:.3f}")

    # === Show what each strategy uniquely rescues ===
    print("\n" + "=" * 80)
    print("UNIQUE RESCUES (found by strategy but NOT by baseline)")
    print("=" * 80)

    baseline_data = strategies["baseline"]
    for sname in ["A_pool_expand", "B_routed", "C_weighted"]:
        data = strategies[sname]
        rescues = []
        for i in range(len(data)):
            b_hit = data[i]["target"] in baseline_data[i]["results"][:200]
            s_hit = data[i]["target"] in data[i]["results"][:200]
            if s_hit and not b_hit:
                rank = data[i]["results"].index(data[i]["target"]) + 1
                rescues.append({"rank": rank, "spec": data[i]["spec"], "tname": data[i]["tname"]})
        rescues.sort(key=lambda x: x["rank"])
        print(f"\n  {sname}: {len(rescues)} new finds")
        for r in rescues[:15]:
            print(f"    [{r['spec']}] rank={r['rank']:>3} {r['tname'][:60]}")

    # === Regressions (baseline found but strategy lost) ===
    print("\n" + "=" * 80)
    print("REGRESSIONS (baseline found@200 but strategy lost)")
    print("=" * 80)
    for sname in ["A_pool_expand", "B_routed", "C_weighted"]:
        data = strategies[sname]
        regressions = []
        for i in range(len(data)):
            b_hit = data[i]["target"] in baseline_data[i]["results"][:200]
            s_hit = data[i]["target"] in data[i]["results"][:200]
            if b_hit and not s_hit:
                regressions.append(data[i])
        print(f"  {sname}: {len(regressions)} regressions")


if __name__ == "__main__":
    main()
