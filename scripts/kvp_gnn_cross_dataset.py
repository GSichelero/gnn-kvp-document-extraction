#!/usr/bin/env python3
"""
GNN Cross-Dataset Experiments v2
=================================
Uses the actual EnhancedQAGNN architecture (matching the checkpoint that
achieved F1=0.747 on FUNSD) for proper cross-dataset evaluation.

Produces two 3x3 matrices:
  - Proposed QA-GNN (EnhancedQAGNN)
  - Doc2Graph (baseline)

Usage:
    python scripts/kvp_gnn_cross_dataset.py
"""

import json, os, sys, time, gc, shutil, copy, re, math
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
from collections import Counter, defaultdict

# Paths. Override DOCUMENT_KVP_PROJECT_ROOT/KVP_DATASETS_DIR/KVP_RESULTS_DIR
# when datasets or external baselines live outside this repository.
SCRIPT_DIR = Path(__file__).resolve().parent
CONTRIB_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = Path(os.environ.get("DOCUMENT_KVP_PROJECT_ROOT", CONTRIB_ROOT)).resolve()
DATASETS_DIR = Path(os.environ.get("KVP_DATASETS_DIR", PROJECT_ROOT / "kvp_comparison" / "datasets")).resolve()
RESULTS_DIR = Path(os.environ.get("KVP_RESULTS_DIR", CONTRIB_ROOT / "outputs" / "gnn_cross_dataset")).resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FUNSD_DIR    = DATASETS_DIR / "FUNSD"
WR_RAW_DIR   = DATASETS_DIR / "wildreceipt"
WR_FUNSD_DIR = DATASETS_DIR / "WR_FUNSD_full"

DOC2GRAPH_ROOT = (PROJECT_ROOT / "extractors" / "key-value-pairs-extractors"
                  / "doc2graph")

sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import dgl
    DGL_OK = True
except ImportError:
    DGL_OK = False
    print("WARNING: DGL not available")

# ======================================================================
# 1. WildReceipt -> FUNSD conversion  (reuse cached data)
# ======================================================================


def _wr_role_and_category(label_id):
    """Map the WildReceipt odd/even label convention to FUNSD roles."""
    if not isinstance(label_id, int) or label_id <= 0:
        return "other", None
    if label_id % 2 == 0:
        return "question", label_id // 2
    return "answer", (label_id + 1) // 2


def _load_wr_jsonl(path, max_samples=None):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
                if max_samples and len(samples) >= max_samples:
                    break
    return samples


def _convert_wr_sample(sample, wr_base, out_ann_dir, out_img_dir, idx,
                       max_img_size=800):
    img_path = wr_base / sample["file_name"]
    if not img_path.exists():
        return False
    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    if max(w, h) > max_img_size:
        ratio = max_img_size / max(w, h)
        nw, nh = int(w*ratio), int(h*ratio)
        image = image.resize((nw, nh), Image.LANCZOS)
        sx, sy = nw/w, nh/h
    else:
        sx, sy = 1.0, 1.0; nw, nh = w, h

    entities = []
    # Track entity IDs grouped by WR category for proper pairing
    # WR scheme: key label N pairs with value label N-1 (same category)
    cat_keys = defaultdict(list)    # category -> list of entity ids (keys)
    cat_values = defaultdict(list)  # category -> list of entity ids (values)
    eid = 0
    for ann in sample["annotations"]:
        text = ann["text"].strip()
        if not text: continue
        poly = ann["box"]
        xs = [poly[i]*sx for i in range(0,8,2)]
        ys = [poly[i]*sy for i in range(1,8,2)]
        box = [max(0,int(min(xs))), max(0,int(min(ys))),
               min(nw,int(max(xs))), min(nh,int(max(ys)))]
        label, cat = _wr_role_and_category(ann["label"])
        if label == "question":
            cat_keys[cat].append(eid)
        elif label == "answer":
            cat_values[cat].append(eid)
        entities.append({
            "id": eid, "text": text, "label": label, "box": box,
            "words": [{"text": text, "box": box}], "linking": []
        })
        eid += 1

    # Pair keys -> values by category + spatial proximity
    def _bbox_center(b):
        return ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0)

    for cat in set(cat_keys.keys()) | set(cat_values.keys()):
        k_ids = cat_keys.get(cat, [])
        v_ids = list(cat_values.get(cat, []))
        if not k_ids or not v_ids:
            continue
        used_values = set()
        for kid in k_ids:
            k_cx, k_cy = _bbox_center(entities[kid]["box"])
            best_vid, best_dist = None, float("inf")
            for vid in v_ids:
                if vid in used_values:
                    continue
                v_cx, v_cy = _bbox_center(entities[vid]["box"])
                d = math.sqrt((k_cx - v_cx)**2 + (k_cy - v_cy)**2)
                if d < best_dist:
                    best_dist = d
                    best_vid = vid
            if best_vid is not None:
                entities[kid]["linking"].append([kid, best_vid])
                used_values.add(best_vid)

    labels_present = {e["label"] for e in entities}
    if "header" not in labels_present and entities:
        for e in sorted(entities, key=lambda x: x["box"][1]):
            if e["label"] == "other":
                e["label"] = "header"; break
        else:
            entities.append({
                "id": eid, "text": "RECEIPT", "label": "header",
                "box": [0,0,10,10],
                "words": [{"text":"RECEIPT","box":[0,0,10,10]}], "linking": []
            })

    sid = f"wr_{idx:05d}"
    with open(out_ann_dir / f"{sid}.json", "w", encoding="utf-8") as f:
        json.dump({"form": entities}, f)
    image.save(out_img_dir / f"{sid}.png")
    return True


def convert_wildreceipt(max_train=200, max_test=50):
    if WR_FUNSD_DIR.exists():
        tr = WR_FUNSD_DIR / "training_data" / "annotations"
        te = WR_FUNSD_DIR / "testing_data" / "annotations"
        if tr.exists() and te.exists():
            nt = len(list(tr.glob("*.json")))
            ne = len(list(te.glob("*.json")))
            if nt >= max_train and ne >= max_test:
                print(f"  WR->FUNSD already cached ({nt} train, {ne} test)")
                return
        shutil.rmtree(WR_FUNSD_DIR)
    for split, jname, limit in [("training_data","train.txt",max_train),
                                 ("testing_data","test.txt",max_test)]:
        ann = WR_FUNSD_DIR / split / "annotations"
        img = WR_FUNSD_DIR / split / "images"
        ann.mkdir(parents=True, exist_ok=True)
        img.mkdir(parents=True, exist_ok=True)
        samples = _load_wr_jsonl(WR_RAW_DIR / jname, limit)
        ok = 0
        for i, s in enumerate(samples):
            if _convert_wr_sample(s, WR_RAW_DIR, ann, img, i):
                ok += 1
        print(f"    {split}: {ok} samples")


# ======================================================================
# 2. EnhancedQAGNN model (exact copy from enhanced_qa_gnn.py)
# ======================================================================

QUESTION_WORDS = {
    'what','when','where','who','why','how','which','whose',
    'name','date','time','number','amount','type','kind','subject'
}
LABEL2ID = {'other':0, 'header':1, 'question':2, 'answer':3}


def _extract_text_features(text):
    """21-dim text features matching EnhancedFUNSDProcessor."""
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    basic = [
        len(text), len(words), len(text.split()),
        text.count(':'), text.count('?'), text.count('.'),
        1.0 if text.isupper() else 0.0,
        1.0 if text.istitle() else 0.0,
        1.0 if any(c.isdigit() for c in text) else 0.0,
    ]
    question = [
        1.0 if any(qw in text_lower for qw in QUESTION_WORDS) else 0.0,
        sum(1.0 for qw in QUESTION_WORDS if qw in text_lower) / len(QUESTION_WORDS),
    ]
    pattern = [
        1.0 if re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text) else 0.0,
        1.0 if re.search(r'\$\d+', text) else 0.0,
        1.0 if re.search(r'\b\d+\b', text) else 0.0,
        1.0 if re.search(r'\b[A-Z]{2,}\b', text) else 0.0,
    ]
    vocab = []
    for w in ['name','date','number','type','total','amount']:
        vocab.append(1.0 if w in text_lower else 0.0)
    return torch.tensor(basic + question + pattern + vocab, dtype=torch.float32)


def _spatial_features(bbox1, bbox2, W=1000, H=1000):
    """15-dim spatial features matching EnhancedFUNSDProcessor."""
    x1_1,y1_1,x2_1,y2_1 = [v/W if i%2==0 else v/H for i,v in enumerate(bbox1)]
    x1_2,y1_2,x2_2,y2_2 = [v/W if i%2==0 else v/H for i,v in enumerate(bbox2)]
    cx1,cy1 = (x1_1+x2_1)/2, (y1_1+y2_1)/2
    cx2,cy2 = (x1_2+x2_2)/2, (y1_2+y2_2)/2
    dx, dy = cx2-cx1, cy2-cy1
    dist = math.sqrt(dx**2+dy**2)
    w1,h1 = x2_1-x1_1, y2_1-y1_1
    w2,h2 = x2_2-x1_2, y2_2-y1_2
    a1,a2 = w1*h1, w2*h2
    ox = max(0, min(x2_1,x2_2)-max(x1_1,x1_2))
    oy = max(0, min(y2_1,y2_2)-max(y1_1,y1_2))
    oa = ox*oy
    iou = oa/(a1+a2-oa) if (a1+a2-oa)>0 else 0
    return torch.tensor([
        dx, dy, dist, abs(dx), abs(dy),
        1.0 if dx>0 else 0.0,
        1.0 if dy>0 else 0.0,
        1.0 if abs(dy)<0.02 else 0.0,
        1.0 if abs(dx)<0.02 else 0.0,
        a1, a2,
        a2/a1 if a1>0 else 0,
        iou,
        1.0 if dist<0.1 else 0.0,
        1.0 if dist<0.05 else 0.0,
    ], dtype=torch.float32)


def build_enhanced_graph(entities):
    """Build graph with strategic edges (Q->A, Q->Other, Q/Header->A/Other).
    Produces 34-dim node features and 15-dim edge features, matching the
    original EnhancedQAGNN checkpoint."""
    n = len(entities)
    if n < 2:
        return None

    qa_set = set()
    for e in entities:
        if e["label"] == "question":
            for lnk in e.get("linking", []):
                if len(lnk) == 2:
                    sid, tid = lnk
                    tgt = next((x for x in entities if x["id"] == tid), None)
                    if tgt and tgt["label"] == "answer":
                        qa_set.add((sid, tid))

    nfeats = []
    for e in entities:
        lbl = torch.zeros(4); lbl[LABEL2ID.get(e["label"],0)] = 1.0
        x1,y1,x2,y2 = e["box"]
        bbox = torch.tensor([
            x1/1000, y1/1000, x2/1000, y2/1000,
            (x2-x1)/1000, (y2-y1)/1000,
            (x1+x2)/2000, (y1+y2)/2000,
            ((x2-x1)*(y2-y1))/1000000,
        ], dtype=torch.float32)
        tf = _extract_text_features(e.get("text",""))
        nfeats.append(torch.cat([lbl, bbox, tf]))  # 4+9+21 = 34

    srcs, dsts, elabels, efeats = [], [], [], []
    for i, ei in enumerate(entities):
        for j, ej in enumerate(entities):
            if i == j: continue
            should = (
                (ei["label"]=="question" and ej["label"]=="answer") or
                (ei["label"]=="question" and ej["label"]=="other") or
                (ei["label"] in ["question","header"] and
                 ej["label"] in ["answer","other"])
            )
            if not should: continue
            srcs.append(i); dsts.append(j)
            is_qa = (ei["id"], ej["id"]) in qa_set
            elabels.append(1 if is_qa else 0)
            efeats.append(_spatial_features(ei["box"], ej["box"]))

    if not srcs:
        # Fallback
        srcs, dsts = [0], [min(1, n-1)]
        elabels = [0]
        efeats = [torch.zeros(15)]

    g = dgl.graph((srcs, dsts))
    g.ndata["feat"]  = torch.stack(nfeats)
    g.edata["feat"]  = torch.stack(efeats)
    g.edata["label"] = torch.tensor(elabels, dtype=torch.long)
    return g


def load_graphs(ann_dir):
    """Load FUNSD-format annotation dir -> list of DGL graphs (enhanced features)."""
    ann_dir = Path(ann_dir)
    graphs = []
    for fp in sorted(ann_dir.glob("*.json")):
        try:
            data = json.load(open(fp, encoding="utf-8"))
            ents = data.get("form", data.get("entities", []))
            g = build_enhanced_graph(ents)
            if g is not None and g.number_of_edges() > 0:
                graphs.append(g)
        except Exception:
            continue
    return graphs


# ---- EnhancedGraphLayer ----
class EnhancedGraphLayer(nn.Module):
    def __init__(self, hdim):
        super().__init__()
        self.W_msg = nn.Sequential(
            nn.Linear(hdim*3, hdim*2), nn.LayerNorm(hdim*2), nn.ReLU(),
            nn.Linear(hdim*2, hdim))
        self.W_node = nn.Sequential(
            nn.Linear(hdim*2, hdim), nn.LayerNorm(hdim), nn.ReLU())
        self.W_edge = nn.Sequential(
            nn.Linear(hdim*3, hdim), nn.LayerNorm(hdim), nn.ReLU())
        self.drop = nn.Dropout(0.1)

    def forward(self, g, h, e):
        with g.local_scope():
            g.ndata["h"] = h; g.edata["e"] = e
            g.apply_edges(lambda ed: {
                "m": self.W_msg(torch.cat([ed.src["h"], ed.dst["h"], ed.data["e"]],1))
            })
            g.update_all(dgl.function.copy_e("m","m"), dgl.function.mean("m","agg"))
            h2 = self.drop(self.W_node(torch.cat([h, g.ndata["agg"]],1)))
            g.ndata["h"] = h2
            g.apply_edges(lambda ed: {
                "e2": self.drop(self.W_edge(torch.cat([ed.data["e"], ed.src["h"], ed.dst["h"]],1)))
            })
            return h2, g.edata["e2"]


# ---- EnhancedQAGNN ----
class EnhancedQAGNN(nn.Module):
    def __init__(self, nfeat, efeat, hdim=128, nlayers=3):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(nfeat, hdim), nn.LayerNorm(hdim), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(hdim, hdim))
        self.edge_encoder = nn.Sequential(
            nn.Linear(efeat, hdim), nn.LayerNorm(hdim), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(hdim, hdim))
        self.gnn_layers = nn.ModuleList([EnhancedGraphLayer(hdim) for _ in range(nlayers)])
        self.attention = nn.MultiheadAttention(hdim, num_heads=4, dropout=0.1, batch_first=True)
        self.edge_classifier = nn.Sequential(
            nn.Linear(hdim*3, hdim*2), nn.LayerNorm(hdim*2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hdim*2, hdim), nn.LayerNorm(hdim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hdim, hdim//2), nn.ReLU(), nn.Linear(hdim//2, 2))
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: m.bias.data.fill_(0.01)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_(); m.weight.data.fill_(1.0)

    def forward(self, g):
        h = self.node_encoder(g.ndata["feat"])
        e = self.edge_encoder(g.edata["feat"])
        h0 = h
        for i, layer in enumerate(self.gnn_layers):
            hn, en = layer(g, h, e)
            h = hn + (h0 if i==0 else h)
            e = en + e
        # Attention (per-graph not batched -- apply to all nodes)
        ha = h.unsqueeze(0)
        ha, _ = self.attention(ha, ha, ha)
        h = ha.squeeze(0)
        with g.local_scope():
            g.ndata["h"] = h; g.edata["e"] = e
            g.apply_edges(lambda ed: {
                "pred": self.edge_classifier(
                    torch.cat([ed.src["h"], ed.dst["h"], ed.data["e"]],1))
            })
            return g.edata["pred"]


# ---- FocalLoss ----
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * (1-pt)**self.gamma * ce).mean()


# ======================================================================
# 3. Training & Evaluation
# ======================================================================

def train_enhanced_gnn(graphs, epochs=50, lr=1e-3, device="cpu", patience=15):
    """Train EnhancedQAGNN with focal loss + CE, matching original recipe."""
    if not graphs:
        return None, {}
    nfeat = graphs[0].ndata["feat"].shape[1]
    efeat = graphs[0].edata["feat"].shape[1]
    model = EnhancedQAGNN(nfeat, efeat, hdim=128, nlayers=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4,
                            betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    focal = FocalLoss(alpha=1, gamma=2)
    cls_w = torch.tensor([1.0, 15.0], device=device)

    best_f1, best_state, wait = 0.0, None, 0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        total_loss = 0; tp_all=0; fp_all=0; fn_all=0
        for g in graphs:
            g = g.to(device)
            logits = model(g)
            labels = g.edata["label"]
            loss = 0.7 * focal(logits, labels) + 0.3 * F.cross_entropy(logits, labels, weight=cls_w)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            with torch.no_grad():
                preds = torch.argmax(logits,1)
                tp_all += ((preds==1)&(labels==1)).sum().item()
                fp_all += ((preds==1)&(labels==0)).sum().item()
                fn_all += ((preds==0)&(labels==1)).sum().item()
        sched.step()

        p = tp_all/(tp_all+fp_all) if (tp_all+fp_all) else 0
        r = tp_all/(tp_all+fn_all) if (tp_all+fn_all) else 0
        f1 = 2*p*r/(p+r) if (p+r) else 0
        avg_loss = total_loss/len(graphs)

        if f1 > best_f1:
            best_f1 = f1; best_state = copy.deepcopy(model.state_dict()); wait = 0
        else:
            wait += 1

        if (ep+1) % 5 == 0 or ep == 0:
            elapsed = time.time() - t0
            print(f"    Epoch {ep+1:3d}/{epochs}: loss={avg_loss:.4f} "
                  f"F1={f1:.4f} best={best_f1:.4f} ({elapsed:.1f}s)", flush=True)

        if wait >= patience:
            print(f"    Early stop at epoch {ep+1}", flush=True)
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, {"best_f1": best_f1}


def evaluate_gnn(model, graphs, device="cpu"):
    """Evaluate EnhancedQAGNN on a list of graphs (per-graph, not batched
    because attention operates per-graph)."""
    model.eval()
    tp_all = fp_all = fn_all = 0
    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            logits = model(g)
            preds = torch.argmax(logits, 1)
            labels = g.edata["label"]
            tp_all += ((preds==1)&(labels==1)).sum().item()
            fp_all += ((preds==1)&(labels==0)).sum().item()
            fn_all += ((preds==0)&(labels==1)).sum().item()
    prec = tp_all/(tp_all+fp_all) if (tp_all+fp_all) else 0
    rec  = tp_all/(tp_all+fn_all) if (tp_all+fn_all) else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp_all, "fp": fp_all, "fn": fn_all}


# ======================================================================
# 4. MAIN
# ======================================================================

def main():
    print("=" * 70)
    print("  GNN CROSS-DATASET EXPERIMENTS v2")
    print("  EnhancedQAGNN + Doc2Graph -- FUNSD / WildReceipt / Combined")
    print("=" * 70)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  Started: {ts}")
    device = "cpu"
    print(f"  Device: {device}")

    # -- Convert WildReceipt --
    print("\n[1/4] Converting WildReceipt ...")
    convert_wildreceipt(max_train=200, max_test=50)

    # -- Load graphs --
    print("\n[2/4] Loading graphs (enhanced features) ...")
    funsd_train = load_graphs(FUNSD_DIR / "training_data" / "annotations")
    funsd_test  = load_graphs(FUNSD_DIR / "testing_data"  / "annotations")
    wr_train    = load_graphs(WR_FUNSD_DIR / "training_data" / "annotations")
    wr_test     = load_graphs(WR_FUNSD_DIR / "testing_data"  / "annotations")
    comb_train  = funsd_train + wr_train
    comb_test   = funsd_test  + wr_test
    print(f"  FUNSD:       {len(funsd_train)} train, {len(funsd_test)} test")
    print(f"  WildReceipt: {len(wr_train)} train, {len(wr_test)} test")
    print(f"  Combined:    {len(comb_train)} train, {len(comb_test)} test")

    train_cfgs = {"FUNSD": funsd_train, "WildReceipt": wr_train, "Combined": comb_train}
    test_cfgs  = {"FUNSD": funsd_test,  "WildReceipt": wr_test,  "Combined": comb_test}

    # -- Step 3: EnhancedQAGNN --
    print("\n[3/4] EnhancedQAGNN cross-dataset experiments ...")
    qa_matrix = {}

    # Check for existing FUNSD checkpoint
    ckpt_path = DOC2GRAPH_ROOT / "FUNSD_Model" / "models" / "enhanced_qa_gnn_best.pt"
    funsd_ckpt_available = ckpt_path.exists()
    if funsd_ckpt_available:
        print(f"  Found FUNSD checkpoint: {ckpt_path}")

    for train_name, train_g in train_cfgs.items():
        print(f"\n  --- {train_name} ---")

        if train_name == "FUNSD" and funsd_ckpt_available:
            # Load from checkpoint (this was trained with 80 docs, best val F1=0.808)
            print(f"  Loading from checkpoint ...")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            nfd = ckpt.get("node_feat_dim", 34)
            efd = ckpt.get("edge_feat_dim", 15)
            model = EnhancedQAGNN(nfd, efd, hdim=128, nlayers=3).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Checkpoint loaded (epoch {ckpt.get('epoch','?')})")
        else:
            # Train from scratch
            print(f"  Training on {train_name} ({len(train_g)} graphs) ...")
            model, info = train_enhanced_gnn(
                train_g, epochs=40, lr=1e-3, device=device, patience=15)
            if model is None:
                for tn in test_cfgs:
                    qa_matrix[f"{train_name}->{tn}"] = {"f1": 0.0}
                continue
            # Save
            save_p = RESULTS_DIR / f"enhanced_qa_gnn_{train_name.lower()}.pt"
            torch.save(model.state_dict(), save_p)
            print(f"    Model saved: {save_p}")

        for test_name, test_g in test_cfgs.items():
            metrics = evaluate_gnn(model, test_g, device)
            key = f"{train_name}->{test_name}"
            qa_matrix[key] = metrics
            print(f"    {key:30s}  P={metrics['precision']:.4f}  "
                  f"R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")

        # Save intermediate
        _save_partial(qa_matrix, {})
        gc.collect()

    # -- Step 4: Doc2Graph --
    print("\n[4/4] Doc2Graph cross-dataset experiments ...")
    d2g_matrix = {}

    funsd_d2g_ckpt = DOC2GRAPH_ROOT / "src" / "models" / "checkpoints" / "e2e-funsd-best.pt"

    if DGL_OK:
        try:
            _init_doc2graph_env()
            for train_name in ["FUNSD", "WildReceipt", "Combined"]:
                print(f"\n  --- Doc2Graph {train_name} ---")
                if train_name == "FUNSD" and funsd_d2g_ckpt.exists():
                    print(f"  Using existing checkpoint ...")
                    for test_name in ["FUNSD", "WildReceipt", "Combined"]:
                        test_path = _get_test_path(test_name)
                        metrics = _eval_d2g_checkpoint(test_path, funsd_d2g_ckpt)
                        key = f"{train_name}->{test_name}"
                        d2g_matrix[key] = metrics
                        print(f"    {key:30s}  edge_F1={metrics.get('edge_f1',0):.4f}")
                else:
                    print(f"  Training from scratch ...")
                    train_path = _get_train_path(train_name)
                    model_d2g, nnc, enc, chunks = _train_d2g(train_path)
                    if model_d2g is None:
                        for tn in ["FUNSD","WildReceipt","Combined"]:
                            d2g_matrix[f"{train_name}->{tn}"] = {"edge_f1": 0.0, "error": "training failed"}
                        continue
                    for test_name in ["FUNSD","WildReceipt","Combined"]:
                        test_path = _get_test_path(test_name)
                        metrics = _eval_d2g_model(model_d2g, test_path, nnc, enc, chunks)
                        key = f"{train_name}->{test_name}"
                        d2g_matrix[key] = metrics
                        print(f"    {key:30s}  edge_F1={metrics.get('edge_f1',0):.4f}")
                _save_partial(qa_matrix, d2g_matrix)
        except Exception as ex:
            print(f"  Doc2Graph error: {ex}")
            import traceback; traceback.print_exc()
    else:
        print("  SKIPPED - DGL not available")

    # -- Results Summary --
    _print_summary(qa_matrix, d2g_matrix)
    _save_results(qa_matrix, d2g_matrix)
    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ======================================================================
# Helpers
# ======================================================================

def _save_partial(qa_m, d2g_m):
    r = {"qa_gnn_matrix": {}, "doc2graph_matrix": {}}
    for k,v in qa_m.items():
        r["qa_gnn_matrix"][k] = {kk: (float(vv) if isinstance(vv,(int,float,np.floating,np.integer)) else vv) for kk,vv in v.items()}
    for k,v in d2g_m.items():
        r["doc2graph_matrix"][k] = {kk: (float(vv) if isinstance(vv,(int,float,np.floating,np.integer)) else vv) for kk,vv in v.items()}
    with open(RESULTS_DIR / "gnn_cross_dataset_results.json", "w") as f:
        json.dump(r, f, indent=2)


def _save_results(qa_m, d2g_m):
    _save_partial(qa_m, d2g_m)
    print(f"  Results saved to: {RESULTS_DIR / 'gnn_cross_dataset_results.json'}")


def _print_summary(qa_m, d2g_m):
    dsets = ["FUNSD","WildReceipt","Combined"]
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    hdr = "Train \\ Test"
    print(f"\n  Proposed QA-GNN (3x3 F1 matrix):")
    print(f"  {hdr:<18s}", end="")
    for d in dsets: print(f"  {d:>12s}", end="")
    print()
    for td in dsets:
        print(f"  {td:<18s}", end="")
        for ed in dsets:
            f1 = qa_m.get(f"{td}->{ed}", {}).get("f1", 0.0)
            print(f"  {f1:12.4f}", end="")
        print()

    print(f"\n  Doc2Graph (3x3 edge-F1 matrix):")
    print(f"  {hdr:<18s}", end="")
    for d in dsets: print(f"  {d:>12s}", end="")
    print()
    for td in dsets:
        print(f"  {td:<18s}", end="")
        for ed in dsets:
            f1 = d2g_m.get(f"{td}->{ed}", {}).get("edge_f1", 0.0)
            print(f"  {f1:12.4f}", end="")
        print()


# Combined data paths
def _ensure_combined():
    ctest = DATASETS_DIR / "combined_funsd" / "testing_data"
    ctrain = DATASETS_DIR / "combined_funsd" / "training_data"
    for d in [ctest/"annotations", ctest/"images", ctrain/"annotations", ctrain/"images"]:
        d.mkdir(parents=True, exist_ok=True)
    for src_dir in [FUNSD_DIR/"testing_data", WR_FUNSD_DIR/"testing_data"]:
        for sub in ["annotations","images"]:
            src = src_dir / sub
            if src.exists():
                for fp in src.iterdir():
                    dst = ctest / sub / fp.name
                    if not dst.exists(): shutil.copy2(fp, dst)
    for src_dir in [FUNSD_DIR/"training_data", WR_FUNSD_DIR/"training_data"]:
        for sub in ["annotations","images"]:
            src = src_dir / sub
            if src.exists():
                for fp in src.iterdir():
                    dst = ctrain / sub / fp.name
                    if not dst.exists(): shutil.copy2(fp, dst)


def _get_test_path(name):
    _ensure_combined()
    if name == "FUNSD":       return FUNSD_DIR / "testing_data"
    if name == "WildReceipt": return WR_FUNSD_DIR / "testing_data"
    return DATASETS_DIR / "combined_funsd" / "testing_data"


def _get_train_path(name):
    _ensure_combined()
    if name == "FUNSD":       return FUNSD_DIR / "training_data"
    if name == "WildReceipt": return WR_FUNSD_DIR / "training_data"
    return DATASETS_DIR / "combined_funsd" / "training_data"


# -- Doc2Graph wrappers --
def _init_doc2graph_env():
    sys.path.insert(0, str(DOC2GRAPH_ROOT))
    sys.path.insert(0, str(DOC2GRAPH_ROOT / "src"))


def _eval_d2g_checkpoint(test_path, ckpt_path):
    original = os.getcwd()
    os.chdir(str(DOC2GRAPH_ROOT))
    try:
        from src.data.dataloader import Document2Graph
        from src.models.graphs import SetModel
        from src.training.utils import get_device, get_f1
        device = get_device(-1)
        out = DOC2GRAPH_ROOT / "outputs" / "cross_ckpt"
        out.mkdir(parents=True, exist_ok=True)
        data = Document2Graph(name="CKPT-EVAL", src_path=str(test_path),
                              device=device, output_dir=str(out))
        sm = SetModel(name="e2e", device=device)
        model = sm.get_model(4, data.edge_num_classes, data.get_chunks())
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        model.eval()
        tg = dgl.batch(data.graphs).to(device)
        with torch.no_grad():
            _, e_scores = model(tg, tg.ndata["feat"])
        e_preds = torch.argmax(F.softmax(e_scores,1),1)
        e_labels = tg.edata["label"]
        tp = int(((e_preds==1)&(e_labels==1)).sum().item())
        fp = int(((e_preds==1)&(e_labels==0)).sum().item())
        fn = int(((e_preds==0)&(e_labels==1)).sum().item())
        prec = tp/(tp+fp) if (tp+fp) else 0
        rec  = tp/(tp+fn) if (tp+fn) else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        os.chdir(original)
        return {"edge_precision": prec, "edge_recall": rec, "edge_f1": f1}
    except Exception as ex:
        os.chdir(original)
        print(f"    D2G ckpt eval error: {ex}")
        import traceback; traceback.print_exc()
        return {"edge_f1": 0.0, "error": str(ex)}


def _train_d2g(train_path):
    original = os.getcwd()
    os.chdir(str(DOC2GRAPH_ROOT))
    try:
        from src.data.dataloader import Document2Graph
        from src.models.graphs import SetModel
        from src.training.utils import get_device
        device = get_device(-1)
        out = DOC2GRAPH_ROOT / "outputs" / "cross_train"
        out.mkdir(parents=True, exist_ok=True)
        data = Document2Graph(name="CROSS-TRAIN", src_path=str(train_path),
                              device=device, output_dir=str(out))
        sm = SetModel(name="e2e", device=device)
        model = sm.get_model(data.node_num_classes, data.edge_num_classes,
                             data.get_chunks())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        tg = dgl.batch(data.graphs).to(device)
        labels = tg.ndata["label"]; e_labels = tg.edata["label"]
        best_f1, best_state, wait = 0.0, None, 0
        for ep in range(60):
            model.train()
            n_scores, e_scores = model(tg, tg.ndata["feat"])
            loss = F.cross_entropy(n_scores, labels) + F.cross_entropy(e_scores, e_labels)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            with torch.no_grad():
                ep2 = torch.argmax(F.softmax(e_scores,1),1)
                tp = ((ep2==1)&(e_labels==1)).sum().item()
                fp = ((ep2==1)&(e_labels==0)).sum().item()
                fn = ((ep2==0)&(e_labels==1)).sum().item()
                p = tp/(tp+fp) if (tp+fp) else 0
                r = tp/(tp+fn) if (tp+fn) else 0
                f1 = 2*p*r/(p+r) if (p+r) else 0
            if f1 > best_f1:
                best_f1 = f1; best_state = copy.deepcopy(model.state_dict()); wait = 0
            else:
                wait += 1
            if wait >= 15: break
            if (ep+1)%10==0:
                print(f"    D2G epoch {ep+1}: loss={loss.item():.4f} edge_F1={f1:.4f}", flush=True)
        if best_state: model.load_state_dict(best_state)
        os.chdir(original)
        return model, data.node_num_classes, data.edge_num_classes, data.get_chunks()
    except Exception as ex:
        os.chdir(original)
        print(f"    D2G training error: {ex}")
        import traceback; traceback.print_exc()
        return None, 0, 0, None


def _eval_d2g_model(model, test_path, nnc, enc, chunks):
    original = os.getcwd()
    os.chdir(str(DOC2GRAPH_ROOT))
    try:
        from src.data.dataloader import Document2Graph
        from src.training.utils import get_device
        device = get_device(-1)
        out = DOC2GRAPH_ROOT / "outputs" / "cross_eval"
        out.mkdir(parents=True, exist_ok=True)
        data = Document2Graph(name="CROSS-EVAL", src_path=str(test_path),
                              device=device, output_dir=str(out))
        model.eval()
        tg = dgl.batch(data.graphs).to(device)
        with torch.no_grad():
            _, e_scores = model(tg, tg.ndata["feat"])
        ep2 = torch.argmax(F.softmax(e_scores,1),1)
        el = tg.edata["label"]
        tp = int(((ep2==1)&(el==1)).sum().item())
        fp = int(((ep2==1)&(el==0)).sum().item())
        fn = int(((ep2==0)&(el==1)).sum().item())
        prec = tp/(tp+fp) if (tp+fp) else 0
        rec  = tp/(tp+fn) if (tp+fn) else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        os.chdir(original)
        return {"edge_precision": prec, "edge_recall": rec, "edge_f1": f1}
    except Exception as ex:
        os.chdir(original)
        print(f"    D2G eval error: {ex}")
        import traceback; traceback.print_exc()
        return {"edge_f1": 0.0, "error": str(ex)}


if __name__ == "__main__":
    main()
