from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import minimize

DEFAULTS = {
    "max_tensile_strain": 0.8,
    "max_shear_strain": 0.8,
    "steps_load": 80,
    "steps_hold": 10,
    "steps_unload": 60,
    "steps_recovery": 50,
    "base_EA": 1.0,
    "inactive_stiffness": 0.01,
    "box_tether_k": 5.0,
    "box_anchor": "center",
    "recovery_release_steps": 0,
    "recovery_release_tether_k": 5.0,
    "recovery_viscous_k": 0.00025,
    "relax_maxiter": 500,
    "state_update_loops": 25,
    "break_ratio": 1.35,
    "rebond_ratio": 1.1,
    "rebond_delay": 10,
    "rebond_in_unload": True,
    "rebond_in_hold": False,
    "rebond_during_loading": False,
    "reversible_fraction": 0.3,
    "reversible_mode": "fraction_random",
    "reversible_select": "longest",
    "reversible_edge_csv": None,
    "fill_reversible_to_fraction": True,
    "cap_reversible_to_fraction": False,
    "random_seed": 52,
    "boundary_fraction": 0.06
}


def default_parameters() -> dict:
    return DEFAULTS.copy()


def write_csv(path: Path | str, rows: List[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path | str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path | str, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_nodes_links(data: dict) -> Tuple[List[dict], List[dict]]:
    nodes = []
    seen = set()
    for n in data.get("nodes", []):
        nid = str(n["id"])
        if nid in seen:
            continue
        seen.add(nid)
        if "x" in n and "y" in n:
            x, y = float(n["x"]), float(n["y"])
        elif "pos" in n and len(n["pos"]) >= 2:
            x, y = float(n["pos"][0]), float(n["pos"][1])
        else:
            raise ValueError(f"Node {nid} has no x/y or pos.")
        nodes.append({"id": nid, "x": x, "y": y, "z": float(n.get("z", 0.0))})

    id_set = {n["id"] for n in nodes}
    links = []
    seen_edges = set()
    for e in data.get("links", []):
        s, t = str(e["source"]), str(e["target"])
        if s == t or s not in id_set or t not in id_set:
            continue
        key = tuple(sorted((s, t)))
        if key in seen_edges:
            continue
        seen_edges.add(key)

        out = dict(e)
        out["source"], out["target"] = key[0], key[1]
        links.append(out)
    return nodes, links


def load_network(path: Path | str, name: str) -> dict:
    nodes_raw, links_raw = clean_nodes_links(load_json(path))
    ids = [n["id"] for n in nodes_raw]
    id_to_idx = {nid: i for i, nid in enumerate(ids)}
    x0 = np.array([[float(n["x"]), float(n["y"])] for n in nodes_raw], dtype=float)

    edge_pairs = []
    link_attrs_by_key = {}
    for e in links_raw:
        a = id_to_idx[e["source"]]
        b = id_to_idx[e["target"]]
        if a == b:
            continue
        key_idx = tuple(sorted((a, b)))
        key_id = tuple(sorted((e["source"], e["target"])))
        edge_pairs.append(key_idx)
        link_attrs_by_key[key_id] = dict(e)

    edges = np.array(sorted(set(edge_pairs)), dtype=int)
    if len(edges) == 0:
        raise ValueError(f"No edges found in {path}")

    l0 = np.linalg.norm(x0[edges[:, 1]] - x0[edges[:, 0]], axis=1)
    keep = l0 > 1e-9
    edges = edges[keep]
    l0 = l0[keep]

    return {
        "name": name,
        "path": str(path),
        "ids": ids,
        "x0": x0,
        "edges": edges,
        "l0": l0,
        "link_attrs_by_key": link_attrs_by_key,
    }


def total_edge_length(net: dict) -> float:
    return float(np.sum(net["l0"]))


def edge_id_key(net: dict, edge_index: int) -> Tuple[str, str]:
    a, b = net["edges"][edge_index]
    return tuple(sorted((net["ids"][int(a)], net["ids"][int(b)])))


def parse_edge_csv(csv_path: Path) -> set[Tuple[str, str]]:
    if not csv_path.exists():
        return set()
    keys = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return set()
        fields = [c.strip() for c in reader.fieldnames]
        pairs = [
            ("source", "target"),
            ("segment_source", "segment_target"),
            ("source_id", "target_id"),
            ("u", "v"),
            ("a", "b"),
            ("edge_source", "edge_target"),
            ("new_source", "new_target"),
        ]
        lower_to_field = {c.lower(): c for c in fields}
        chosen = None
        for a, b in pairs:
            if a in lower_to_field and b in lower_to_field:
                chosen = (lower_to_field[a], lower_to_field[b])
                break
        if chosen is None:
            return set()
        for row in reader:
            s = str(row.get(chosen[0], "")).strip()
            t = str(row.get(chosen[1], "")).strip()
            if s and t and s != t:
                keys.add(tuple(sorted((s, t))))
    return keys


def edge_indices_from_added_csv(net: dict, args) -> set[int]:
    paths: List[Path] = []
    if args.reversible_edge_csv:
        paths.append(Path(args.reversible_edge_csv))
    parent = Path(net["path"]).parent
    paths.extend([
        parent / "new_bridge_segments.csv",
        parent / "added_bridges.csv",
        parent / "added_edges.csv",
        parent / "new_edges.csv",
    ])

    id_key_to_idx = {edge_id_key(net, i): i for i in range(len(net["edges"]))}
    out = set()
    for p in paths:
        keys = parse_edge_csv(p)
        for key in keys:
            if key in id_key_to_idx:
                out.add(id_key_to_idx[key])
        if out:
            break
    return out


def edge_indices_from_json_attrs(net: dict) -> set[int]:
    out = set()
    truthy = {"1", "true", "yes", "added", "new", "reversible", "sacrificial", "bridge"}
    attrs = net.get("link_attrs_by_key", {})
    for i in range(len(net["edges"])):
        key = edge_id_key(net, i)
        e = attrs.get(key, {})
        vals = []
        for field in ("reversible", "is_reversible", "bond_type", "edge_type", "kind", "state", "added", "is_added", "source_type"):
            if field in e:
                vals.append(str(e[field]).strip().lower())
        if any(v in truthy for v in vals):
            out.add(i)
    return out


def choose_extra_edges(net: dict, candidates: np.ndarray, n_take: int, method: str, rng: np.random.Generator) -> List[int]:
    candidates = np.array(list(candidates), dtype=int)
    if n_take <= 0 or len(candidates) == 0:
        return []
    n_take = min(n_take, len(candidates))
    if method == "random":
        return list(rng.choice(candidates, size=n_take, replace=False))
    if method == "longest":
        order = candidates[np.argsort(net["l0"][candidates])[::-1]]
        return list(order[:n_take])
    if method == "shortest":
        order = candidates[np.argsort(net["l0"][candidates])]
        return list(order[:n_take])
    if method == "low_betweenness":
        G = nx.Graph()
        G.add_nodes_from(range(len(net["x0"])))
        for k, (a, b) in enumerate(net["edges"]):
            G.add_edge(int(a), int(b), edge_index=k, weight=float(net["l0"][k]))
        eb = nx.edge_betweenness_centrality(G, weight="weight")
        scores = []
        for k in candidates:
            a, b = map(int, net["edges"][k])
            scores.append(eb.get((a, b), eb.get((b, a), 0.0)))
        order = candidates[np.argsort(scores)]
        return list(order[:n_take])
    raise ValueError(f"Unknown reversible_select: {method}")


def assign_reversible_mask(net: dict, args) -> np.ndarray:
    m = len(net["edges"])
    if args.reversible_mode == "all":
        return np.ones(m, dtype=bool)
    if args.reversible_mode == "none":
        return np.zeros(m, dtype=bool)

    target_count = int(round(float(args.reversible_fraction) * m))
    target_count = max(0, min(m, target_count))
    seed_offset = sum(ord(c) for c in net["name"])
    rng = np.random.default_rng(args.random_seed + 1009 + seed_offset)

    selected: set[int] = set()

    if args.reversible_mode in ("added_then_fraction", "json_then_fraction"):
        selected |= edge_indices_from_json_attrs(net)

    if args.reversible_mode == "added_then_fraction" and net["name"] == "random_to_star1":
        selected |= edge_indices_from_added_csv(net, args)

    if args.reversible_mode in ("fraction_random", "fraction_longest", "fraction_shortest", "fraction_low_betweenness"):
        method = {
            "fraction_random": "random",
            "fraction_longest": "longest",
            "fraction_shortest": "shortest",
            "fraction_low_betweenness": "low_betweenness",
        }[args.reversible_mode]
        selected = set(choose_extra_edges(net, np.arange(m), target_count, method, rng))
    else:

        if args.fill_reversible_to_fraction and len(selected) < target_count:
            remaining = np.array([i for i in range(m) if i not in selected], dtype=int)
            selected |= set(choose_extra_edges(net, remaining, target_count - len(selected), args.reversible_select, rng))

        if args.cap_reversible_to_fraction and len(selected) > target_count:
            selected = set(choose_extra_edges(net, np.array(list(selected)), target_count, args.reversible_select, rng))

    mask = np.zeros(m, dtype=bool)
    if selected:
        mask[list(selected)] = True
    return mask


def boundary_nodes(x0: np.ndarray, mode: str, fraction: float = 0.06) -> Tuple[np.ndarray, np.ndarray]:
    x = x0[:, 0]
    y = x0[:, 1]
    n = len(x0)
    min_count = max(2, int(math.ceil(0.04 * n)))

    if mode == "tensile":
        xmin, xmax = float(x.min()), float(x.max())
        band = max((xmax - xmin) * fraction, 1e-9)
        left = np.where(x <= xmin + band)[0]
        right = np.where(x >= xmax - band)[0]
        if len(left) < min_count:
            left = np.argsort(x)[:min_count]
        if len(right) < min_count:
            right = np.argsort(x)[-min_count:]
        return np.asarray(left, dtype=int), np.asarray(right, dtype=int)

    if mode == "shear":
        ymin, ymax = float(y.min()), float(y.max())
        band = max((ymax - ymin) * fraction, 1e-9)
        bottom = np.where(y <= ymin + band)[0]
        top = np.where(y >= ymax - band)[0]
        if len(bottom) < min_count:
            bottom = np.argsort(y)[:min_count]
        if len(top) < min_count:
            top = np.argsort(y)[-min_count:]
        return np.asarray(bottom, dtype=int), np.asarray(top, dtype=int)

    raise ValueError("mode must be tensile or shear")


def recovery_anchors(x0: np.ndarray, boundary_a: np.ndarray) -> np.ndarray:
    if len(boundary_a) >= 2:

        xb = x0[boundary_a]
        d2 = np.sum((xb[:, None, :] - xb[None, :, :]) ** 2, axis=2)
        i, j = np.unravel_index(int(np.argmax(d2)), d2.shape)
        return np.array([boundary_a[i], boundary_a[j]], dtype=int)
    return np.array([boundary_a[0]], dtype=int)


def smooth_release_weight(local_i: int, release_steps: int) -> float:
    if release_steps <= 0:
        return 0.0
    if release_steps == 1:
        return 0.0
    progress = float(np.clip((local_i - 1) / (release_steps - 1), 0.0, 1.0))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def prescribed_boundary_positions(x0: np.ndarray, ba: np.ndarray, bb: np.ndarray, mode: str, load: float) -> Tuple[np.ndarray, np.ndarray]:
    xmin, xmax = float(x0[:, 0].min()), float(x0[:, 0].max())
    ymin, ymax = float(x0[:, 1].min()), float(x0[:, 1].max())
    width = xmax - xmin
    height = ymax - ymin
    fixed_idx = np.concatenate([ba, bb])
    fixed_pos = x0[fixed_idx].copy()
    na = len(ba)

    if mode == "tensile":
        fixed_pos[:na] = x0[ba]
        fixed_pos[na:, 0] = x0[bb, 0] + load * width
        fixed_pos[na:, 1] = x0[bb, 1]
    elif mode == "shear":
        fixed_pos[:na] = x0[ba]
        fixed_pos[na:, 0] = x0[bb, 0] + load * height
        fixed_pos[na:, 1] = x0[bb, 1]
    else:
        raise ValueError("mode must be tensile or shear")
    return fixed_idx, fixed_pos


def affine_box_positions(x0: np.ndarray, mode: str, load: float, anchor: str = "center") -> np.ndarray:
    out = x0.copy()
    xmin, ymin = x0.min(axis=0)
    xmax, ymax = x0.max(axis=0)
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    if mode == "tensile":
        if anchor == "center":
            out[:, 0] = cx + (1.0 + load) * (x0[:, 0] - cx)
        else:
            out[:, 0] = xmin + (1.0 + load) * (x0[:, 0] - xmin)
        out[:, 1] = x0[:, 1]
        return out

    if mode == "shear":
        if anchor == "center":
            out[:, 0] = x0[:, 0] + load * (x0[:, 1] - cy)
        else:
            out[:, 0] = x0[:, 0] + load * (x0[:, 1] - ymin)
        out[:, 1] = x0[:, 1]
        return out

    raise ValueError("mode must be tensile or shear")


def get_constraints_and_tethers(xref: np.ndarray, ba: np.ndarray, bb: np.ndarray, anchors: np.ndarray,
                                mode: str, loading_method: str, phase: str, load: float,
                                local_i: int, args):
    n = len(xref)
    if loading_method == "boundary":
        if phase in ("load", "hold", "unload"):
            fixed_idx, fixed_pos = prescribed_boundary_positions(xref, ba, bb, mode, load)
            return fixed_idx, fixed_pos, None, None, 0.0
        fixed_idx = anchors.copy()
        fixed_pos = xref[fixed_idx].copy()
        release_weight = smooth_release_weight(local_i, args.recovery_release_steps)
        if release_weight > 0.0 and args.recovery_release_tether_k > 0.0:

            tether_idx = np.unique(np.concatenate([ba, bb])).astype(int)
            tether_pos = xref[tether_idx].copy()
            tether_k = args.recovery_release_tether_k * release_weight
            return fixed_idx, fixed_pos, tether_idx, tether_pos, tether_k
        return fixed_idx, fixed_pos, None, None, 0.0

    if loading_method == "box":
        if phase in ("load", "hold", "unload"):
            target = affine_box_positions(xref, mode, load, anchor=args.box_anchor)
            fixed_idx = anchors.copy()
            fixed_pos = target[fixed_idx].copy()
            tether_idx = np.arange(n, dtype=int)
            tether_pos = target.copy()
            return fixed_idx, fixed_pos, tether_idx, tether_pos, args.box_tether_k
        fixed_idx = anchors.copy()
        fixed_pos = xref[fixed_idx].copy()
        release_weight = smooth_release_weight(local_i, args.recovery_release_steps)
        if release_weight > 0.0 and args.recovery_release_tether_k > 0.0:
            tether_idx = np.arange(n, dtype=int)
            tether_pos = xref.copy()
            tether_k = args.recovery_release_tether_k * release_weight
            return fixed_idx, fixed_pos, tether_idx, tether_pos, tether_k
        return fixed_idx, fixed_pos, None, None, 0.0

    raise ValueError("loading_method must be boundary or box")


def edge_ratio(x: np.ndarray, edges: np.ndarray, l0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    L = np.linalg.norm(x[edges[:, 1]] - x[edges[:, 0]], axis=1)
    return L / np.maximum(l0, 1e-12), L


def edge_stiffness(l0: np.ndarray, active: np.ndarray, args) -> np.ndarray:
    k0 = args.base_EA / np.maximum(l0, 1e-9)
    return k0 * np.where(active, 1.0, args.inactive_stiffness)


def spring_elastic_energy(x: np.ndarray, edges: np.ndarray, l0: np.ndarray,
                          active: np.ndarray, args) -> float:
    length = np.linalg.norm(x[edges[:, 1]] - x[edges[:, 0]], axis=1)
    extension = length - l0
    return float(0.5 * np.sum(edge_stiffness(l0, active, args) * extension * extension))


def energy_grad(q: np.ndarray, template: np.ndarray, free_idx: np.ndarray,
                edges: np.ndarray, l0: np.ndarray, active: np.ndarray, args,
                tether_idx: Optional[np.ndarray] = None,
                tether_pos: Optional[np.ndarray] = None,
                tether_k: float = 0.0,
                viscous_pos: Optional[np.ndarray] = None,
                viscous_k: float = 0.0) -> Tuple[float, np.ndarray]:
    x = template.copy()
    x[free_idx] = q.reshape((-1, 2))

    a = edges[:, 0]
    b = edges[:, 1]
    vec = x[b] - x[a]
    L = np.linalg.norm(vec, axis=1)
    Lsafe = np.maximum(L, 1e-12)
    dL = L - l0
    k = edge_stiffness(l0, active, args)

    energy = 0.5 * np.sum(k * dL * dL)
    coeff = k * dL / Lsafe
    fb = coeff[:, None] * vec
    fa = -fb

    grad = np.zeros_like(x)
    np.add.at(grad, a, fa)
    np.add.at(grad, b, fb)

    if tether_idx is not None and len(tether_idx) > 0 and tether_k > 0 and tether_pos is not None:
        diff = x[tether_idx] - tether_pos
        energy += 0.5 * tether_k * float(np.sum(diff * diff))
        np.add.at(grad, tether_idx, tether_k * diff)

    if viscous_pos is not None and viscous_k > 0.0:

        diff = x - viscous_pos
        energy += 0.5 * viscous_k * float(np.sum(diff * diff))
        grad += viscous_k * diff

    return float(energy), grad[free_idx].reshape(-1)


def relax_network(x_start: np.ndarray, fixed_idx: np.ndarray, fixed_pos: np.ndarray,
                  edges: np.ndarray, l0: np.ndarray, active: np.ndarray, args,
                  tether_idx: Optional[np.ndarray] = None,
                  tether_pos: Optional[np.ndarray] = None,
                  tether_k: float = 0.0,
                  viscous_pos: Optional[np.ndarray] = None,
                  viscous_k: float = 0.0) -> Tuple[np.ndarray, float, bool]:
    template = x_start.copy()
    template[fixed_idx] = fixed_pos
    fixed_mask = np.zeros(len(x_start), dtype=bool)
    fixed_mask[fixed_idx] = True
    free_idx = np.where(~fixed_mask)[0]
    if len(free_idx) == 0:
        return template.copy(), 0.0, True

    q0 = template[free_idx].reshape(-1)

    def fun(q):
        return energy_grad(q, template, free_idx, edges, l0, active, args,
                           tether_idx, tether_pos, tether_k, viscous_pos, viscous_k)

    res = minimize(lambda q: fun(q)[0], q0, jac=lambda q: fun(q)[1], method="L-BFGS-B",
                   options={"maxiter": args.relax_maxiter, "ftol": 1e-10, "gtol": 1e-8})
    if not res.success:

        res = minimize(lambda q: fun(q)[0], res.x, jac=lambda q: fun(q)[1], method="L-BFGS-B",
                       options={"maxiter": 2 * args.relax_maxiter, "ftol": 1e-10, "gtol": 1e-8})
    x = template.copy()
    x[free_idx] = res.x.reshape((-1, 2))
    return x, float(res.fun), bool(res.success)


def update_bonds(x: np.ndarray, edges: np.ndarray, l0: np.ndarray,
                 active: np.ndarray, reversible: np.ndarray, cooldown: np.ndarray,
                 phase: str, args) -> Tuple[np.ndarray, np.ndarray, int, int]:
    ratio, _ = edge_ratio(x, edges, l0)
    n_debond = 0
    n_rebond = 0

    active[~reversible] = True

    ai = np.where(active & reversible)[0]
    if len(ai):
        to_off = ratio[ai] >= args.break_ratio
        if np.any(to_off):
            idx = ai[to_off]
            active[idx] = False
            cooldown[idx] = 0
            n_debond = int(len(idx))

    di = np.where((~active) & reversible)[0]
    if len(di):
        cooldown[di] += 1
        allow_rebond = False
        if phase == "load":
            allow_rebond = bool(args.rebond_during_loading)
        elif phase == "hold":
            allow_rebond = bool(args.rebond_in_hold)
        elif phase == "unload":
            allow_rebond = bool(args.rebond_in_unload)
        elif phase == "recovery":
            allow_rebond = True
        if allow_rebond:
            can_rebond = (ratio[di] <= args.rebond_ratio) & (cooldown[di] >= args.rebond_delay)
            if np.any(can_rebond):
                idx = di[can_rebond]
                active[idx] = True
                cooldown[idx] = 0
                n_rebond = int(len(idx))

    active[~reversible] = True
    return active, cooldown, n_debond, n_rebond


def relax_update_stable(x: np.ndarray, fixed_idx: np.ndarray, fixed_pos: np.ndarray,
                        edges: np.ndarray, l0: np.ndarray, active: np.ndarray,
                        reversible: np.ndarray, cooldown: np.ndarray, phase: str, args,
                        tether_idx: Optional[np.ndarray] = None,
                        tether_pos: Optional[np.ndarray] = None,
                        tether_k: float = 0.0,
                        viscous_pos: Optional[np.ndarray] = None,
                        viscous_k: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, int, int]:
    total_debond = 0
    total_rebond = 0
    energy = 0.0
    ok = True
    for _ in range(args.state_update_loops):
        x, energy, ok = relax_network(
            x, fixed_idx, fixed_pos, edges, l0, active, args,
            tether_idx, tether_pos, tether_k, viscous_pos, viscous_k,
        )
        active, cooldown, nb, nr = update_bonds(x, edges, l0, active, reversible, cooldown, phase, args)
        total_debond += nb
        total_rebond += nr
        if nb == 0 and nr == 0:
            break
    return x, active, cooldown, energy, ok, total_debond, total_rebond


def spring_reaction_x_on_nodes(x: np.ndarray, node_idx: np.ndarray,
                               edges: np.ndarray, l0: np.ndarray, active: np.ndarray, args) -> float:
    force = np.zeros_like(x)
    a = edges[:, 0]
    b = edges[:, 1]
    vec = x[b] - x[a]
    L = np.linalg.norm(vec, axis=1)
    Lsafe = np.maximum(L, 1e-12)
    dL = L - l0
    k = edge_stiffness(l0, active, args)
    f_on_a = k[:, None] * dL[:, None] * vec / Lsafe[:, None]
    f_on_b = -f_on_a
    np.add.at(force, a, f_on_a)
    np.add.at(force, b, f_on_b)
    return float(np.sum(-force[node_idx, 0]))


def box_tether_response(x: np.ndarray, target: Optional[np.ndarray], args) -> float:
    if target is None or args.box_tether_k <= 0:
        return 0.0
    f = -args.box_tether_k * (x - target)
    return float(np.sum(np.linalg.norm(f, axis=1)))


def active_graph(n_nodes: int, edges: np.ndarray, active: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(map(tuple, edges[active]))
    return G


def lcc_fraction(n_nodes: int, edges: np.ndarray, active: np.ndarray) -> float:
    G = active_graph(n_nodes, edges, active)
    if G.number_of_edges() == 0:
        return 1.0 / max(n_nodes, 1)
    return float(len(max(nx.connected_components(G), key=len)) / max(n_nodes, 1))


def boundary_connected(n_nodes: int, edges: np.ndarray, active: np.ndarray, ba: np.ndarray, bb: np.ndarray) -> bool:
    G = active_graph(n_nodes, edges, active)
    bset = set(map(int, bb))
    for a in ba:
        ai = int(a)
        if ai not in G:
            continue
        comp = nx.node_connected_component(G, ai)
        if bset.intersection(comp):
            return True
    return False


def topology_redundancy_metrics(n_nodes: int, edges: np.ndarray, active: np.ndarray,
                                ba: np.ndarray, bb: np.ndarray) -> Tuple[float, int]:
    G = active_graph(n_nodes, edges, active)
    m = G.number_of_edges()
    if m == 0:
        return 0.0, 0
    n_bridges = sum(1 for _ in nx.bridges(G))
    alternative_path_fraction = 1.0 - n_bridges / m

    H = G.copy()
    source = "__boundary_source__"
    sink = "__boundary_sink__"
    cap = float(m + 1)
    for u, v in H.edges:
        H[u][v]["capacity"] = 1.0
    H.add_node(source)
    H.add_node(sink)
    for node in ba:
        H.add_edge(source, int(node), capacity=cap)
    for node in bb:
        H.add_edge(int(node), sink, capacity=cap)
    try:
        multiplicity = int(round(nx.minimum_cut_value(H, source, sink, capacity="capacity")))
    except (nx.NetworkXError, nx.NetworkXUnbounded):
        multiplicity = 0
    return float(alternative_path_fraction), multiplicity


def normalized_rms(x: np.ndarray, xref: np.ndarray) -> float:
    diag = math.hypot(float(xref[:, 0].max() - xref[:, 0].min()), float(xref[:, 1].max() - xref[:, 1].min()))
    return float(np.sqrt(np.mean(np.sum((x - xref) ** 2, axis=1))) / max(diag, 1e-12))


def recovery_ratio(loaded_rms: Optional[float], current_rms: float) -> float:
    if loaded_rms is None or loaded_rms <= 1e-12:
        return 0.0
    return float(np.clip(1.0 - current_rms / loaded_rms, -1.0, 1.0))


def mean_abs_edge_strain(x: np.ndarray, edges: np.ndarray, l0: np.ndarray) -> Tuple[float, float]:
    ratio, _ = edge_ratio(x, edges, l0)
    eps = np.abs(ratio - 1.0)
    return float(np.mean(eps)), float(np.max(eps))


def simulate_one(net: dict, mode: str, loading_method: str, out_dir: Path | str, args) -> Tuple[dict, List[dict]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xref = net["x0"].copy()
    x = xref.copy()
    edges = net["edges"].copy()
    l0 = net["l0"].copy()
    m = len(edges)

    reversible = assign_reversible_mask(net, args)
    active = np.ones(m, dtype=bool)
    cooldown = np.zeros(m, dtype=int)

    ba, bb = boundary_nodes(xref, mode, args.boundary_fraction)
    anchors = recovery_anchors(xref, ba)

    max_load = args.max_tensile_strain if mode == "tensile" else args.max_shear_strain

    rows: List[dict] = []
    loaded_rms: Optional[float] = None
    loaded_reversible_active_fraction: Optional[float] = None
    loaded_idx: Optional[int] = None
    x_loaded = None
    cum_debond = 0
    cum_rebond = 0
    trajectory_x: List[np.ndarray] = []
    trajectory_active: List[np.ndarray] = []
    rev_count = int(np.sum(reversible))
    perm_count = int(m - rev_count)

    def record(phase: str, time_step: int, load: float, energy: float, response: float,
               n_debond: int, n_rebond: int, opt_success: bool,
               release_weight: float = 0.0, release_tether_k: float = 0.0):
        nonlocal rows
        nrms = normalized_rms(x, xref)
        recov = recovery_ratio(loaded_rms, nrms)
        mean_eps, max_eps = mean_abs_edge_strain(x, edges, l0)
        rev_active = int(np.sum(active & reversible))
        rev_debond = int(np.sum((~active) & reversible))
        rev_active_fraction = rev_active / rev_count if rev_count else 1.0
        rev_debond_fraction = rev_debond / rev_count if rev_count else 0.0
        ratio_now, lengths_now = edge_ratio(x, edges, l0)
        inactive_rev = (~active) & reversible
        if np.any(inactive_rev):
            geometric_rebond_candidate_fraction = float(
                np.mean(ratio_now[inactive_rev] <= args.rebond_ratio)
            )
            mean_capture_gap_ratio = float(
                np.mean(np.maximum(ratio_now[inactive_rev] - args.rebond_ratio, 0.0))
            )
            mean_capture_gap_absolute = float(
                np.mean(np.maximum(lengths_now[inactive_rev] - args.rebond_ratio * l0[inactive_rev], 0.0))
            )
        else:
            geometric_rebond_candidate_fraction = 1.0
            mean_capture_gap_ratio = 0.0
            mean_capture_gap_absolute = 0.0
        alternative_path_fraction, boundary_path_multiplicity = topology_redundancy_metrics(
            len(xref), edges, active, ba, bb
        )

        if loaded_reversible_active_fraction is None or (1.0 - loaded_reversible_active_fraction) <= 1e-12:
            adhesion_repair = 0.0
        else:
            adhesion_repair = (rev_active_fraction - loaded_reversible_active_fraction) / (1.0 - loaded_reversible_active_fraction)
            adhesion_repair = float(np.clip(adhesion_repair, 0.0, 1.0))

        rows.append({
            "network": net["name"],
            "mode": mode,
            "loading_method": loading_method,
            "phase": phase,
            "time": int(time_step),
            "load": float(load),
            "boundary_release_progress": float(1.0 - release_weight) if phase == "recovery" else 0.0,
            "recovery_release_tether_k": float(release_tether_k),
            "recovery_viscous_k": float(args.recovery_viscous_k) if phase == "recovery" else 0.0,
            "n_edges_total": int(m),
            "permanent_edges": int(perm_count),
            "reversible_edges": int(rev_count),
            "permanent_fraction": float(perm_count / max(m, 1)),
            "reversible_fraction": float(rev_count / max(m, 1)),
            "active_bonds_total": int(np.sum(active)),
            "debonded_bonds_total": int(np.sum(~active)),
            "active_fraction_total": float(np.mean(active)),
            "total_debond_fraction": float(np.mean(~active)),
            "reversible_active_bonds": int(rev_active),
            "reversible_debonded_bonds": int(rev_debond),
            "reversible_active_fraction": float(rev_active_fraction),
            "reversible_debond_fraction": float(rev_debond_fraction),
            "geometric_rebond_candidate_fraction": geometric_rebond_candidate_fraction,
            "mean_capture_gap_ratio": mean_capture_gap_ratio,
            "mean_capture_gap_absolute": mean_capture_gap_absolute,
            "n_debond_this_step": int(n_debond),
            "n_rebond_this_step": int(n_rebond),
            "cumulative_debond_events": int(cum_debond),
            "cumulative_rebond_events": int(cum_rebond),
            "lcc_fraction": float(lcc_fraction(len(xref), edges, active)),
            "boundary_connected": int(boundary_connected(len(xref), edges, active, ba, bb)),
            "alternative_path_fraction": float(alternative_path_fraction),
            "boundary_path_multiplicity": int(boundary_path_multiplicity),
            "normalized_rms_displacement": float(nrms),
            "recovery_ratio": float(recov),
            "adhesion_repair_ratio": float(adhesion_repair),
            "mean_abs_edge_strain": float(mean_eps),
            "max_abs_edge_strain": float(max_eps),
            "elastic_energy": float(energy),
            "response_value": float(response),
            "opt_success": int(opt_success),
        })
        trajectory_x.append(x.copy())
        trajectory_active.append(active.copy())

    def run_state(phase: str, time_step: int, load: float, local_i: int):
        nonlocal x, active, cooldown, cum_debond, cum_rebond
        previous_x = x.copy()
        fixed_idx, fixed_pos, tether_idx, tether_pos, tether_k = get_constraints_and_tethers(
            xref=xref, ba=ba, bb=bb, anchors=anchors, mode=mode,
            loading_method=loading_method, phase=phase, load=load, local_i=local_i, args=args,
        )
        x, active, cooldown, energy, ok, nb, nr = relax_update_stable(
            x=x, fixed_idx=fixed_idx, fixed_pos=fixed_pos, edges=edges, l0=l0,
            active=active, reversible=reversible, cooldown=cooldown, phase=phase, args=args,
            tether_idx=tether_idx, tether_pos=tether_pos, tether_k=tether_k,
            viscous_pos=previous_x if phase == "recovery" else None,
            viscous_k=args.recovery_viscous_k if phase == "recovery" else 0.0,
        )
        cum_debond += nb
        cum_rebond += nr

        energy = spring_elastic_energy(x, edges, l0, active, args)
        if loading_method == "boundary":
            response = spring_reaction_x_on_nodes(x, bb, edges, l0, active, args)
        else:
            response = box_tether_response(x, tether_pos, args) if tether_pos is not None else 0.0
        release_weight = smooth_release_weight(local_i, args.recovery_release_steps) if phase == "recovery" else 0.0
        record(phase, time_step, load, energy, response, nb, nr, ok,
               release_weight=release_weight,
               release_tether_k=tether_k if phase == "recovery" else 0.0)

    for i in range(args.steps_load + 1):
        load = max_load * i / max(args.steps_load, 1)
        run_state("load", i, load, i)

    for i in range(1, args.steps_hold + 1):
        t = args.steps_load + i
        run_state("hold", t, max_load, i)

    x_loaded = x.copy()
    loaded_rms = normalized_rms(x_loaded, xref)
    loaded_reversible_active_fraction = float(np.sum(active & reversible) / max(np.sum(reversible), 1))
    loaded_idx = len(rows) - 1
    for r in rows:
        r["recovery_ratio"] = recovery_ratio(loaded_rms, r["normalized_rms_displacement"])
        if (1.0 - loaded_reversible_active_fraction) > 1e-12:
            r["adhesion_repair_ratio"] = float(
                np.clip((r["reversible_active_fraction"] - loaded_reversible_active_fraction) / (1.0 - loaded_reversible_active_fraction), 0.0, 1.0)
            )
        else:
            r["adhesion_repair_ratio"] = 0.0

    for i in range(1, args.steps_unload + 1):
        t = args.steps_load + args.steps_hold + i
        load = max_load * (1.0 - i / max(args.steps_unload, 1))
        run_state("unload", t, load, i)

    for i in range(1, args.steps_recovery + 1):
        t = args.steps_load + args.steps_hold + args.steps_unload + i
        run_state("recovery", t, 0.0, i)

    write_csv(out_dir / "history.csv", rows)
    np.savez_compressed(
        out_dir / "state_trajectory.npz",
        x=np.stack(trajectory_x, axis=0),
        active=np.stack(trajectory_active, axis=0).astype(np.uint8),
        xref=xref,
        edges=edges,
        l0=l0,
        reversible=reversible.astype(np.uint8),
        step=np.array([r["time"] for r in rows], dtype=np.int64),
        time=np.array([r["time"] for r in rows], dtype=np.int64),
        phase=np.array([r["phase"] for r in rows]),
    )

    release_start = args.steps_load + args.steps_hold
    t_shape50 = t_shape80 = t_shape90 = t_shape95 = None
    t_adh50 = t_adh80 = t_adh90 = None
    first_rebond_step = None
    first_rebond_phase = None
    early_window = 10
    recovery_ratio_at_release_start = None
    recovery_ratio_after_early = None
    for r in rows:
        if r["phase"] not in ("unload", "recovery"):
            continue
        rel = int(r["time"]) - release_start
        if recovery_ratio_at_release_start is None:
            recovery_ratio_at_release_start = float(r["recovery_ratio"])
        if recovery_ratio_after_early is None and rel >= early_window:
            recovery_ratio_after_early = float(r["recovery_ratio"])
        if first_rebond_step is None and int(r["n_rebond_this_step"]) > 0:
            first_rebond_step = rel
            first_rebond_phase = r["phase"]
        if t_shape50 is None and r["recovery_ratio"] >= 0.50:
            t_shape50 = rel
        if t_shape80 is None and r["recovery_ratio"] >= 0.80:
            t_shape80 = rel
        if t_shape90 is None and r["recovery_ratio"] >= 0.90:
            t_shape90 = rel
        if t_shape95 is None and r["recovery_ratio"] >= 0.95:
            t_shape95 = rel
        if t_adh50 is None and r["adhesion_repair_ratio"] >= 0.50:
            t_adh50 = rel
        if t_adh80 is None and r["adhesion_repair_ratio"] >= 0.80:
            t_adh80 = rel
        if t_adh90 is None and r["adhesion_repair_ratio"] >= 0.90:
            t_adh90 = rel
    if recovery_ratio_at_release_start is None:
        recovery_ratio_at_release_start = 0.0
    if recovery_ratio_after_early is None:
        recovery_ratio_after_early = rows[-1]["recovery_ratio"]
    early_recovery_slope = (recovery_ratio_after_early - recovery_ratio_at_release_start) / max(early_window, 1)
    closure_demand_auc = float(sum(
        r["mean_capture_gap_absolute"] for r in rows if int(r["time"]) >= release_start
    ))

    loaded = rows[loaded_idx]
    final = rows[-1]
    summary = {
        "network": net["name"],
        "mode": mode,
        "loading_method": loading_method,
        "n_nodes": len(xref),
        "n_edges_total": m,
        "permanent_edges": int(perm_count),
        "reversible_edges": int(rev_count),
        "permanent_fraction": float(perm_count / max(m, 1)),
        "reversible_fraction": float(rev_count / max(m, 1)),
        "total_edge_length": total_edge_length(net),
        "max_load": max_load,
        "loaded_reversible_active_fraction": loaded["reversible_active_fraction"],
        "loaded_reversible_debond_fraction": loaded["reversible_debond_fraction"],
        "loaded_total_debond_fraction": loaded["total_debond_fraction"],
        "loaded_lcc_fraction": loaded["lcc_fraction"],
        "loaded_boundary_connected": loaded["boundary_connected"],
        "loaded_normalized_rms_displacement": loaded["normalized_rms_displacement"],
        "final_reversible_active_fraction": final["reversible_active_fraction"],
        "final_reversible_debond_fraction": final["reversible_debond_fraction"],
        "final_total_debond_fraction": final["total_debond_fraction"],
        "final_lcc_fraction": final["lcc_fraction"],
        "final_boundary_connected": final["boundary_connected"],
        "final_recovery_ratio": final["recovery_ratio"],
        "final_normalized_residual_displacement": final["normalized_rms_displacement"],
        "final_mean_abs_edge_strain": final["mean_abs_edge_strain"],
        "cumulative_debond_events": int(cum_debond),
        "cumulative_rebond_events": int(cum_rebond),
        "first_rebond_step_after_release": first_rebond_step,
        "first_rebond_phase": first_rebond_phase,
        "t_shape_recovery_50": t_shape50,
        "t_shape_recovery_80": t_shape80,
        "t_shape_recovery_90": t_shape90,
        "t_shape_recovery_95": t_shape95,
        "t_adhesion_repair_50": t_adh50,
        "t_adhesion_repair_80": t_adh80,
        "t_adhesion_repair_90": t_adh90,
        "early_recovery_slope_10steps": early_recovery_slope,
        "closure_demand_auc": closure_demand_auc,
        "peak_response_abs": float(np.max(np.abs([r["response_value"] for r in rows]))),
        "break_ratio": args.break_ratio,
        "rebond_ratio": args.rebond_ratio,
        "inactive_stiffness": args.inactive_stiffness,
        "recovery_anchor_node_indices": [int(i) for i in anchors],
        "recovery_anchor_separation": float(
            np.linalg.norm(xref[anchors[1]] - xref[anchors[0]]) if len(anchors) > 1 else 0.0
        ),
        "recovery_release_steps": int(args.recovery_release_steps),
        "recovery_release_tether_k": float(args.recovery_release_tether_k),
        "recovery_viscous_k": float(args.recovery_viscous_k),
        "box_tether_k": args.box_tether_k if loading_method == "box" else None,
        "box_anchor": args.box_anchor if loading_method == "box" else None,
        "reversible_mode": args.reversible_mode,
        "reversible_select": args.reversible_select,
    }
    save_json(out_dir / "summary.json", summary)
    return summary, rows
