from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import minimize


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


def clean_network_json(data: dict) -> dict:
    if "nodes" not in data or "links" not in data:
        raise ValueError("JSON must contain 'nodes' and 'links'.")

    nodes = []
    seen_nodes = set()

    for n in data["nodes"]:
        nid = str(n["id"])
        if nid in seen_nodes:
            continue
        seen_nodes.add(nid)

        if "x" in n and "y" in n:
            x = float(n["x"])
            y = float(n["y"])
        elif "pos" in n and len(n["pos"]) >= 2:
            x = float(n["pos"][0])
            y = float(n["pos"][1])
        else:
            raise ValueError(f"Node {nid} has neither x/y nor pos.")

        nodes.append({
            "id": nid,
            "x": x,
            "y": y,
            "z": float(n.get("z", 0.0)),
            "pos": [x, y, float(n.get("z", 0.0))],
        })

    id_set = {n["id"] for n in nodes}
    links = []
    seen_edges = set()

    for e in data["links"]:
        s = str(e["source"])
        t = str(e["target"])
        if s == t or s not in id_set or t not in id_set:
            continue
        key = tuple(sorted((s, t)))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        links.append({"source": key[0], "target": key[1]})

    return {"nodes": nodes, "links": links}


def load_network(path: Path | str, name: str) -> dict:
    data = clean_network_json(load_json(path))
    ids = [n["id"] for n in data["nodes"]]
    id_to_idx = {nid: i for i, nid in enumerate(ids)}

    x0 = np.zeros((len(ids), 2), dtype=float)
    for i, n in enumerate(data["nodes"]):
        x0[i, 0] = float(n["x"])
        x0[i, 1] = float(n["y"])

    edges = []
    for e in data["links"]:
        a = id_to_idx[e["source"]]
        b = id_to_idx[e["target"]]
        if a != b:
            edges.append(tuple(sorted((a, b))))
    edges = sorted(set(edges))

    if len(edges) == 0:
        raise ValueError(f"No valid edges found in {path}")

    edge_arr = np.asarray(edges, dtype=int)
    vec = x0[edge_arr[:, 1]] - x0[edge_arr[:, 0]]
    l0 = np.linalg.norm(vec, axis=1)
    keep = l0 > 1e-12
    edge_arr = edge_arr[keep]
    l0 = l0[keep]

    G = nx.Graph()
    G.add_nodes_from(range(len(ids)))
    G.add_edges_from([tuple(e) for e in edge_arr])
    degree = np.array([G.degree[i] for i in range(len(ids))], dtype=int)

    return {
        "name": name,
        "path": str(path),
        "ids": ids,
        "x0": x0,
        "edges": edge_arr,
        "l0": l0,
        "degree": degree,
    }


def total_edge_length(net: dict) -> float:
    return float(np.sum(net["l0"]))


def alive_graph(n_nodes: int, edges: np.ndarray, alive: np.ndarray) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from([tuple(e) for e in edges[alive]])
    return G


def giant_component_nodes(G: nx.Graph) -> set:
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return set()
    return set(max(nx.connected_components(G), key=len))


def connected_between_sets(G: nx.Graph, set_a: Sequence[int], set_b: Sequence[int]) -> bool:
    bset = set(int(i) for i in set_b)
    for a in set_a:
        a = int(a)
        if a not in G:
            continue
        comp = nx.node_connected_component(G, a)
        if bset.intersection(comp):
            return True
    return False


def topology_redundancy_metrics(G: nx.Graph,
                                set_a: Sequence[int],
                                set_b: Sequence[int]) -> Tuple[float, int]:
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
    for node in set_a:
        H.add_edge(source, int(node), capacity=cap)
    for node in set_b:
        H.add_edge(int(node), sink, capacity=cap)
    try:
        multiplicity = int(round(nx.minimum_cut_value(H, source, sink, capacity="capacity")))
    except (nx.NetworkXError, nx.NetworkXUnbounded):
        multiplicity = 0
    return float(alternative_path_fraction), multiplicity


def boundary_nodes(x0: np.ndarray, mode: str, fraction: float = 0.06) -> Tuple[np.ndarray, np.ndarray]:
    x = x0[:, 0]
    y = x0[:, 1]
    n = len(x0)

    if mode == "tensile":
        xmin, xmax = float(x.min()), float(x.max())
        band = max((xmax - xmin) * fraction, 1e-9)
        left = np.where(x <= xmin + band)[0]
        right = np.where(x >= xmax - band)[0]

        min_count = max(2, int(math.ceil(0.04 * n)))
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

        min_count = max(2, int(math.ceil(0.04 * n)))
        if len(bottom) < min_count:
            bottom = np.argsort(y)[:min_count]
        if len(top) < min_count:
            top = np.argsort(y)[-min_count:]
        return np.asarray(bottom, dtype=int), np.asarray(top, dtype=int)

    raise ValueError("mode must be 'tensile' or 'shear'.")


def edge_strain(x: np.ndarray, edges: np.ndarray, l0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vec = x[edges[:, 1]] - x[edges[:, 0]]
    L = np.linalg.norm(vec, axis=1)
    eps = (L - l0) / l0
    return eps, L


def spring_energy_grad(q_free: np.ndarray,
                       x_template: np.ndarray,
                       free_idx: np.ndarray,
                       edges: np.ndarray,
                       l0: np.ndarray,
                       alive: np.ndarray,
                       EA: float) -> Tuple[float, np.ndarray]:
    x = x_template.copy()
    x[free_idx] = q_free.reshape((-1, 2))

    ed = edges[alive]
    l0a = l0[alive]
    if len(ed) == 0:
        return 0.0, np.zeros_like(q_free)

    a = ed[:, 0]
    b = ed[:, 1]
    vec = x[b] - x[a]
    L = np.linalg.norm(vec, axis=1)
    Lsafe = np.maximum(L, 1e-12)

    dL = L - l0a
    k = EA / l0a
    energy = 0.5 * np.sum(k * dL * dL)

    coeff = k * dL / Lsafe
    grad_b = coeff[:, None] * vec
    grad_a = -grad_b

    grad_full = np.zeros_like(x)
    np.add.at(grad_full, a, grad_a)
    np.add.at(grad_full, b, grad_b)

    return float(energy), grad_full[free_idx].reshape(-1)


def relax_network(x_start: np.ndarray,
                  fixed_mask: np.ndarray,
                  edges: np.ndarray,
                  l0: np.ndarray,
                  alive: np.ndarray,
                  EA: float,
                  maxiter: int = 500) -> Tuple[np.ndarray, float, bool]:
    free_idx = np.where(~fixed_mask)[0]
    if len(free_idx) == 0:
        return x_start.copy(), 0.0, True

    x_template = x_start.copy()
    q0 = x_start[free_idx].reshape(-1)

    def fun(q):
        return spring_energy_grad(q, x_template, free_idx, edges, l0, alive, EA)

    res = minimize(
        fun=lambda q: fun(q)[0],
        x0=q0,
        jac=lambda q: fun(q)[1],
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-8},
    )

    x_new = x_template.copy()
    x_new[free_idx] = res.x.reshape((-1, 2))
    return x_new, float(res.fun), bool(res.success)


def boundary_reaction_force(x: np.ndarray,
                            boundary_idx: np.ndarray,
                            edges: np.ndarray,
                            l0: np.ndarray,
                            alive: np.ndarray,
                            EA: float,
                            mode: str) -> float:
    force = np.zeros_like(x)
    ed = edges[alive]
    l0a = l0[alive]
    if len(ed) == 0:
        return 0.0

    a = ed[:, 0]
    b = ed[:, 1]
    vec = x[b] - x[a]
    L = np.linalg.norm(vec, axis=1)
    Lsafe = np.maximum(L, 1e-12)

    dL = L - l0a
    k = EA / l0a
    f_on_a = k[:, None] * dL[:, None] * vec / Lsafe[:, None]
    f_on_b = -f_on_a

    np.add.at(force, a, f_on_a)
    np.add.at(force, b, f_on_b)

    reaction = -force[boundary_idx]
    return float(np.sum(reaction[:, 0]))


def box_deform(x0: np.ndarray, mode: str, load: float, box_anchor: str = "center") -> np.ndarray:
    x = x0.copy()
    xmin = float(x0[:, 0].min())
    xmax = float(x0[:, 0].max())
    ymin = float(x0[:, 1].min())
    ymax = float(x0[:, 1].max())
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)

    if box_anchor not in {"min", "center"}:
        raise ValueError("box_anchor must be 'min' or 'center'.")

    if mode == "tensile":
        if box_anchor == "center":
            x[:, 0] = xmid + (1.0 + load) * (x0[:, 0] - xmid)
        else:
            x[:, 0] = xmin + (1.0 + load) * (x0[:, 0] - xmin)
        x[:, 1] = x0[:, 1]
        return x

    if mode == "shear":
        if box_anchor == "center":
            x[:, 0] = x0[:, 0] + load * (x0[:, 1] - ymid)
        else:
            x[:, 0] = x0[:, 0] + load * (x0[:, 1] - ymin)
        x[:, 1] = x0[:, 1]
        return x

    raise ValueError("mode must be 'tensile' or 'shear'.")


def box_stress_proxy(x: np.ndarray,
                     x0: np.ndarray,
                     edges: np.ndarray,
                     l0: np.ndarray,
                     alive: np.ndarray,
                     EA: float,
                     mode: str) -> float:
    ed = edges[alive]
    l0a = l0[alive]
    if len(ed) == 0:
        return 0.0

    a = ed[:, 0]
    b = ed[:, 1]
    vec = x[b] - x[a]
    L = np.linalg.norm(vec, axis=1)
    Lsafe = np.maximum(L, 1e-12)
    nvec = vec / Lsafe[:, None]

    dL = L - l0a
    k = EA / l0a
    tension = k * dL

    xmin, xmax = float(x0[:, 0].min()), float(x0[:, 0].max())
    ymin, ymax = float(x0[:, 1].min()), float(x0[:, 1].max())
    area = max((xmax - xmin) * (ymax - ymin), 1e-12)

    if mode == "tensile":
        return float(np.sum(tension * L * nvec[:, 0] * nvec[:, 0]) / area)

    if mode == "shear":
        return float(np.sum(tension * L * nvec[:, 0] * nvec[:, 1]) / area)

    raise ValueError("mode must be 'tensile' or 'shear'.")


def elastic_energy(x: np.ndarray,
                   edges: np.ndarray,
                   l0: np.ndarray,
                   alive: np.ndarray,
                   EA: float) -> float:
    ed = edges[alive]
    l0a = l0[alive]
    if len(ed) == 0:
        return 0.0
    L = np.linalg.norm(x[ed[:, 1]] - x[ed[:, 0]], axis=1)
    dL = L - l0a
    k = EA / l0a
    return float(0.5 * np.sum(k * dL * dL))


def record_step(net: dict,
                loading_style: str,
                mode: str,
                step: int,
                load: float,
                x: np.ndarray,
                alive: np.ndarray,
                boundary_a: np.ndarray,
                boundary_b: np.ndarray,
                EA: float,
                eps_break: float,
                energy: float,
                response: float,
                n_break_this_step: int,
                success: bool,
                lcc_failure_threshold: float) -> dict:
    edges = net["edges"]
    l0 = net["l0"]
    strain, _ = edge_strain(x, edges, l0)
    G = alive_graph(len(net["x0"]), edges, alive)
    connected = connected_between_sets(G, boundary_a, boundary_b)
    alternative_path_fraction, boundary_path_multiplicity = topology_redundancy_metrics(
        G, boundary_a, boundary_b
    )
    gc = giant_component_nodes(G)
    lcc_frac = len(gc) / max(len(net["x0"]), 1)
    broken_edges = int(len(edges) - int(np.sum(alive)))
    broken_fraction = broken_edges / max(len(edges), 1)
    failed = (not connected) or (lcc_frac < lcc_failure_threshold)

    alive_strain = strain[alive] if np.any(alive) else np.array([], dtype=float)

    return {
        "network": net["name"],
        "loading_style": loading_style,
        "mode": mode,
        "step": int(step),
        "load": float(load),
        "energy": float(energy),
        "response": float(response),
        "response_abs": float(abs(response)),
        "alive_edges": int(np.sum(alive)),
        "broken_edges": broken_edges,
        "broken_fraction": float(broken_fraction),
        "n_break_this_step": int(n_break_this_step),
        "largest_component_fraction": float(lcc_frac),
        "connected_between_boundaries": int(connected),
        "alternative_path_fraction": float(alternative_path_fraction),
        "boundary_path_multiplicity": int(boundary_path_multiplicity),
        "failed": int(failed),
        "mean_strain_alive": float(np.mean(alive_strain)) if alive_strain.size else 0.0,
        "max_strain_alive": float(np.max(alive_strain)) if alive_strain.size else 0.0,
        "eps_break": float(eps_break),
        "EA": float(EA),
        "opt_success": int(success),
    }


def load_summary(rows: List[dict],
                 net: dict,
                 loading_style: str,
                 mode: str,
                 max_load: float,
                 steps: int,
                 EA: float,
                 eps_break: float,
                 failure_step: Optional[int],
                 failure_load: Optional[float],
                 lcc_failure_threshold: float) -> dict:
    loads = np.array([float(r["load"]) for r in rows], dtype=float)
    resp = np.array([abs(float(r["response"])) for r in rows], dtype=float)
    peak_response = float(np.max(resp)) if resp.size else 0.0
    area_like = float(np.trapezoid(resp, loads)) if len(loads) > 1 else 0.0

    last = rows[-1]
    return {
        "network": net["name"],
        "input_path": net["path"],
        "loading_style": loading_style,
        "mode": mode,
        "n_nodes": len(net["x0"]),
        "initial_edges": len(net["edges"]),
        "initial_total_edge_length": total_edge_length(net),
        "final_alive_edges": int(last["alive_edges"]),
        "final_broken_edges": int(last["broken_edges"]),
        "final_broken_fraction": float(last["broken_fraction"]),
        "final_largest_component_fraction": float(last["largest_component_fraction"]),
        "final_connected_between_boundaries": int(last["connected_between_boundaries"]),
        "failure_step": failure_step,
        "failure_load": failure_load,
        "peak_response_abs": peak_response,
        "response_area_like": area_like,
        "EA": EA,
        "eps_break": eps_break,
        "max_load": max_load,
        "steps": steps,
        "lcc_failure_threshold": lcc_failure_threshold,
    }


def simulate_boundary(
    net: dict,
    mode: str,
    out_dir: Path | str,
    EA: float,
    eps_break: float,
    max_load: float,
    steps: int,
    boundary_fraction: float,
    lcc_failure_threshold: float,
) -> Tuple[dict, List[dict]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x0 = net["x0"].copy()
    edges = net["edges"].copy()
    l0 = net["l0"].copy()
    alive = np.ones(len(edges), dtype=bool)

    boundary_a, boundary_b = boundary_nodes(x0, mode, boundary_fraction)
    fixed_mask = np.zeros(len(x0), dtype=bool)
    fixed_mask[boundary_a] = True
    fixed_mask[boundary_b] = True

    xmin, xmax = float(x0[:, 0].min()), float(x0[:, 0].max())
    ymin, ymax = float(x0[:, 1].min()), float(x0[:, 1].max())
    width = xmax - xmin
    height = ymax - ymin

    x_current = x0.copy()
    rows: List[dict] = []
    trajectory_x: List[np.ndarray] = []
    trajectory_active: List[np.ndarray] = []
    failure_step = None
    failure_load = None

    for step in range(steps + 1):
        load = max_load * step / steps
        x_step = x_current.copy()

        x_step[boundary_a] = x0[boundary_a]
        x_step[boundary_b] = x0[boundary_b]
        if mode == "tensile":
            x_step[boundary_b, 0] += load * width
        elif mode == "shear":
            x_step[boundary_b, 0] += load * height
        else:
            raise ValueError(mode)

        n_break_this_step = 0
        energy = 0.0
        success = True

        for _ in range(30):
            x_relaxed, energy, success = relax_network(x_step, fixed_mask, edges, l0, alive, EA)
            strain, _ = edge_strain(x_relaxed, edges, l0)
            to_break = alive & (strain >= eps_break)
            if not np.any(to_break):
                x_step = x_relaxed
                break
            alive[to_break] = False
            n_break_this_step += int(np.sum(to_break))
            x_step = x_relaxed

        x_current = x_step.copy()
        response = boundary_reaction_force(x_current, boundary_b, edges, l0, alive, EA, mode)

        row = record_step(net, "boundary", mode, step, load, x_current, alive,
                          boundary_a, boundary_b, EA, eps_break, energy, response,
                          n_break_this_step, success, lcc_failure_threshold)
        rows.append(row)
        trajectory_x.append(x_current.copy())
        trajectory_active.append(alive.copy())

        if row["failed"] and failure_step is None:
            failure_step = step
            failure_load = load

    summary = load_summary(
        rows, net, "boundary", mode, max_load, steps, EA, eps_break,
        failure_step, failure_load, lcc_failure_threshold,
    )
    save_run_outputs(net, out_dir, rows, summary, trajectory_x, trajectory_active)
    return summary, rows


def simulate_box(
    net: dict,
    mode: str,
    out_dir: Path | str,
    EA: float,
    eps_break: float,
    max_load: float,
    steps: int,
    boundary_fraction: float,
    lcc_failure_threshold: float,
    box_anchor: str = "center",
) -> Tuple[dict, List[dict]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x0 = net["x0"].copy()
    edges = net["edges"].copy()
    l0 = net["l0"].copy()
    alive = np.ones(len(edges), dtype=bool)

    boundary_a, boundary_b = boundary_nodes(x0, mode, boundary_fraction)

    rows: List[dict] = []
    trajectory_x: List[np.ndarray] = []
    trajectory_active: List[np.ndarray] = []
    failure_step = None
    failure_load = None
    x_current = x0.copy()

    for step in range(steps + 1):
        load = max_load * step / steps
        x_current = box_deform(x0, mode, load, box_anchor=box_anchor)

        strain, _ = edge_strain(x_current, edges, l0)
        to_break = alive & (strain >= eps_break)
        n_break_this_step = int(np.sum(to_break))
        if n_break_this_step > 0:
            alive[to_break] = False

        energy = elastic_energy(x_current, edges, l0, alive, EA)
        response = box_stress_proxy(x_current, x0, edges, l0, alive, EA, mode)

        row = record_step(net, "box", mode, step, load, x_current, alive,
                          boundary_a, boundary_b, EA, eps_break, energy, response,
                          n_break_this_step, True, lcc_failure_threshold)
        rows.append(row)
        trajectory_x.append(x_current.copy())
        trajectory_active.append(alive.copy())

        if row["failed"] and failure_step is None:
            failure_step = step
            failure_load = load

    summary = load_summary(
        rows, net, "box", mode, max_load, steps, EA, eps_break,
        failure_step, failure_load, lcc_failure_threshold,
    )
    save_run_outputs(net, out_dir, rows, summary, trajectory_x, trajectory_active)
    return summary, rows


def save_run_outputs(
    net: dict,
    out_dir: Path,
    rows: List[dict],
    summary: dict,
    trajectory_x: List[np.ndarray],
    trajectory_active: List[np.ndarray],
) -> None:
    write_csv(out_dir / "history.csv", rows)
    save_json(out_dir / "summary.json", summary)
    np.savez_compressed(
        out_dir / "state_trajectory.npz",
        x=np.stack(trajectory_x, axis=0),
        active=np.stack(trajectory_active, axis=0).astype(np.uint8),
        xref=net["x0"],
        edges=net["edges"],
        l0=net["l0"],
        step=np.array([r["step"] for r in rows], dtype=np.int64),
        time=np.array([r["step"] for r in rows], dtype=np.int64),
        phase=np.full(len(rows), "load"),
    )
