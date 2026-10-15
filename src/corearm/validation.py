from pathlib import Path
import json

import networkx as nx
import numpy as np
import pandas as pd

from . import analysis


DATA = Path(__file__).parent / "data"
RTOL = 1e-9
ATOL = 1e-10


def _add(checks, case, name, passed, **details):
    checks.append({"case": case, "check": name, "passed": bool(passed), **details})
    return bool(passed)


def _compare(checks, case, name, actual, expected, exact=False):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return _add(checks, case, name, False, actual_shape=list(actual.shape), expected_shape=list(expected.shape))
    if actual.dtype.kind in "biuf" and expected.dtype.kind in "biuf":
        finite = np.isfinite(actual) & np.isfinite(expected)
        good = finite & ((actual == expected) if exact else np.isclose(actual, expected, rtol=RTOL, atol=ATOL))
        difference = np.abs(actual.astype(float) - expected.astype(float))
        maximum = float(difference[finite].max()) if finite.any() else None
    else:
        good = actual == expected
        maximum = None
    details = {}
    if not np.all(good):
        details = {"mismatch_count": int(np.sum(~good)), "max_absolute_difference": maximum,
                   "first_mismatch_indices": np.argwhere(~good)[:8].tolist()}
    return _add(checks, case, name, np.all(good), **details)


def _history(checks, case, frame, name, expected, exact=False):
    if name not in frame:
        return _add(checks, case, f"history.{name}", False, error="Missing column.")
    if np.isscalar(expected):
        expected = np.full(len(frame), expected)
    return _compare(checks, case, f"history.{name}", frame[name].to_numpy(), expected, exact)


def _summary(checks, case, summary, expected):
    for name, value in expected.items():
        if name not in summary:
            _add(checks, case, f"summary.{name}", False, error="Missing field.")
        elif value is None:
            _add(checks, case, f"summary.{name}", summary[name] is None)
        else:
            _compare(checks, case, f"summary.{name}", summary[name], value,
                     isinstance(value, (int, str, bool, np.integer)))


def _network(name):
    source = json.loads((DATA / "networks" / f"{name}.json").read_text(encoding="utf-8"))
    nodes = source["nodes"]
    index = {node["id"]: i for i, node in enumerate(nodes)}
    xref = np.array([[node["x"], node["y"]] for node in nodes], dtype=float)
    pairs = {tuple(sorted((index[edge["source"]], index[edge["target"]]))) for edge in source["links"]}
    edges = np.array(sorted(pair for pair in pairs if pair[0] != pair[1]), dtype=int)
    lengths = np.linalg.norm(xref[edges[:, 1]] - xref[edges[:, 0]], axis=1)
    keep = lengths > 1e-9
    return {"x0": xref, "edges": edges[keep], "l0": lengths[keep]}


def _boundaries(xref, fraction):
    y = xref[:, 1]
    band = max(float(np.ptp(y)) * fraction, 1e-9)
    count = max(2, int(np.ceil(0.04 * len(y))))
    low, high = np.flatnonzero(y <= y.min() + band), np.flatnonzero(y >= y.max() - band)
    if len(low) < count:
        low = np.argsort(y)[:count]
    if len(high) < count:
        high = np.argsort(y)[-count:]
    return low, high


def _topology(n, edges, active, low, high):
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(map(tuple, edges[active]))
    components = list(nx.connected_components(graph))
    lcc = max(map(len, components)) / n
    low_set, high_set = set(map(int, low)), set(map(int, high))
    connected = int(any(component & low_set and component & high_set for component in components))
    m = graph.number_of_edges()
    alternative = 1.0 - sum(1 for _ in nx.bridges(graph)) / m if m else 0.0
    if not connected:
        return lcc, connected, alternative, 0
    for a, b in graph.edges:
        graph[a][b]["capacity"] = 1.0
    for node in low:
        graph.add_edge("source", int(node), capacity=float(m + 1))
    for node in high:
        graph.add_edge(int(node), "sink", capacity=float(m + 1))
    paths = int(round(nx.minimum_cut_value(graph, "source", "sink", capacity="capacity")))
    return lcc, connected, alternative, paths


def _case(checks, case, folder, protocol, assignments, net, topology_cache, distances):
    label = case["path"]
    frame = pd.read_csv(folder / "history.csv")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    with np.load(folder / "state_trajectory.npz", allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    reversible = case["kind"] == "reversible"
    nframes = 201 if reversible else 81
    required = {"x", "active", "xref", "edges", "l0", "step", "time", "phase"}
    if reversible:
        required.add("reversible")
    if not _add(checks, label, "state.required_arrays", required <= state.keys(), missing=sorted(required - state.keys())):
        return None, 0
    n, m = len(net["x0"]), len(net["edges"])
    shapes = {"x": (nframes, n, 2), "active": (nframes, m), "xref": (n, 2),
              "edges": (m, 2), "l0": (m,), "step": (nframes,), "time": (nframes,), "phase": (nframes,)}
    if reversible:
        shapes["reversible"] = (m,)
    shape_ok = True
    for key, shape in shapes.items():
        shape_ok &= _add(checks, label, f"state.{key}.shape", state[key].shape == shape,
                         actual_shape=list(state[key].shape), expected_shape=list(shape))
    shape_ok &= _add(checks, label, "history.row_count", len(frame) == nframes, actual=len(frame), expected=nframes)
    _add(checks, label, "history.unique_columns", not frame.columns.duplicated().any())
    if not shape_ok:
        return None, int(len(state["x"]))
    numeric = frame.select_dtypes(include=[np.number]).to_numpy()
    _add(checks, label, "history.finite_numeric_values", np.isfinite(numeric).all())
    finite = all(np.isfinite(state[key]).all() for key in ("x", "xref", "edges", "l0", "active"))
    if not _add(checks, label, "state.finite_values", finite):
        return None, nframes
    edges, l0, xref, x = state["edges"], state["l0"], state["xref"], state["x"]
    edge_ok = np.issubdtype(edges.dtype, np.integer) and np.all((edges >= 0) & (edges < n))
    edge_ok = edge_ok and np.all(edges[:, 0] != edges[:, 1]) and len(np.unique(np.sort(edges, axis=1), axis=0)) == m
    if not _add(checks, label, "state.valid_edges", edge_ok):
        return None, nframes
    if not _add(checks, label, "state.positive_rest_lengths", np.all(l0 > 0)):
        return None, nframes
    _compare(checks, label, "input.edges", edges, net["edges"], True)
    _compare(checks, label, "input.coordinates", xref, net["x0"], True)
    _compare(checks, label, "input.rest_lengths", l0, net["l0"])
    _compare(checks, label, "state.initial_coordinates", x[0], xref)
    _add(checks, label, "state.active_binary", np.isin(state["active"], [0, 1]).all())
    active = state["active"].astype(bool)
    _add(checks, label, "state.initial_active", active[0].all())
    step = np.arange(nframes)
    for key in ("step", "time"):
        _compare(checks, label, f"state.{key}", state[key], step, True)
    _history(checks, label, frame, "time" if reversible else "step", step, True)
    _history(checks, label, frame, "network", case["network"], True)
    _history(checks, label, frame, "mode", "shear", True)
    _history(checks, label, frame, "loading_method" if reversible else "loading_style", case["loading_style"], True)
    _history(checks, label, frame, "opt_success", 1, True)
    phase = np.where(step <= 80, "load", np.where(step <= 90, "hold", np.where(step <= 150, "unload", "recovery"))) if reversible else np.full(nframes, "load")
    load = np.where(step <= 80, 0.8 * step / 80, np.where(step <= 90, 0.8, np.where(step <= 150, 0.8 * (1 - (step - 90) / 60), 0.0))) if reversible else 0.8 * step / 80
    _compare(checks, label, "state.phase", state["phase"], phase, True)
    if reversible:
        _history(checks, label, frame, "phase", phase, True)
        _history(checks, label, frame, "seed", case["seed"], True)
    _history(checks, label, frame, "load", load)
    parameters = protocol[case["kind"]]
    low, high = _boundaries(xref, parameters["boundary_fraction"])
    height = np.ptp(xref[:, 1])
    if case["loading_style"] == "boundary":
        stop = 151 if reversible else nframes
        _compare(checks, label, "constraints.lower_boundary", x[:stop, low], np.broadcast_to(xref[low], x[:stop, low].shape))
        targets = np.broadcast_to(xref[high], x[:stop, high].shape).copy()
        targets[:, :, 0] += load[:stop, None] * height
        _compare(checks, label, "constraints.upper_boundary", x[:stop, high], targets)
    center_y = (xref[:, 1].min() + xref[:, 1].max()) / 2
    affine = np.broadcast_to(xref, x.shape).copy()
    affine[:, :, 0] += load[:, None] * (xref[None, :, 1] - center_y)
    if not reversible and case["loading_style"] == "box":
        _compare(checks, label, "constraints.affine_box", x, affine)
    vector = x[:, edges[:, 1]] - x[:, edges[:, 0]]
    length = np.linalg.norm(vector, axis=2)
    extension = length - l0
    strain = extension / l0
    counts = active.sum(axis=1)
    stiffness = parameters["base_EA"] / np.maximum(l0, 1e-9) * np.where(active, 1.0, parameters["inactive_stiffness"]) if reversible else parameters["EA"] / l0 * active
    energy = 0.5 * np.sum(stiffness * extension ** 2, axis=1)
    _history(checks, label, frame, "elastic_energy" if reversible else "energy", energy)
    if case["loading_style"] == "boundary":
        signs = np.isin(edges[:, 1], high).astype(int) - np.isin(edges[:, 0], high).astype(int)
        response = np.sum(signs * stiffness * extension * vector[:, :, 0] / np.maximum(length, 1e-12), axis=1)
    elif reversible:
        response = np.sum(np.linalg.norm(parameters["box_tether_k"] * (x - affine), axis=2), axis=1)
        response[151:] = 0.0
    else:
        area = max(float(np.ptp(xref[:, 0]) * height), 1e-12)
        response = np.sum(stiffness * extension * vector[:, :, 0] * vector[:, :, 1] / np.maximum(length, 1e-12), axis=1) / area
    _history(checks, label, frame, "response_value" if reversible else "response", response)
    topology = []
    for mask in active:
        cache_key = (case["network"], mask.tobytes())
        if cache_key not in topology_cache:
            topology_cache[cache_key] = _topology(n, edges, mask, low, high)
        topology.append(topology_cache[cache_key])
    topology = np.asarray(topology)
    _history(checks, label, frame, "lcc_fraction" if reversible else "largest_component_fraction", topology[:, 0])
    _history(checks, label, frame, "boundary_connected" if reversible else "connected_between_boundaries", topology[:, 1], True)
    _history(checks, label, frame, "alternative_path_fraction", topology[:, 2])
    _history(checks, label, frame, "boundary_path_multiplicity", topology[:, 3], True)
    peak = float(np.max(np.abs(response)))
    expected = {"network": case["network"], "mode": "shear", "n_nodes": n,
                "loading_method" if reversible else "loading_style": case["loading_style"],
                "max_load": 0.8, "peak_response_abs": peak}
    if reversible:
        _add(checks, label, "state.reversible_binary", np.isin(state["reversible"], [0, 1]).all())
        mask = state["reversible"].astype(bool)
        nrev = int(mask.sum())
        _add(checks, label, "state.permanent_edges_always_active", active[:, ~mask].all())
        _add(checks, label, "assignment.reversible_count", nrev == int(round(parameters["reversible_fraction"] * m)))
        rows = assignments[(assignments["seed"] == case["seed"]) & (assignments["network"] == case["network"])]
        selected = rows["edge_index"].to_numpy(dtype=int)
        if len(rows):
            _compare(checks, label, "assignment.original_edge_indices", np.flatnonzero(mask), np.sort(selected), True)
        internal = protocol["networks"][case["network"]]
        generator = np.random.default_rng(case["seed"] + 1009 + sum(map(ord, internal)))
        selected = generator.choice(np.arange(m), size=int(round(parameters["reversible_fraction"] * m)), replace=False)
        _compare(checks, label, "assignment.seeded_edge_indices", np.flatnonzero(mask), np.sort(selected), True)
        rev_active = active[:, mask].sum(axis=1)
        rev_fraction = rev_active / nrev if nrev else np.ones(nframes)
        debond = 1 - rev_fraction
        for name, value in {"n_edges_total": m, "permanent_edges": m - nrev, "reversible_edges": nrev,
                            "active_bonds_total": counts, "debonded_bonds_total": m - counts,
                            "reversible_active_bonds": rev_active, "reversible_debonded_bonds": nrev - rev_active}.items():
            _history(checks, label, frame, name, value, True)
        for name, value in {"permanent_fraction": (m - nrev) / m, "reversible_fraction": nrev / m,
                            "active_fraction_total": counts / m, "total_debond_fraction": 1 - counts / m,
                            "reversible_active_fraction": rev_fraction, "reversible_debond_fraction": debond}.items():
            _history(checks, label, frame, name, value)
        rms = np.sqrt(np.mean(np.sum((x - xref) ** 2, axis=2), axis=1)) / max(float(np.linalg.norm(np.ptp(xref, axis=0))), 1e-12)
        recovery = np.clip(1 - rms / rms[90], -1, 1) if rms[90] > 1e-12 else np.zeros(nframes)
        repair = np.clip((rev_fraction - rev_fraction[90]) / debond[90], 0, 1) if debond[90] > 1e-12 else np.zeros(nframes)
        _history(checks, label, frame, "normalized_rms_displacement", rms)
        _history(checks, label, frame, "recovery_ratio", recovery)
        _history(checks, label, frame, "adhesion_repair_ratio", repair)
        _history(checks, label, frame, "mean_abs_edge_strain", np.mean(np.abs(strain), axis=1))
        _history(checks, label, frame, "max_abs_edge_strain", np.max(np.abs(strain), axis=1))
        gap, gap_ratio, candidate = [], [], []
        for i, row in enumerate(active):
            inactive = ~row & mask
            gap.append(float(np.mean(np.maximum(length[i, inactive] - parameters["rebond_ratio"] * l0[inactive], 0))) if inactive.any() else 0.0)
            gap_ratio.append(float(np.mean(np.maximum(length[i, inactive] / l0[inactive] - parameters["rebond_ratio"], 0))) if inactive.any() else 0.0)
            candidate.append(float(np.mean(length[i, inactive] / l0[inactive] <= parameters["rebond_ratio"])) if inactive.any() else 1.0)
        _history(checks, label, frame, "mean_capture_gap_absolute", gap)
        _history(checks, label, frame, "mean_capture_gap_ratio", gap_ratio)
        _history(checks, label, frame, "geometric_rebond_candidate_fraction", candidate)
        _history(checks, label, frame, "boundary_release_progress", (step >= 151).astype(float))
        _history(checks, label, frame, "recovery_release_tether_k", 0.0)
        _history(checks, label, frame, "recovery_viscous_k", np.where(step >= 151, parameters["recovery_viscous_k"], 0.0))
        separation = np.sum((xref[low, None] - xref[low][None, :]) ** 2, axis=2)
        a, b = np.unravel_index(int(np.argmax(separation)), separation.shape)
        anchors = low[[a, b]]
        _compare(checks, label, "constraints.recovery_anchors", x[151:, anchors], np.broadcast_to(xref[anchors], x[151:, anchors].shape))
        if case["loading_style"] == "box":
            _compare(checks, label, "constraints.box_anchors", x[:151, anchors], affine[:151, anchors])
        event_columns = ("n_debond_this_step", "n_rebond_this_step", "cumulative_debond_events", "cumulative_rebond_events")
        if all(column in frame for column in event_columns):
            events = frame[list(event_columns)].to_numpy()
            _add(checks, label, "events.nonnegative_integers", np.isfinite(events).all() and np.all(events >= 0) and np.all(events == np.floor(events)))
            _history(checks, label, frame, "cumulative_debond_events", frame["n_debond_this_step"].cumsum(), True)
            _history(checks, label, frame, "cumulative_rebond_events", frame["n_rebond_this_step"].cumsum(), True)
            _compare(checks, label, "events.active_count_balance", frame["cumulative_debond_events"] - frame["cumulative_rebond_events"], m - counts, True)
            lost = np.r_[0, np.sum(active[:-1] & ~active[1:], axis=1)]
            gained = np.r_[0, np.sum(~active[:-1] & active[1:], axis=1)]
            _add(checks, label, "events.cover_saved_state_transitions", np.all(frame["n_debond_this_step"] >= lost) and np.all(frame["n_rebond_this_step"] >= gained))
            _add(checks, label, "events.no_loading_rebond", np.all(frame.loc[step <= 90, "n_rebond_this_step"] == 0))
            expected.update({name: int(frame[name].iloc[-1]) for name in event_columns[2:]})
            after = np.flatnonzero((step > 90) & (frame["n_rebond_this_step"].to_numpy() > 0))
            expected["first_rebond_step_after_release"] = int(step[after[0]] - 90) if len(after) else None
            expected["first_rebond_phase"] = str(phase[after[0]]) if len(after) else None
        else:
            _add(checks, label, "events.required_columns", False)
        for metric, values, thresholds in (("shape_recovery", recovery, (50, 80, 90, 95)), ("adhesion_repair", repair, (50, 80, 90))):
            for threshold in thresholds:
                found = np.flatnonzero((step > 90) & (values >= threshold / 100))
                expected[f"t_{metric}_{threshold}"] = int(step[found[0]] - 90) if len(found) else None
        expected.update({"seed": case["seed"], "n_edges_total": m, "permanent_edges": m - nrev,
                         "reversible_edges": nrev, "permanent_fraction": (m - nrev) / m,
                         "reversible_fraction": nrev / m, "total_edge_length": float(l0.sum()),
                         "final_recovery_ratio": float(recovery[-1]), "final_normalized_residual_displacement": float(rms[-1]),
                         "final_mean_abs_edge_strain": float(np.mean(np.abs(strain[-1]))),
                         "early_recovery_slope_10steps": float((recovery[100] - recovery[91]) / 10),
                         "closure_demand_auc": float(sum(gap[90:])),
                         "peak_response_normalized": protocol["response_scale"] * peak / float(l0.sum()),
                         "Drep_selected_mean": float(np.nanmean(distances[mask])),
                         "recovery_anchor_node_indices": anchors.tolist(),
                         "recovery_anchor_separation": float(np.linalg.norm(xref[anchors[1]] - xref[anchors[0]]))})
        for prefix, i in (("loaded", 90), ("final", 200)):
            expected.update({f"{prefix}_reversible_active_fraction": float(rev_fraction[i]),
                             f"{prefix}_reversible_debond_fraction": float(debond[i]),
                             f"{prefix}_total_debond_fraction": float(1 - counts[i] / m),
                             f"{prefix}_lcc_fraction": float(topology[i, 0]),
                             f"{prefix}_boundary_connected": int(topology[i, 1])})
        expected["loaded_normalized_rms_displacement"] = float(rms[90])
        for name in ("break_ratio", "rebond_ratio", "inactive_stiffness", "recovery_release_steps", "recovery_release_tether_k", "recovery_viscous_k", "reversible_mode", "reversible_select"):
            expected[name] = parameters[name]
    else:
        _add(checks, label, "state.irreversible_active_monotone", not np.any(~active[:-1] & active[1:]))
        if "reversible" in state:
            _compare(checks, label, "state.no_reversible_edges", state["reversible"], np.zeros(m), True)
        broken = m - counts
        _history(checks, label, frame, "alive_edges", counts, True)
        _history(checks, label, frame, "broken_edges", broken, True)
        _history(checks, label, frame, "broken_fraction", broken / m)
        _history(checks, label, frame, "n_break_this_step", np.r_[broken[0], np.diff(broken)], True)
        _history(checks, label, frame, "response_abs", np.abs(response))
        _history(checks, label, frame, "EA", parameters["EA"])
        _history(checks, label, frame, "eps_break", parameters["eps_break"])
        means = [float(np.mean(value[mask])) if mask.any() else 0.0 for value, mask in zip(strain, active)]
        maxima = [float(np.max(value[mask])) if mask.any() else 0.0 for value, mask in zip(strain, active)]
        _history(checks, label, frame, "mean_strain_alive", means)
        _history(checks, label, frame, "max_strain_alive", maxima)
        _add(checks, label, "fracture.no_unbroken_overthreshold_edges", np.all(strain[active] < parameters["eps_break"]))
        failed = (topology[:, 1] == 0) | (topology[:, 0] < parameters["lcc_failure_threshold"])
        _history(checks, label, frame, "failed", failed.astype(int), True)
        first = np.flatnonzero(failed)
        interior = ~np.isin(np.arange(n), np.r_[low, high])
        survival = []
        for boundary in (low, high):
            selected = np.isin(np.arange(n), boundary)
            mask = (selected[edges[:, 0]] & interior[edges[:, 1]]) | (selected[edges[:, 1]] & interior[edges[:, 0]])
            if not mask.any():
                raise ValueError("No boundary-to-interior edges.")
            survival.append(active[:, mask].mean(axis=1))
        _history(checks, label, frame, "boundary_link_survival", np.mean(survival, axis=0))
        expected.update({"initial_edges": m, "initial_total_edge_length": float(l0.sum()),
                         "final_alive_edges": int(counts[-1]), "final_broken_edges": int(broken[-1]),
                         "final_broken_fraction": float(broken[-1] / m),
                         "final_largest_component_fraction": float(topology[-1, 0]),
                         "final_connected_between_boundaries": int(topology[-1, 1]),
                         "failure_step": int(first[0]) if len(first) else None,
                         "failure_load": float(load[first[0]]) if len(first) else None,
                         "response_area_like": float(np.trapezoid(np.abs(response), load)),
                         "steps": 80, "EA": parameters["EA"], "eps_break": parameters["eps_break"],
                         "lcc_failure_threshold": parameters["lcc_failure_threshold"]})
    _summary(checks, label, summary, expected)
    return summary, nframes


def _reference_report(summaries, reference):
    if not summaries:
        return {"compared_cases": 0, "all_values_exact": None, "metrics": []}
    actual = pd.DataFrame(summaries)
    if not np.isfinite(actual[list(analysis.METRICS)].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite summary values cannot be compared with the archived reference.")
    comparison = analysis.reference_comparison(actual, reference)
    metrics = []
    for metric, rows in comparison.groupby("metric", sort=False):
        difference = rows["absolute_difference"].to_numpy()
        finite = np.isfinite(difference)
        item = {"metric": metric, "compared_values": len(rows),
                "exact_value_matches": int(np.sum(difference == 0)),
                "different_values": int(np.sum(difference != 0)),
                "nonfinite_values": int(np.sum(~finite)),
                "max_absolute_difference": float(difference[finite].max()) if finite.any() else None}
        if len(rows) and finite.any():
            worst = rows.loc[rows["absolute_difference"].idxmax()]
            item["largest_difference_case"] = {"network": worst["network"], "seed": int(worst["seed"]),
                                               "reference": float(worst["reference"]), "reproduced": float(worst["reproduced"])}
        if metric in analysis.SELECTION_METRICS:
            for precision in (1, 2):
                changed = rows.apply(lambda row: f"{100 * row['reference']:.{precision}f}" != f"{100 * row['reproduced']:.{precision}f}", axis=1)
                item[f"percent_display_changes_{precision}_decimal"] = rows.loc[changed, ["network", "seed"]].to_dict("records")
        metrics.append(item)
    comparable = comparison[["network", "seed"]].drop_duplicates()
    missing = actual.merge(comparable, on=["network", "seed"], how="left", indicator=True)
    return {"compared_cases": len(comparable), "all_values_exact": bool((comparison["absolute_difference"] == 0).all()) if len(comparison) else None,
            "cases_without_original_reference": missing.loc[missing["_merge"] != "both", ["network", "seed"]].to_dict("records"),
            "metrics": metrics,
            "interpretation": "Historical numerical differences are reported separately. Passing internal checks does not establish exact reproduction of the archived original results.",
            "detailed_csv": "reference_comparison.csv"}


def verify(output: Path) -> dict:
    output = Path(output).resolve()
    checks, summaries = [], []
    saved_states = 0
    report = {"checks_passed": False, "case_count": 0, "saved_states": 0, "checks": checks,
              "internal_numeric_tolerance": {"rtol": RTOL, "atol": ATOL,
                                             "scope": "Recomputing values from the same saved states; never used to accept archived-reference drift."},
              "reference_comparison": None}
    try:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        protocol = json.loads((DATA / "protocol.json").read_text(encoding="utf-8"))
        assignments = pd.read_csv(DATA / "reference" / "assignments.csv")
        reference = pd.read_csv(DATA / "reference" / "reversible_summary.csv")
        cases = manifest["cases"]
        if not isinstance(cases, list):
            raise ValueError("Manifest cases must be a list.")
        report["case_count"] = len(cases)
        report["profile"] = manifest.get("profile")
        _add(checks, "manifest", "nonempty_cases", bool(cases))
        if "status" in manifest:
            _add(checks, "manifest", "complete_status", manifest["status"] == "complete", actual=manifest["status"])
        if "protocol" in manifest:
            _add(checks, "manifest", "canonical_protocol", manifest["protocol"] == protocol)
        if "case_count" in manifest:
            _compare(checks, "manifest", "case_count", manifest["case_count"], len(cases), True)
        keys = [(case["kind"], case["network"], case["seed"], case["loading_style"]) for case in cases]
        _add(checks, "manifest", "unique_case_keys", len(set(keys)) == len(keys))
        _add(checks, "manifest", "unique_case_paths", len({case["path"] for case in cases}) == len(cases))
        if manifest.get("profile") in ("representative", "manuscript"):
            seeds = protocol["ensemble_seeds"] if manifest["profile"] == "manuscript" else [protocol["representative_seed"]]
            expected = {("reversible", network, seed, "boundary") for seed in seeds for network in analysis.NETWORKS}
            if manifest["profile"] == "manuscript":
                expected |= {("irreversible", network, None, style) for network in analysis.NETWORKS for style in ("boundary", "box")}
            _add(checks, "manifest", "profile_case_coverage", set(keys) == expected, expected_cases=len(expected))
        networks = {name: _network(name) for name in analysis.NETWORKS}
        distances = {name: analysis.replacement_lengths(net) for name, net in networks.items()}
        topology_cache = {}
        for case in cases:
            label = str(case.get("path", "unknown"))
            try:
                valid = (case["kind"] in ("reversible", "irreversible") and case["network"] in analysis.NETWORKS
                         and case["loading_style"] in ("boundary", "box")
                         and ((case["kind"] == "reversible" and type(case["seed"]) is int and case["seed"] >= 0)
                              or (case["kind"] == "irreversible" and case["seed"] is None)))
                if not _add(checks, label, "case.valid_metadata", valid):
                    continue
                folder = (output / case["path"]).resolve()
                if not _add(checks, label, "case.path_within_output", folder != output and folder.is_relative_to(output)):
                    continue
                summary, frames = _case(checks, case, folder, protocol, assignments, networks[case["network"]], topology_cache, distances[case["network"]])
                saved_states += frames
                if summary is not None and case["kind"] == "reversible":
                    summaries.append(summary)
            except Exception as error:
                _add(checks, label, "case.read_or_validation_error", False, error=f"{type(error).__name__}: {error}")
        try:
            report["reference_comparison"] = _reference_report(summaries, reference)
        except Exception as error:
            _add(checks, "reference", "comparison_available", False, error=f"{type(error).__name__}: {error}")
    except Exception as error:
        _add(checks, "manifest", "read_or_validation_error", False, error=f"{type(error).__name__}: {error}")
    report["saved_states"] = saved_states
    report["passed_check_count"] = sum(check["passed"] for check in checks)
    report["failed_check_count"] = len(checks) - report["passed_check_count"]
    report["checks_passed"] = bool(checks) and report["failed_check_count"] == 0
    (output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
