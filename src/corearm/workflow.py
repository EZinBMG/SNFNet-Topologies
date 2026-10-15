from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd

from . import __version__, analysis, irreversible, reversible


DATA = Path(__file__).parent / "data"


def read_protocol():
    return json.loads((DATA / "protocol.json").read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def new_output(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def network_metrics():
    protocol = read_protocol()
    records = []
    for name, internal in protocol["networks"].items():
        net = reversible.load_network(DATA / "networks" / f"{name}.json", internal)
        degrees = np.bincount(net["edges"].ravel(), minlength=len(net["x0"]))
        cores = degrees >= 10
        records.append({
            "network": name,
            "nodes": len(net["x0"]),
            "edges": len(net["edges"]),
            "mean_degree": float(degrees.mean()),
            "core_nodes": int(cores.sum()),
            "mean_core_degree": float(degrees[cores].mean()) if cores.any() else np.nan,
            "total_bridge_length": float(net["l0"].sum()),
            "mean_bridge_length": float(net["l0"].mean()),
        })
    return pd.DataFrame(records)


def _run_case(case, output, protocol):
    started = time.perf_counter()
    folder = Path(output) / case["path"]
    name = case["network"]
    source = DATA / "networks" / f"{name}.json"
    internal_name = protocol["networks"][name]
    if case["kind"] == "reversible":
        net = reversible.load_network(source, internal_name)
        parameters = dict(protocol["reversible"], random_seed=case["seed"])
        summary, rows = reversible.simulate_one(
            net, protocol["mode"], "boundary", folder, SimpleNamespace(**parameters)
        )
        with np.load(folder / "state_trajectory.npz", allow_pickle=False) as states:
            selected = states["reversible"].astype(bool)
        distances = analysis.replacement_lengths(net)
        summary["seed"] = case["seed"]
        summary["Drep_selected_mean"] = float(np.nanmean(distances[selected]))
        summary["peak_response_normalized"] = (
            protocol["response_scale"] * summary["peak_response_abs"] / summary["total_edge_length"]
        )
        for row in rows:
            row["seed"] = case["seed"]
    else:
        net = irreversible.load_network(source, internal_name)
        parameters = protocol["irreversible"].copy()
        if case["loading_style"] == "boundary":
            parameters.pop("box_anchor")
            summary, rows = irreversible.simulate_boundary(net, protocol["mode"], folder, **parameters)
        else:
            summary, rows = irreversible.simulate_box(net, protocol["mode"], folder, **parameters)
        summary.pop("input_path", None)
        with np.load(folder / "state_trajectory.npz", allow_pickle=False) as states:
            active = states["active"].astype(bool)
        boundary_a, boundary_b = irreversible.boundary_nodes(
            net["x0"], protocol["mode"], parameters["boundary_fraction"]
        )
        boundary = np.concatenate((boundary_a, boundary_b))
        interior = ~np.isin(np.arange(len(net["x0"])), boundary)
        edges = net["edges"]
        fractions = []
        for nodes in (boundary_a, boundary_b):
            selected = np.isin(np.arange(len(net["x0"])), nodes)
            mask = (selected[edges[:, 0]] & interior[edges[:, 1]]) | (selected[edges[:, 1]] & interior[edges[:, 0]])
            if not mask.any():
                raise ValueError(f"No boundary-to-interior links for {name}.")
            fractions.append(active[:, mask].mean(axis=1))
        survival = np.mean(fractions, axis=0)
        for row, fraction in zip(rows, survival):
            row["boundary_link_survival"] = float(fraction)
    summary["network"] = name
    for row in rows:
        row["network"] = name
    save_json(folder / "summary.json", summary)
    pd.DataFrame(rows).to_csv(folder / "history.csv", index=False)
    return {**case, "seconds": round(time.perf_counter() - started, 3)}


def _cases(profile, protocol):
    seeds = [protocol["representative_seed"]]
    if profile == "manuscript":
        seeds += [seed for seed in protocol["ensemble_seeds"] if seed not in seeds]
    cases = [
        {"kind": "reversible", "network": name, "seed": seed, "loading_style": "boundary",
         "path": f"reversible/seed_{seed:04d}/{name}"}
        for seed in seeds for name in analysis.NETWORKS
    ]
    if profile == "manuscript":
        cases += [
            {"kind": "irreversible", "network": name, "seed": None, "loading_style": style,
             "path": f"irreversible/{style}/{name}"}
            for style in ("boundary", "box") for name in analysis.NETWORKS
        ]
    return cases


def aggregate(output, cases):
    network_metrics().to_csv(Path(output) / "network_metrics.csv", index=False)
    summaries, histories, pure_summaries, pure_histories = [], [], [], []
    for case in cases:
        folder = Path(output) / case["path"]
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        history = pd.read_csv(folder / "history.csv")
        if case["kind"] == "reversible":
            summaries.append(summary)
            histories.append(history)
        else:
            pure_summaries.append(summary)
            pure_histories.append(history)
    runs = pd.DataFrame(summaries)
    analysis.write_analysis(runs, Path(output) / "reversible", pd.concat(histories, ignore_index=True))
    reference = pd.read_csv(DATA / "reference/reversible_summary.csv")
    comparison = analysis.reference_comparison(runs, reference)
    comparison.to_csv(Path(output) / "reference_comparison.csv", index=False)
    if pure_summaries:
        pure = pd.DataFrame(pure_summaries)
        pure.to_csv(Path(output) / "irreversible/run_summary.csv", index=False)
        pd.concat(pure_histories, ignore_index=True).to_csv(Path(output) / "irreversible/history.csv", index=False)
        reference = pd.read_csv(DATA / "reference/irreversible_summary.csv")
        metrics = ["final_broken_fraction", "final_largest_component_fraction", "peak_response_abs", "response_area_like"]
        joined = pure.merge(reference, on=["network", "loading_style"], how="left", indicator=True,
                            suffixes=("_actual", "_reference"), validate="one_to_one")
        if (joined["_merge"] != "both").any():
            raise ValueError("Missing irreversible reference run keys.")
        parts = []
        for metric in metrics:
            part = joined[["network", "loading_style"]].copy()
            part["metric"] = metric
            part["reference"] = joined[f"{metric}_reference"]
            part["reproduced"] = joined[f"{metric}_actual"]
            part["absolute_difference"] = abs(part["reproduced"] - part["reference"])
            parts.append(part)
        pd.concat(parts, ignore_index=True).to_csv(Path(output) / "irreversible/reference_comparison.csv", index=False)


def run(profile, output):
    output = new_output(output)
    protocol = read_protocol()
    cases = _cases(profile, protocol)
    manifest = {
        "schema_version": 1,
        "package_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "status": "running",
        "case_count": len(cases),
        "cases": cases,
        "protocol": protocol,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    **{name: version(name) for name in ("numpy", "scipy", "pandas", "networkx")}},
        "input_sha256": {name: hashlib.sha256((DATA / "networks" / f"{name}.json").read_bytes()).hexdigest()
                         for name in analysis.NETWORKS},
    }
    save_json(output / "manifest.json", manifest)
    completed = []
    started = time.perf_counter()
    try:
        for case in cases:
            result = _run_case(case, str(output), protocol)
            completed.append(result)
            label = f"seed {result['seed']}" if result["seed"] is not None else result["loading_style"]
            print(f"[{len(completed)}/{len(cases)}] {result['kind']} {result['network']} {label}: {result['seconds']:.1f} s", flush=True)
        by_path = {case["path"]: case for case in completed}
        manifest["cases"] = [by_path[case["path"]] for case in cases]
        aggregate(output, cases)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["seconds"] = round(time.perf_counter() - started, 3)
        manifest["completed_cases"] = len(completed)
        save_json(output / "manifest.json", manifest)
    return output


def summarize_reference(output):
    output = new_output(output)
    runs = pd.read_csv(DATA / "reference/reversible_summary.csv")
    histories = pd.read_csv(DATA / "reference/reversible_history.csv")
    analysis.write_analysis(runs, output, histories, include_selection=True)
    network_metrics().to_csv(output / "network_metrics.csv", index=False)
    save_json(output / "source.json", {"source": "archived original simulations", "runs": len(runs), "statistics": "mean and sample standard deviation (ddof=1)"})
    return output
