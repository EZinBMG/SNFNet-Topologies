from pathlib import Path
import warnings

import networkx as nx
import numpy as np
import pandas as pd


NETWORKS = ("random", "ca_low", "ca_high")
METRICS = (
    "loaded_reversible_debond_fraction",
    "final_reversible_debond_fraction",
    "final_recovery_ratio",
    "peak_response_normalized",
    "Drep_selected_mean",
    "closure_demand_auc",
)
SELECTION_METRICS = METRICS[:3]
HISTORY_METRICS = (
    "response_value",
    "reversible_debond_fraction",
    "recovery_ratio",
)


def _require_columns(frame, columns):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")


def _validate_networks(frame):
    if frame["network"].isna().any():
        raise ValueError("Network labels must not be missing.")
    unknown = set(frame["network"]) - set(NETWORKS)
    if unknown:
        raise ValueError(f"Unknown network labels: {sorted(unknown)}")


def _validated_runs(frame, metrics):
    _require_columns(frame, ("network", "seed", *metrics))
    result = frame.copy()
    _validate_networks(result)
    seeds = pd.to_numeric(result["seed"], errors="raise")
    if not np.isfinite(seeds).all() or (seeds != np.floor(seeds)).any():
        raise ValueError("Seeds must be finite integers.")
    result["seed"] = seeds.astype(int)
    duplicates = result.duplicated(["network", "seed"], keep=False)
    if duplicates.any():
        keys = result.loc[duplicates, ["network", "seed"]].to_dict("records")
        raise ValueError(f"Duplicate run keys: {keys}")
    for metric in metrics:
        result[metric] = pd.to_numeric(result[metric], errors="raise")
    return result


def _statistics(frame, group_columns, metrics):
    pieces = []
    for metric in metrics:
        values = frame.groupby(list(group_columns), sort=False)[metric]
        summary = values.agg(n="count", mean="mean", sd="std").reset_index()
        summary.insert(len(group_columns), "metric", metric)
        pieces.append(summary)
    result = pd.concat(pieces, ignore_index=True)
    result["_network_order"] = result["network"].map(
        dict(zip(NETWORKS, range(len(NETWORKS))))
    )
    result["_metric_order"] = result["metric"].map(dict(zip(metrics, range(len(metrics)))))
    order = [
        "_network_order",
        *[column for column in group_columns if column != "network"],
        "_metric_order",
    ]
    return (
        result.sort_values(order)
        .drop(columns=["_network_order", "_metric_order"])
        .reset_index(drop=True)
    )


def replacement_lengths(net: dict) -> np.ndarray:
    edges = np.asarray(net["edges"], dtype=int)
    lengths = np.asarray(net["l0"], dtype=float)
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) != len(lengths):
        raise ValueError("Edges and initial lengths must have matching dimensions.")
    if not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError("Initial edge lengths must be finite and positive.")
    graph = nx.Graph()
    graph.add_nodes_from(range(len(net["x0"])))
    for (source, target), length in zip(edges, lengths):
        if source == target or graph.has_edge(int(source), int(target)):
            raise ValueError("The network must contain unique non-self edges.")
        graph.add_edge(int(source), int(target), weight=float(length))
    result = np.empty(len(edges), dtype=float)
    for index, ((source, target), length) in enumerate(zip(edges, lengths)):
        source, target = int(source), int(target)
        graph.remove_edge(source, target)
        try:
            result[index] = nx.shortest_path_length(graph, source, target, weight="weight")
        except nx.NetworkXNoPath:
            result[index] = np.nan
        finally:
            graph.add_edge(source, target, weight=float(length))
    return result


def summarize_runs(frame: pd.DataFrame) -> pd.DataFrame:
    runs = _validated_runs(frame, METRICS)
    return _statistics(runs, ("network",), METRICS)


def selection_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    runs = _validated_runs(frame, SELECTION_METRICS)
    if runs["seed"].nunique() < 2:
        raise ValueError("Selection ranking requires at least two complete common-seed runs.")
    if not np.isfinite(runs[list(SELECTION_METRICS)].to_numpy()).all():
        raise ValueError("Selection metrics must be finite.")
    counts = runs.groupby("seed")["network"].nunique()
    if (counts != len(NETWORKS)).any():
        raise ValueError("Each candidate seed must include all three networks.")
    wide = runs.pivot(index="seed", columns="network", values=list(SELECTION_METRICS))
    deviations = wide - wide.mean(axis=0)
    scales = wide.std(axis=0, ddof=1)
    standardized = deviations.divide(scales.mask(scales == 0), axis=1).fillna(0.0)
    loading = wide["loaded_reversible_debond_fraction"]
    result = pd.DataFrame(
        {
            "seed": wide.index,
            "eligible": (loading["ca_high"] > loading["ca_low"])
            & (loading["ca_low"] > loading["random"]),
            "score": standardized.pow(2).sum(axis=1),
        }
    ).reset_index(drop=True)
    result = result.sort_values(
        ["eligible", "score", "seed"], ascending=[False, True, True]
    ).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result[["seed", "eligible", "score", "rank"]]


def write_analysis(
    frame: pd.DataFrame,
    output: Path,
    histories: pd.DataFrame | None = None,
    include_selection: bool = False,
) -> None:
    runs = _validated_runs(frame, METRICS)
    statistics = summarize_runs(runs)
    ranking = selection_ranking(runs) if include_selection else None
    history_statistics = None
    if histories is not None:
        histories = histories.copy()
        history_values = histories.copy()
        if "network_label" in history_values.columns:
            if "network" in history_values.columns:
                public_names = history_values["network"].replace(
                    {"random_to_star1": "random", "star_1": "ca_low", "star_2": "ca_high"}
                )
                if not public_names.equals(history_values["network_label"]):
                    raise ValueError("History network and network_label columns disagree.")
            history_values["network"] = history_values["network_label"]
        _require_columns(history_values, ("network", "time", *HISTORY_METRICS))
        _validate_networks(history_values)
        if history_values["time"].isna().any():
            raise ValueError("History time values must not be missing.")
        if "seed" in history_values.columns and history_values.duplicated(
            ["network", "seed", "time"]
        ).any():
            raise ValueError("Duplicate network/seed/time history keys are not allowed.")
        for metric in HISTORY_METRICS:
            history_values[metric] = pd.to_numeric(history_values[metric], errors="raise")
        history_statistics = _statistics(history_values, ("network", "time"), HISTORY_METRICS)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output / "run_summary.csv", index=False)
    statistics.to_csv(output / "summary_statistics.csv", index=False)
    if histories is not None:
        histories.to_csv(output / "history.csv", index=False)
        history_statistics.to_csv(output / "pointwise_statistics.csv", index=False)
    if ranking is not None:
        ranking.to_csv(output / "representative_selection.csv", index=False)


def reference_comparison(actual: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    actual = _validated_runs(actual, METRICS)
    reference = _validated_runs(reference, METRICS)
    unknown_seeds = sorted(set(actual["seed"]) - set(reference["seed"]))
    if unknown_seeds:
        warnings.warn(
            f"No original reference is available for seeds {unknown_seeds}; these seeds are excluded from comparison.",
            UserWarning,
            stacklevel=2,
        )
    comparable = actual[actual["seed"].isin(reference["seed"])]
    joined = comparable.merge(
        reference[["network", "seed", *METRICS]],
        on=["network", "seed"],
        how="left",
        suffixes=("_actual", "_reference"),
        validate="one_to_one",
        indicator=True,
    )
    missing = joined["_merge"] != "both"
    if missing.any():
        keys = joined.loc[missing, ["network", "seed"]].to_dict("records")
        raise ValueError(f"Missing reference run keys: {keys}")
    pieces = []
    for metric in METRICS:
        reproduced = joined[f"{metric}_actual"].to_numpy(dtype=float)
        original = joined[f"{metric}_reference"].to_numpy(dtype=float)
        absolute = np.abs(reproduced - original)
        relative = np.full(len(joined), np.nan, dtype=float)
        np.divide(absolute, np.abs(original), out=relative, where=original != 0)
        relative[(original == 0) & (absolute == 0)] = 0.0
        relative[(original == 0) & (absolute > 0)] = np.inf
        pieces.append(
            pd.DataFrame(
                {
                    "network": joined["network"],
                    "seed": joined["seed"],
                    "metric": metric,
                    "reference": original,
                    "reproduced": reproduced,
                    "absolute_difference": absolute,
                    "relative_difference": relative,
                }
            )
        )
    return pd.concat(pieces, ignore_index=True)
