from importlib.resources import files
from types import SimpleNamespace
import unittest

import networkx as nx
import numpy as np
import pandas as pd

from corearm import analysis, irreversible, reversible


DATA = files("corearm").joinpath("data")
NETWORK_NAMES = {
    "random": "random_to_star1",
    "ca_low": "star_1",
    "ca_high": "star_2",
}


def central_difference(function, coordinates, step=1e-6):
    result = np.empty_like(coordinates)
    for index in range(len(coordinates)):
        above = coordinates.copy()
        below = coordinates.copy()
        above[index] += step
        below[index] -= step
        result[index] = (function(above)[0] - function(below)[0]) / (2 * step)
    return result


class NetworkDataTests(unittest.TestCase):
    def test_input_topologies_and_edge_order(self):
        expected_sizes = {"random": (87, 174), "ca_low": (80, 188), "ca_high": (112, 270)}
        for public, internal in NETWORK_NAMES.items():
            with self.subTest(network=public):
                path = DATA.joinpath("networks", f"{public}.json")
                reversible_net = reversible.load_network(path, internal)
                irreversible_net = irreversible.load_network(path, internal)
                self.assertEqual(
                    (len(reversible_net["x0"]), len(reversible_net["edges"])),
                    expected_sizes[public],
                )
                np.testing.assert_array_equal(reversible_net["x0"], irreversible_net["x0"])
                np.testing.assert_array_equal(reversible_net["edges"], irreversible_net["edges"])
                np.testing.assert_allclose(reversible_net["l0"], irreversible_net["l0"], rtol=0, atol=0)
                self.assertTrue(np.all(reversible_net["l0"] > 0))
                graph = nx.Graph()
                graph.add_nodes_from(range(len(reversible_net["x0"])))
                graph.add_edges_from(reversible_net["edges"])
                self.assertEqual(graph.number_of_edges(), expected_sizes[public][1])
                self.assertTrue(nx.is_connected(graph))

    def test_seed_assignments_match_original_edge_indices(self):
        reference = pd.read_csv(DATA.joinpath("reference", "assignments.csv"))
        expected_counts = {"random": 52, "ca_low": 56, "ca_high": 81}
        for seed in (42, 52):
            for public, internal in NETWORK_NAMES.items():
                with self.subTest(seed=seed, network=public):
                    net = reversible.load_network(DATA.joinpath("networks", f"{public}.json"), internal)
                    parameters = reversible.default_parameters()
                    parameters["random_seed"] = seed
                    mask = reversible.assign_reversible_mask(net, SimpleNamespace(**parameters))
                    selected = reference.loc[
                        (reference["seed"] == seed) & (reference["network"] == public),
                        "edge_index",
                    ].to_numpy(dtype=int)
                    self.assertEqual(len(selected), expected_counts[public])
                    np.testing.assert_array_equal(np.flatnonzero(mask), np.sort(selected))


class MechanicsTests(unittest.TestCase):
    def setUp(self):
        self.initial = np.array([[0., 0.], [1.2, 0.], [1.1, 1.3], [0., 1.1]])
        self.edges = np.array([[0, 1], [1, 2], [2, 3], [0, 3], [0, 2]], dtype=int)
        self.lengths = np.linalg.norm(
            self.initial[self.edges[:, 1]] - self.initial[self.edges[:, 0]], axis=1
        )
        self.free = np.array([1, 2, 3], dtype=int)
        self.coordinates = (
            self.initial[self.free] + np.array([[.12, .04], [-.08, .07], [.05, -.09]])
        ).ravel()
        self.active = np.array([True, False, True, True, True])

    def test_reversible_energy_gradient_with_tethers_and_damping(self):
        parameters = reversible.default_parameters()
        parameters["base_EA"] = 1.7
        arguments = SimpleNamespace(**parameters)
        tether_nodes = np.array([0, 2, 3])
        tether_positions = self.initial[tether_nodes] + np.array([.03, -.02])

        def objective(coordinates):
            return reversible.energy_grad(
                coordinates, self.initial, self.free, self.edges, self.lengths,
                self.active, arguments, tether_nodes, tether_positions, .43,
                self.initial + np.array([.01, -.03]), .021,
            )

        energy, gradient = objective(self.coordinates)
        self.assertGreater(energy, 0)
        np.testing.assert_allclose(
            gradient, central_difference(objective, self.coordinates), rtol=2e-6, atol=1e-8
        )

    def test_irreversible_energy_gradient_after_edge_deletion(self):
        def objective(coordinates):
            return irreversible.spring_energy_grad(
                coordinates, self.initial, self.free, self.edges, self.lengths,
                self.active, 1.7,
            )

        energy, gradient = objective(self.coordinates)
        self.assertGreater(energy, 0)
        np.testing.assert_allclose(
            gradient, central_difference(objective, self.coordinates), rtol=2e-6, atol=1e-8
        )


class AnalysisTests(unittest.TestCase):
    def test_replacement_path_lengths_and_disconnected_bridge(self):
        net = {
            "x0": np.array([[0., 0.], [3., 0.], [0., 4.], [0., 6.]]),
            "edges": np.array([[0, 1], [1, 2], [0, 2], [2, 3]], dtype=int),
            "l0": np.array([3., 5., 4., 2.]),
        }
        distances = analysis.replacement_lengths(net)
        np.testing.assert_allclose(distances[:3], [9., 7., 8.], rtol=0, atol=0)
        self.assertTrue(np.isnan(distances[3]))

    def test_summary_uses_sample_standard_deviation(self):
        records = []
        for network in analysis.NETWORKS:
            for seed, value in enumerate([1., 2., 3.], start=42):
                records.append({
                    "network": network,
                    "seed": seed,
                    **{metric: value for metric in analysis.METRICS},
                })
        statistics = analysis.summarize_runs(pd.DataFrame(records))
        self.assertEqual(len(statistics), len(analysis.NETWORKS) * len(analysis.METRICS))
        np.testing.assert_array_equal(statistics["n"], 3)
        np.testing.assert_allclose(statistics["mean"], 2., rtol=0, atol=0)
        np.testing.assert_allclose(statistics["sd"], 1., rtol=0, atol=0)

    def test_original_representative_selection(self):
        reference = pd.read_csv(DATA.joinpath("reference", "reversible_summary.csv"))
        self.assertEqual(len(reference), 60)
        self.assertEqual(set(reference["seed"]), set(range(42, 62)))
        self.assertTrue((reference.groupby("seed")["network"].nunique() == 3).all())
        ranking = analysis.selection_ranking(reference)
        self.assertEqual(len(ranking), 20)
        self.assertTrue(np.isfinite(ranking["score"]).all())
        selected = ranking.iloc[0]
        self.assertEqual(int(selected["seed"]), 52)
        self.assertTrue(bool(selected["eligible"]))
        self.assertEqual(int(selected["rank"]), 1)
        loading = reference.loc[
            reference["seed"] == 52, ["network", "loaded_reversible_debond_fraction"]
        ].set_index("network")["loaded_reversible_debond_fraction"]
        self.assertGreater(loading["ca_high"], loading["ca_low"])
        self.assertGreater(loading["ca_low"], loading["random"])


if __name__ == "__main__":
    unittest.main()
