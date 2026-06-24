# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from types import SimpleNamespace

from ogx.core.resolver import topological_sort


def _make_provider(api: str, deps: list[str], provider_id: str = "default"):
    """Create a mock provider with the given API name and dependency list."""
    return SimpleNamespace(spec=SimpleNamespace(deps__=deps), provider_id=provider_id)


def _make_pws(providers_by_api: dict[str, list[tuple[str, list[str], str]]]):
    """
    Build the dict[str, list[provider]] input for topological_sort.

    Each value in providers_by_api is a list of (api_name, deps, provider_id) tuples.
    The api_name key in the outer dict is the API string.
    """
    result: dict[str, list] = {}
    for api, provider_list in providers_by_api.items():
        result[api] = [_make_provider(p[0], p[1], p[2]) for p in provider_list]
    return result


class TestTopologicalSort:
    def test_empty(self):
        result = topological_sort({})
        assert result == []

    def test_single_no_deps(self):
        pws = _make_pws(
            {
                "inference": [("inference", [], "p1")],
            }
        )
        result = topological_sort(pws)
        assert len(result) == 1
        assert result[0][0] == "inference"

    def test_multiple_no_deps(self):
        pws = _make_pws(
            {
                "inference": [("inference", [], "p1")],
                "chat": [("chat", [], "p2")],
            }
        )
        result = topological_sort(pws)
        api_names = [r[0] for r in result]
        assert len(result) == 2
        assert "inference" in api_names
        assert "chat" in api_names

    def test_linear_chain(self):
        pws = _make_pws(
            {
                "responses": [("responses", ["inference"], "p1")],
                "inference": [("inference", [], "p2")],
            }
        )
        result = topological_sort(pws)
        assert result[0][0] == "inference"
        assert result[1][0] == "responses"

    def test_three_node_chain(self):
        pws = _make_pws(
            {
                "a": [("a", ["b"], "pa")],
                "b": [("b", ["c"], "pb")],
                "c": [("c", [], "pc")],
            }
        )
        api_names = [r[0] for r in topological_sort(pws)]
        assert api_names == ["c", "b", "a"]

    def test_fan_in(self):
        # Both "a" and "b" depend on "c"
        pws = _make_pws(
            {
                "a": [("a", ["c"], "pa")],
                "b": [("b", ["c"], "pb")],
                "c": [("c", [], "pc")],
            }
        )
        api_names = [r[0] for r in topological_sort(pws)]
        assert len(api_names) == 3
        assert api_names.index("c") < api_names.index("a")
        assert api_names.index("c") < api_names.index("b")

    def test_fan_out(self):
        # "c" depends on both "a" and "b"
        pws = _make_pws(
            {
                "a": [("a", [], "pa")],
                "b": [("b", [], "pb")],
                "c": [("c", ["a", "b"], "pc")],
            }
        )
        api_names = [r[0] for r in topological_sort(pws)]
        assert len(api_names) == 3
        assert api_names.index("a") < api_names.index("c")
        assert api_names.index("b") < api_names.index("c")

    def test_diamond(self):
        # a -> b -> d
        # a -> c -> d
        pws = _make_pws(
            {
                "a": [("a", ["b", "c"], "pa")],
                "b": [("b", ["d"], "pb")],
                "c": [("c", ["d"], "pc")],
                "d": [("d", [], "pd")],
            }
        )
        api_names = [r[0] for r in topological_sort(pws)]
        assert len(api_names) == 4
        assert api_names[0] == "d"
        assert api_names.index("b") < api_names.index("a")
        assert api_names.index("c") < api_names.index("a")

    def test_multiple_providers_per_api(self):
        pws = _make_pws(
            {
                "inference": [
                    ("inference", [], "p1"),
                    ("inference", [], "p2"),
                ],
            }
        )
        result = topological_sort(pws)
        assert len(result) == 2
        assert all(r[0] == "inference" for r in result)
        provider_ids = [r[1].provider_id for r in result]
        assert "p1" in provider_ids
        assert "p2" in provider_ids

    def test_ignores_missing_dep(self):
        pws = _make_pws(
            {
                "responses": [("responses", ["inference", "nonexistent"], "p1")],
                "inference": [("inference", [], "p2")],
            }
        )
        result = topological_sort(pws)
        assert result[0][0] == "inference"
        assert result[1][0] == "responses"

    def test_cycle_raises(self):
        # Behavioral note: the legacy DFS implementation silently handled cycles
        # by skipping already-visited nodes. The graphlib.TopologicalSorter version
        # correctly raises an exception for cycles. This test documents the intended
        # new behavior.
        pws = _make_pws(
            {
                "a": [("a", ["b"], "pa")],
                "b": [("b", ["a"], "pb")],
            }
        )
        try:
            topological_sort(pws)
            raise AssertionError("Expected CycleError for circular dependency")
        except RuntimeError as e:
            assert "Failed to" in str(e)
            assert "circular dependency" in str(e)
