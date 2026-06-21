import os

from backward_traversal import route_finder
from backward_traversal.route_finder import Endpoint
from backward_traversal.runner import chainless_caller_walker as caller_walker
from vbt.reach import route as route_mod
from vbt.reach.route import RouteEngine


def _engine(tmp_path, *, job_id=None):
    eng = RouteEngine(tmp_path, tmp_path / "file_call_graph.json", job_id=job_id)
    eng._reverse_adj = {}
    eng._node_type = {}
    return eng


def test_db_backed_route_engine_defaults_to_bfs(monkeypatch, tmp_path):
    seen = []

    def fake_find_routes(*args, **kwargs):
        seen.append(kwargs.get("use_bfs"))
        return {"reachable": False, "routes": []}

    monkeypatch.delenv("VBT_BFS_FOREST", raising=False)
    monkeypatch.setattr(route_mod, "find_routes", fake_find_routes)

    _engine(tmp_path, job_id="job-1").routes(
        Endpoint("aa71", "asm"),
        Endpoint("af71", "asm"),
    )

    assert seen == [True]
    assert os.environ.get("VBT_BFS_FOREST") is None


def test_route_engine_preserves_explicit_bfs_setting(monkeypatch, tmp_path):
    seen = []

    def fake_find_routes(*args, **kwargs):
        seen.append(kwargs.get("use_bfs"))
        return {"reachable": False, "routes": []}

    monkeypatch.setenv("VBT_BFS_FOREST", "0")
    monkeypatch.setattr(route_mod, "find_routes", fake_find_routes)

    _engine(tmp_path, job_id="job-1").routes(
        Endpoint("aa71", "asm"),
        Endpoint("af71", "asm"),
    )

    assert seen == [None]
    assert os.environ.get("VBT_BFS_FOREST") == "0"


def test_non_db_route_engine_keeps_legacy_default(monkeypatch, tmp_path):
    seen = []

    def fake_find_routes(*args, **kwargs):
        seen.append(kwargs.get("use_bfs"))
        return {"reachable": False, "routes": []}

    monkeypatch.delenv("VBT_BFS_FOREST", raising=False)
    monkeypatch.setattr(route_mod, "find_routes", fake_find_routes)

    _engine(tmp_path).routes(
        Endpoint("aa71", "asm"),
        Endpoint("af71", "asm"),
    )

    assert seen == [None]
    assert os.environ.get("VBT_BFS_FOREST") is None


def test_db_backed_route_engine_passes_parent_distances(monkeypatch, tmp_path):
    seen = []
    eng = _engine(tmp_path, job_id="job-1")
    eng._fwd_file_adj = {"aa71": {"dw730000"}, "dw730000": {"af71"}}

    def fake_find_routes(*args, **kwargs):
        seen.append(kwargs.get("parent_distances"))
        return {"reachable": False, "routes": []}

    monkeypatch.delenv("VBT_BFS_FOREST", raising=False)
    monkeypatch.setattr(route_mod, "find_routes", fake_find_routes)

    eng.routes(Endpoint("aa71", "asm"), Endpoint("af71", "asm"))

    assert seen == [{"aa71": 0, "dw730000": 1, "af71": 2}]


def test_find_routes_skips_forest_when_parent_distance_excludes_child(monkeypatch, tmp_path):
    def fail_walk(*args, **kwargs):
        raise AssertionError("caller forest should not be built")

    monkeypatch.setattr(caller_walker, "walk_callers_bfs", fail_walk)

    res = route_finder.find_routes(
        Endpoint("aa71", "asm"),
        Endpoint("af71", "asm"),
        blueprint_dir=tmp_path,
        graph_file=tmp_path / "file_call_graph.json",
        reverse_adj={},
        node_type={},
        use_bfs=True,
        max_len=4,
        parent_distances={"aa71": 0, "dw730000": 1},
    )

    assert res["reachable"] is False
    assert res["stats"]["pruned_by_parent_distance"] is True


def test_bfs_parent_distance_prunes_before_bridge(monkeypatch, tmp_path):
    def fail_bridge(*args, **kwargs):
        raise AssertionError("bridge should not run for impossible parent-distance node")

    monkeypatch.setattr(caller_walker, "_bridge_for", fail_bridge)

    out = caller_walker.walk_callers_bfs(
        "af71",
        "asm",
        tmp_path,
        None,
        {"AF71": [("faraway", "CALL AF71")]},
        {"af71": "asm", "faraway": "asm"},
        max_caller_depth=4,
        attach_conditions=False,
        parent_distances={"aa71": 0},
        max_route_len=4,
    )

    assert out == []


def test_db_backed_reachable_uses_fast_bounded_walk(monkeypatch, tmp_path):
    eng = _engine(tmp_path, job_id="job-1")
    eng._reverse_adj = {"AF71": [("aa71", "CALL AF71")]}
    eng._node_type = {"af71": "asm", "aa71": "asm"}
    eng._fwd_file_adj = {"aa71": {"af71"}}

    def fail_find_routes(*args, **kwargs):
        raise AssertionError("reachable() should not build full routes")

    def fake_bridge_for(source_type, callee_type):
        def bridge(source_stem, callee_stem, blueprint_dir, asm_dir):
            return [{"routine": "AA71", "line": 10}]
        return bridge

    monkeypatch.setattr(route_mod, "find_routes", fail_find_routes)
    monkeypatch.setattr(caller_walker, "_bridge_for", fake_bridge_for)

    assert eng.reachable(Endpoint("aa71", "asm"), Endpoint("af71", "asm")) is True


def test_db_backed_reachable_prunes_excluded_child_without_bridge(monkeypatch, tmp_path):
    eng = _engine(tmp_path, job_id="job-1")
    eng._reverse_adj = {"AF71": [("faraway", "CALL AF71")]}
    eng._node_type = {"af71": "asm", "faraway": "asm", "aa71": "asm"}
    eng._fwd_file_adj = {"aa71": {"dw730000"}}

    def fail_bridge_for(*args, **kwargs):
        raise AssertionError("excluded child should not reach bridge verification")

    monkeypatch.setattr(caller_walker, "_bridge_for", fail_bridge_for)

    assert eng.reachable(Endpoint("aa71", "asm"), Endpoint("af71", "asm")) is False


def test_db_backed_reachable_uses_fn_graph_index_before_bounded_walk(monkeypatch, tmp_path):
    eng = _engine(tmp_path, job_id="job-1")
    eng._fn_graph_adj = {("aa71", "aa71"): {("af71", "af71", 10, True)}}
    eng._fn_graph_adj_loaded = True

    def fail_bounded_walk(*args, **kwargs):
        raise AssertionError("fn_graph reachability should avoid bounded caller walk")

    monkeypatch.setattr(route_mod, "is_reachable_bounded", fail_bounded_walk)

    assert eng.reachable(Endpoint("aa71", "asm"), Endpoint("af71", "asm")) is True


def test_db_backed_reachable_trusts_fn_graph_negative_when_endpoints_representable(
    monkeypatch, tmp_path
):
    # With graph_db v2 the fn-graph fans cross-file edges out to every owner, so its forward
    # closure matches the bridge walk — a FALSE for two representable endpoints is now
    # authoritative and PRUNES the off-chain over-keep. It must be trusted even when the
    # coarser file cone would (over-)approximate the pair as reachable.
    eng = _engine(tmp_path, job_id="job-1")
    eng._fn_graph_adj = {("aa71", "aa71"): {("dw730000", "dw730000", 10, True)}}
    eng._fn_graph_adj_loaded = True
    eng._fwd_file_adj = {"aa71": {"af71"}}  # file cone WOULD say reachable

    def fail_bounded_walk(*args, **kwargs):
        raise AssertionError("reachable() must not run the backward bounded walk")

    monkeypatch.setattr(route_mod, "is_reachable_bounded", fail_bounded_walk)

    # both asm endpoints are fn-graph nodes; no path → trusted False (NOT the file cone's True)
    assert eng.reachable(Endpoint("aa71", "asm"), Endpoint("af71", "asm")) is False
    # path present → True
    assert eng.reachable(Endpoint("aa71", "asm"), Endpoint("dw730000", "asm")) is True


def test_db_backed_reachable_fn_graph_matches_cpp_tail_variant(monkeypatch, tmp_path):
    eng = _engine(tmp_path, job_id="job-1")
    eng._fn_graph_adj = {("aa71", "aa71"): {("dw710000", "processACreditTransaction", 10, True)}}
    eng._fn_graph_adj_loaded = True

    def fail_bounded_walk(*args, **kwargs):
        raise AssertionError("fn_graph function variant should avoid bounded caller walk")

    monkeypatch.setattr(route_mod, "is_reachable_bounded", fail_bounded_walk)

    assert eng.reachable(
        Endpoint("aa71", "asm"),
        Endpoint("dw710000", "cpp", "DW71::processACreditTransaction"),
    ) is True


def test_db_backed_reachable_falls_back_to_file_reach_when_fn_graph_endpoint_is_unknown(
    monkeypatch, tmp_path
):
    # The endpoint has no fn-graph node (cpp parent without a function), so the fn-graph
    # fast-accept cannot fire. The authoritative decision is forward FILE reachability over
    # the corpus file-call-graph — NOT the (removed) backward bounded walk.
    eng = _engine(tmp_path, job_id="job-1")
    eng._fn_graph_adj = {("aa71", "aa71"): {("af71", "af71", 10, True)}}
    eng._fn_graph_adj_loaded = True
    eng._fwd_file_adj = {"aa71": {"af71"}}

    def fail_bounded_walk(*args, **kwargs):
        raise AssertionError("reachable() must not run the backward bounded walk")

    monkeypatch.setattr(route_mod, "is_reachable_bounded", fail_bounded_walk)

    # file-reachable → kept
    assert eng.reachable(Endpoint("aa71", "cpp"), Endpoint("af71", "asm")) is True
    # not file-reachable → dropped (no fn-graph node, no file path)
    assert eng.reachable(Endpoint("aa71", "cpp"), Endpoint("zz99", "asm")) is False
