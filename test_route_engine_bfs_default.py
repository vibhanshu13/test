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
