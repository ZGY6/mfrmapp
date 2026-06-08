"""MFRMSight 包版核心冒烟测试 (v0.8.0)

测试 mfrmapp.engine.MFRMEngine — 与根目录 test_engine.py 并列。
运行: cd mfrmapp && uv run python -m pytest tests/ -v
"""
import sys
from pathlib import Path
import numpy as np
import pytest

# 确保包源码在 sys.path
PACKAGE_ROOT = Path(__file__).parent.parent
SRC = PACKAGE_ROOT / "src"
sys.path.insert(0, str(SRC))

from mfrmapp.engine import parse_facets_txt, MFRMEngine

# 测试数据路径: 相对于项目根目录
PROJECT_ROOT = PACKAGE_ROOT.parent
TEST_DATA = PROJECT_ROOT / "2026Raterbias10.txt"

# Facets 参考值
FACETS_REF = {
    "N": 256, "ns": 4, "nr": 8, "nc": 2, "ni": 4,
    "obs_mean": 10.62, "var_exp": 88.10,
}

# ═══════════════════════════════════════════════════
# 数据解析测试
# ═══════════════════════════════════════════════════

class TestParseFacetsTxt:
    """parse_facets_txt() 解析器测试"""

    @pytest.fixture(scope="class")
    def data(self):
        assert TEST_DATA.exists(), f"测试数据文件 {TEST_DATA} 不存在"
        return parse_facets_txt(str(TEST_DATA))

    def test_basic_dimensions(self, data):
        assert data["N"] == FACETS_REF["N"]
        assert data["n_s"] == FACETS_REF["ns"]
        assert data["n_r"] == FACETS_REF["nr"]
        assert data["n_c"] == FACETS_REF["nc"]
        assert data["n_i"] == FACETS_REF["ni"]

    def test_labels_correct(self, data):
        assert data["s_labels"] == ["Student1","Student2","Student3","Student4"]
        assert data["r_labels"][:8] == [f"Rater{i}" for i in range(1,9)]
        assert data["c_labels"] == ["comp","inte"]
        assert data["i_labels"] == ["Item1","Item2","Item3","Item4"]

    def test_noncentered(self, data):
        assert data["noncentered"] == 1

    def test_data_integrity(self, data):
        raw = data["raw"]
        assert raw.shape[0] == FACETS_REF["N"]
        assert raw.shape[1] >= 5
        for col in range(4):
            assert raw[:, col].min() >= 1


# ═══════════════════════════════════════════════════
# 引擎核心测试
# ═══════════════════════════════════════════════════

class TestMFRMEngine:
    """MFRMEngine 核心功能测试"""

    @pytest.fixture(scope="class")
    def engine(self):
        d = parse_facets_txt(str(TEST_DATA))
        return MFRMEngine(d).fit()

    @pytest.fixture(scope="class")
    def report(self, engine):
        return engine.report()

    # ── 收敛 ──
    def test_obs_exp_mean_match(self, engine):
        obs, exp = engine.scores.mean(), engine.exp_scores.mean()
        assert abs(obs - exp) < 0.2, f"ObsMean={obs:.3f} ExpMean={exp:.3f}"

    def test_var_exp_close_to_facets(self, report):
        ve = report["summary"]["var_exp"]
        assert abs(ve - FACETS_REF["var_exp"]) < 5, f"VarExp={ve}% vs Facets {FACETS_REF['var_exp']}%"

    def test_var_exp_reasonable(self, report):
        assert report["summary"]["var_exp"] >= 85

    # ── 参数 ──
    def test_student_order(self, report):
        rows = report["facets"]["students"]["rows"]
        s3 = next(r["meas"] for r in rows if r["label"] == "Student3")
        s4 = next(r["meas"] for r in rows if r["label"] == "Student4")
        assert s4 > s3

    def test_item_order(self, report):
        rows = report["facets"]["items"]["rows"]
        i3 = next(r["meas"] for r in rows if r["label"] == "Item3")
        i4 = next(r["meas"] for r in rows if r["label"] == "Item4")
        assert i3 > i4

    def test_params_in_range(self, report):
        for key in ["students","raters","criteria","items"]:
            fd = report["facets"].get(key)
            if fd and fd["rows"]:
                for r in fd["rows"]:
                    assert -5 <= r["meas"] <= 20

    # ── 报告结构 ──
    def test_all_facets_present(self, report):
        for k in ["students","raters","criteria","items"]:
            assert k in report["facets"]

    def test_se_positive(self, report):
        for key in ["students","raters","criteria","items"]:
            fd = report["facets"].get(key)
            if fd and fd["rows"]:
                for r in fd["rows"]:
                    assert r["se"] > 0


# ═══════════════════════════════════════════════════
# 回归测试
# ═══════════════════════════════════════════════════

class TestBugRegression:
    """已知 BUG 回归测试 (包版)"""

    @pytest.fixture(scope="class")
    def engine(self):
        d = parse_facets_txt(str(TEST_DATA))
        return MFRMEngine(d).fit()

    def test_bug010_first_order_condition(self, engine):
        obs, exp = engine.scores.sum(), engine.exp_scores.sum()
        assert abs(obs - exp) / obs < 0.05

    def test_bug011_noncentered(self, engine):
        assert engine.noncentered == 1

    def test_bug012_s2l_exists(self):
        assert hasattr(MFRMEngine, '_s2l')

    def test_bug013_deterministic(self):
        """两次拟合结果一致"""
        d1 = parse_facets_txt(str(TEST_DATA))
        d2 = parse_facets_txt(str(TEST_DATA))
        e1 = MFRMEngine(d1).fit()
        e2 = MFRMEngine(d2).fit()
        assert np.allclose(e1.theta, e2.theta)
        assert np.allclose(e1.delta, e2.delta)
        assert np.allclose(e1.tau, e2.tau)
