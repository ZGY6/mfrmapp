#!/usr/bin/env python3
"""
MFRMSight 核心冒烟测试 (v0.8.0)
================================
使用 pytest 框架，依赖: mfrm_app.py + 2026Raterbias10.txt
运行: uv run pytest tests/ -v
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mfrm_app import parse_minifac_txt, Engine

TEST_DATA = PROJECT_ROOT / "2026Raterbias10.txt"

# ═══════════════════════════════════════════════════
# 参考值 (来自 Facets v4.5.0 对 2026Raterbias10.txt 的输出)
# ═══════════════════════════════════════════════════
FACETS_REF = {
    "N": 256, "ns": 4, "nr": 8, "nc": 2, "ni": 4,
    "mn": 2, "mx": 22,
    "obs_mean": 10.62,
    "var_exp": 88.10,
}

# ═══════════════════════════════════════════════════
# 数据解析测试
# ═══════════════════════════════════════════════════

class TestParseMinifac:
    """Minifac .txt 解析器测试"""

    @pytest.fixture(scope="class")
    def data(self):
        return parse_minifac_txt(str(TEST_DATA))

    def test_file_exists(self):
        """测试数据文件存在"""
        assert TEST_DATA.exists(), f"{TEST_DATA} 不存在"

    def test_basic_dimensions(self, data):
        """测试基本维度: N, ns, nr, nc, ni"""
        assert data["N"] == FACETS_REF["N"]
        assert data["ns"] == FACETS_REF["ns"]
        assert data["nr"] == FACETS_REF["nr"]
        assert data["nc"] == FACETS_REF["nc"]
        assert data["ni"] == FACETS_REF["ni"]

    def test_score_range(self, data):
        """测试分数范围: mn=2, mx=22 (R23)"""
        assert data["mn"] == FACETS_REF["mn"]
        assert data["mx"] == FACETS_REF["mx"]

    def test_raw_array_shape(self, data):
        """测试 raw 数组形状"""
        assert data["raw"].shape == (FACETS_REF["N"], 5)
        assert data["raw"].dtype in (np.dtype("int32"), np.dtype("int64"))

    def test_labels_not_empty(self, data):
        """测试标签非空且数量正确"""
        assert len(data["sl"]) == FACETS_REF["ns"]
        assert len(data["rl"]) == FACETS_REF["nr"]
        assert len(data["cl"]) == FACETS_REF["nc"]
        assert len(data["il"]) == FACETS_REF["ni"]
        assert all(s for s in data["sl"])
        assert all(r for r in data["rl"])
        assert all(c for c in data["cl"])
        assert all(i for i in data["il"])

    def test_labels_correct(self, data):
        """测试标签内容正确 (BUG-009 验证)"""
        assert data["sl"] == ["Student1", "Student2", "Student3", "Student4"]
        assert data["rl"] == ["Rater1", "Rater2", "Rater3", "Rater4",
                              "Rater5", "Rater6", "Rater7", "Rater8"]
        assert data["cl"] == ["comp", "inte"]
        assert data["il"] == ["Item1", "Item2", "Item3", "Item4"]

    def test_noncentered(self, data):
        """测试 Noncentered=1 正确解析 (BUG-011 验证)"""
        assert data["noncentered"] == 1

    def test_model_string(self, data):
        """测试 Model 字符串解析"""
        assert data.get("model") == "?,?,?,?,R23"

    def test_score_values_in_range(self, data):
        """测试所有分数在 [min, max] 范围内"""
        scores = data["raw"][:, -1]
        assert scores.min() >= data["mn"]
        assert scores.max() <= data["mx"]

    def test_facet_ids_start_at_1(self, data):
        """测试面向 ID 从 1 开始"""
        raw = data["raw"]
        for col in range(4):
            assert raw[:, col].min() >= 1


# ═══════════════════════════════════════════════════
# 引擎核心测试
# ═══════════════════════════════════════════════════

class TestEngineCore:
    """MFRM Engine 核心功能测试"""

    @pytest.fixture(scope="class")
    def engine(self):
        """创建并拟合好的引擎实例"""
        d = parse_minifac_txt(str(TEST_DATA))
        return Engine(d).fit()

    @pytest.fixture(scope="class")
    def report(self, engine):
        return engine.report()

    # ── 收敛性 ──

    def test_obs_exp_mean_match(self, engine):
        """测试 ObsMean ≈ ExpMean (Rasch 一阶条件, BUG-010 验证)"""
        obs = engine.sc.mean()
        exp = engine.es.mean()
        assert abs(obs - exp) < 0.1, (
            f"ObsMean={obs:.3f} ExpMean={exp:.3f} 差距过大"
        )

    def test_variance_explained_reasonable(self, report):
        """测试方差解释率 >= 85%"""
        ve = report["summary"]["ve"]
        assert ve >= 85, f"VarExp={ve}% 过低"

    def test_variance_explained_close_to_facets(self, report):
        """测试方差解释率接近 Facets 88.10%"""
        ve = report["summary"]["ve"]
        assert abs(ve - FACETS_REF["var_exp"]) < 5, (
            f"VarExp={ve}% vs Facets {FACETS_REF['var_exp']}% 差距过大"
        )

    def test_residual_std_reasonable(self, report):
        """测试标准化残差 SD 在合理范围 (约 1.0)"""
        zs = report["summary"]["zs"]
        assert 0.8 < zs < 1.5, f"StResSD={zs} 偏差过大"

    # ── 参数合理性 ──

    def test_student_params_not_zero(self, report):
        """测试学生参数不全为 0"""
        meas = [r["m"] for r in report["facets"]["students"]["rows"]]
        assert max(meas) - min(meas) > 0.01, "学生参数范围过窄"

    def test_rater_params_not_zero(self, report):
        """测试评分者参数不全为 0"""
        meas = [r["m"] for r in report["facets"]["raters"]["rows"]]
        assert max(meas) - min(meas) > 0.01, "评分者参数范围过窄"

    def test_student_order_preserved(self, report):
        """测试学生排序: Student4 > Student3 (已知 Facets 参考)"""
        rows = report["facets"]["students"]["rows"]
        s3 = next(r["m"] for r in rows if r["l"] == "Student3")
        s4 = next(r["m"] for r in rows if r["l"] == "Student4")
        assert s4 > s3, f"Student4({s4:.3f}) 应 > Student3({s3:.3f})"

    def test_item_order_preserved(self, report):
        """测试题目排序: Item3 最易 (meas最大), Item4 最难 (meas最小)"""
        rows = report["facets"]["items"]["rows"]
        i3 = next(r["m"] for r in rows if r["l"] == "Item3")
        i4 = next(r["m"] for r in rows if r["l"] == "Item4")
        assert i3 > i4, f"Item3({i3:.3f}) 应 > Item4({i4:.3f})"

    def test_params_in_reasonable_range(self, report):
        """测试参数在 [-5, 20] 合理范围内"""
        for facet_key in ["students", "raters", "criteria", "items"]:
            fd = report["facets"].get(facet_key)
            if fd and fd["rows"]:
                for r in fd["rows"]:
                    assert -5 <= r["m"] <= 20, (
                        f"{facet_key}/{r['l']} meas={r['m']:.3f} 超出范围"
                    )

    def test_se_positive(self, report):
        """测试标准误全为正"""
        for facet_key in ["students", "raters", "criteria", "items"]:
            fd = report["facets"].get(facet_key)
            if fd and fd["rows"]:
                for r in fd["rows"]:
                    assert r["se"] > 0, (
                        f"{facet_key}/{r['l']} SE <= 0: {r['se']}"
                    )

    # ── Fit 统计量 ──

    def test_infit_outfit_positive(self, report):
        """测试 Infit/Outfit 全为正"""
        for facet_key in ["students", "raters", "criteria", "items"]:
            fd = report["facets"].get(facet_key)
            if fd and fd["rows"]:
                for r in fd["rows"]:
                    assert r["inf"] > 0, f"{facet_key}/{r['l']} infit={r['inf']}"
                    assert r["otf"] > 0, f"{facet_key}/{r['l']} outfit={r['otf']}"

    def test_separation_positive(self, report):
        """测试 Separation 为正"""
        for facet_key in ["students", "raters", "criteria", "items"]:
            fd = report["facets"].get(facet_key)
            if fd and fd["rows"]:
                assert fd["sep"] > 0, f"{facet_key} Sep={fd['sep']} <= 0"

    # ── 报告结构 ──

    def test_report_has_all_facets(self, report):
        """测试报告包含 4 个面向"""
        assert "students" in report["facets"]
        assert "raters" in report["facets"]
        assert "criteria" in report["facets"]
        assert "items" in report["facets"]

    def test_report_summary_keys(self, report):
        """测试 summary 包含必要字段"""
        keys = ["N", "ns", "nr", "nc", "ni", "sc", "om", "em", "rs", "zs", "ve"]
        for k in keys:
            assert k in report["summary"], f"summary 缺少字段 {k}"


# ═══════════════════════════════════════════════════
# Bug 回归测试
# ═══════════════════════════════════════════════════

class TestBugRegression:
    """已知 Bug 的回归测试"""

    @pytest.fixture(scope="class")
    def engine(self):
        d = parse_minifac_txt(str(TEST_DATA))
        return Engine(d).fit()

    # BUG-009: 标签解析器
    def test_bug009_student_labels(self, engine):
        """Student 标签正确 (不是 Rater 或 Item)"""
        assert engine.sl[:engine.ns] == [
            "Student1", "Student2", "Student3", "Student4"
        ]

    def test_bug009_rater_labels(self, engine):
        """Rater 标签正确 (不是 'Rater5'-'Rater8' 跑到 Items)"""
        assert engine.rl[:engine.nr] == [
            "Rater1", "Rater2", "Rater3", "Rater4",
            "Rater5", "Rater6", "Rater7", "Rater8"
        ]

    # BUG-010: JMLE 收敛
    def test_bug010_first_order_condition(self, engine):
        """Rasch 一阶条件: Σobs = Σexp"""
        obs = engine.sc.sum()
        exp = engine.es.sum()
        ratio = abs(obs - exp) / obs
        assert ratio < 0.05, (
            f"一阶条件不满足: ObsSum={obs:.1f} ExpSum={exp:.1f} 差异={ratio:.1%}"
        )

    # BUG-011: Noncentered
    def test_bug011_noncentered_stored(self, engine):
        """noncentered 已存储"""
        assert hasattr(engine, 'noncentered')
        assert engine.noncentered == 1

    def test_bug011_no_excess_constraints(self, engine):
        """Noncentered=1 时 theta 不需要均值=0,
        但 delta(raters) 需要 (仅1个sum-to-zero约束)"""
        # delta(raters) sum ≈ 0 (被约束)
        assert abs(engine.dl.mean()) < 1e-10, (
            f"delta 均值应=0, 实际={engine.dl.mean():.6f}"
        )

    # BUG-012: PROX 初始化
    def test_bug012_s2l_exists(self, engine):
        """_s2l 方法存在"""
        assert hasattr(engine, '_s2l')

    def test_bug012_prox_not_trivial(self, engine):
        """PROX 初始化后参数不为全 0"""
        d_clean = parse_minifac_txt(str(TEST_DATA))
        e_prox = Engine(d_clean)
        e_prox._idx()
        # 直接跑 PROX
        e_prox._model_type = 'rating_scale'
        from mfrm_app import parse_minifac_txt as _pf
        e_prox._px()
        assert abs(e_prox.th).max() > 0.5, "PROX theta 过于平坦"

    # BUG-013: Sequential update
    def test_bug013_sequential_update(self, engine):
        """多次拟合结果一致 (确定性)"""
        d1 = parse_minifac_txt(str(TEST_DATA))
        d2 = parse_minifac_txt(str(TEST_DATA))
        e1 = Engine(d1).fit()
        e2 = Engine(d2).fit()
        # 参数应完全一致 (无随机性)
        assert np.allclose(e1.th, e2.th)
        assert np.allclose(e1.dl, e2.dl)
        assert np.allclose(e1.al, e2.al)
        assert np.allclose(e1.bt, e2.bt)
        assert np.allclose(e1.tu, e2.tu)
