"""
MFRMSight — 多面Rasch模型分析引擎
===================================
基于Andrich Rating Scale Model (1978)
Fisher-scoring JMLE估计, 分阶段PROX→JMLE
"""
import numpy as np
from pathlib import Path
from typing import Optional
import re


__version__ = "0.2.0"


def parse_facets_txt(filepath: str) -> dict:
    """解析Facets风格的.txt输入文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    data_rows, infos = [], [[], [], [], []]
    labels_ok, in_data, cur_facet, n_facets = False, False, 0, 4
    for line in lines:
        if line.lower().startswith("facets="): n_facets = int(line.split("=")[1])
        if line == "*": labels_ok, cur_facet = True, 0; continue
        if line.lower().startswith("labels="): continue
        if line.lower().startswith("data="): labels_ok = False; in_data = True; continue
        if labels_ok and not in_data:
            for fid in range(1, 5):
                if line.startswith(f"{fid},"): cur_facet = fid; break
            if any(line.startswith(f"{d},") for d in "1234"): continue
            if cur_facet and "," in line:
                p = [x.strip() for x in line.split(",", 1)]
                if p[0].isdigit(): infos[cur_facet - 1].append((int(p[0]), p[1]))
            continue
        if in_data:
            p = [x.strip() for x in line.split(",")]
            if len(p) >= n_facets + 1:
                try: data_rows.append([int(v) for v in p])
                except ValueError: pass
    raw = np.array(data_rows)
    return _build_dict(raw, n_facets,
                       [l for _, l in infos[0]], [l for _, l in infos[1]],
                       [l for _, l in infos[2]], [l for _, l in infos[3]])


def parse_excel(filepath: str) -> dict:
    """从Excel加载3面向数据"""
    import pandas as pd
    df_raw = pd.read_excel(filepath, header=None)
    header_row = 1
    for i in range(len(df_raw)):
        s = [str(v) for v in df_raw.iloc[i] if pd.notna(v)]
        if any("编号" in x or "评分人" in x for x in s):
            header_row = i; break
    col_names = [str(df_raw.iloc[header_row, j]) if pd.notna(df_raw.iloc[header_row, j]) else f"Col{j}"
                 for j in range(df_raw.shape[1])]
    s_set, c_set = set(), set()
    for cn in col_names[2:]:
        if "-" in cn: crit_name, stu_name = cn.rsplit("-", 1); c_set.add(crit_name); s_set.add(stu_name)
    s_list, c_list = sorted(s_set), sorted(c_set)
    n_s, n_c = len(s_list), len(c_list)
    data_rows = []
    rater_labels = []
    for ri in range(header_row + 1, len(df_raw)):
        row = df_raw.iloc[ri]
        if pd.isna(row.iloc[0]): continue
        r_label = str(row.iloc[1])
        r_num = _extract_num(r_label)
        for si in range(n_s):
            for ci in range(n_c):
                col = 2 + ci * n_s + si
                if col < df_raw.shape[1]:
                    v = row.iloc[col]
                    if pd.notna(v): data_rows.append([si + 1, r_num, ci + 1, 1, int(v)])
        rater_labels.append(r_label)
    raw = np.array(data_rows)
    return _build_dict(raw, 3, s_list, rater_labels, c_list, ["Item1"])


def _extract_num(s): return int(re.findall(r'\d+', str(s))[-1]) if re.findall(r'\d+', str(s)) else 1


def _build_dict(raw, n_facets, s_labels, r_labels, c_labels, i_labels):
    return {
        "raw": raw, "n_facets": n_facets, "N": len(raw),
        "n_s": int(raw[:, 0].max()), "n_r": int(raw[:, 1].max()),
        "n_c": int(raw[:, 2].max()), "n_i": int(raw[:, 3].max()) if n_facets >= 4 and raw.shape[1] > 3 else 1,
        "min_s": int(raw[:, -1].min()), "max_s": int(raw[:, -1].max()),
        "s_labels": s_labels or [f"Student{i+1}" for i in range(int(raw[:, 0].max()))],
        "r_labels": r_labels or [f"Rater{i+1}" for i in range(int(raw[:, 1].max()))],
        "c_labels": c_labels or [f"Criterion{i+1}" for i in range(int(raw[:, 2].max()))],
        "i_labels": i_labels or [f"Item{i+1}" for i in range(max(1, int(raw[:, 3].max()) if n_facets >= 4 and raw.shape[1] > 3 else 1))],
    }


class MFRMEngine:
    """多面Rasch Rating Scale模型估计器"""

    def __init__(self, data: dict):
        for k, v in data.items(): setattr(self, k, v)
        self._index()

    def _index(self):
        self.s_idx = self.raw[:, 0] - 1
        self.r_idx = self.raw[:, 1] - 1
        self.c_idx = np.clip(self.raw[:, 2] - 1, 0, self.n_c - 1)
        self.i_idx = self.raw[:, 3] - 1 if self.n_facets >= 4 else np.zeros(self.N, dtype=int)
        self.scores = self.raw[:, -1].astype(float)
        self.x = self.scores - self.min_s
        self.K = int(self.max_s - self.min_s)
        self.cats = np.arange(self.K + 1, dtype=float)
        self.obs_s = [np.where(self.s_idx == s)[0] for s in range(self.n_s)]
        self.obs_r = [np.where(self.r_idx == r)[0] for r in range(self.n_r)]
        self.obs_c = [np.where(self.c_idx == c)[0] for c in range(self.n_c)]
        self.obs_i = [np.where(self.i_idx == it)[0] for it in range(max(self.n_i, 1))]

    def _probs(self):
        intc = (self.theta[self.s_idx] - self.delta[self.r_idx]
                - self.alpha[self.c_idx] - self.beta[self.i_idx])
        logits = np.zeros((self.N, self.K + 1)); cum = np.zeros(self.N)
        for k in range(self.K): cum += intc - self.tau[k]; logits[:, k + 1] = cum
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits); return e / e.sum(axis=1, keepdims=True)

    def _ll(self, p):
        return sum(np.log(max(p[i, int(self.x[i])], 1e-300)) for i in range(self.N))

    def _prox(self):
        def _l(m, M): p = np.clip(m.mean() / M, 0.02, 0.98); return np.log(p / (1 - p))
        self.theta = np.array([_l(self.scores[self.obs_s[s]], self.max_s) for s in range(self.n_s)]) - self.theta.mean() if hasattr(self, 'theta') else np.zeros(self.n_s)
        self.theta = np.array([_l(self.scores[self.obs_s[s]], self.max_s) for s in range(self.n_s)]); self.theta -= self.theta.mean()
        self.delta = -np.array([_l(self.scores[self.obs_r[r]], self.max_s) for r in range(self.n_r)]); self.delta -= self.delta.mean()
        self.alpha = -np.array([_l(self.scores[self.obs_c[c]], self.max_s) for c in range(self.n_c)]); self.alpha -= self.alpha.mean()
        self.beta = -np.array([_l(self.scores[self.obs_i[it]], self.max_s) for it in range(self.n_i)]) if self.n_i > 1 else np.zeros(1); self.beta -= self.beta.mean()
        self.tau = np.array([-np.log(np.clip((self.scores >= self.min_s + k + 1).mean(), 0.02, 0.98) / np.clip((self.scores < self.min_s + k + 1).mean(), 0.02, 0.98)) for k in range(self.K)]); self.tau -= self.tau[0]
        p = self._probs(); U = np.clip(np.std(self.scores) / max(np.std(self.min_s + p @ self.cats), 0.01), 0.5, 3.0)
        self.theta *= U; self.delta *= U; self.alpha *= U; self.beta *= U; self.tau *= U

    def fit(self, p1=200, p2=300):
        self._prox()
        best = (-1e100, None)
        for phase, (n_it, rs, rd, dmp) in enumerate([(p1, 0.5, 0.99, 0.2), (p2, 0.05, 0.998, 0.15)]):
            for it in range(1, n_it + 1):
                rid = max(rs * (rd ** it), 0.001)
                self._up_theta(dmp, rid); self._up_delta(dmp, rid)
                self._up_alpha(dmp, rid); self._up_beta(dmp, rid)
                self._up_tau(dmp * (0.5 if phase == 0 else 0.3), rid * (10 if phase == 0 else 2))
                if it % 10 == 0:
                    p = self._probs(); ll = self._ll(p)
                    if ll > best[0]: best = (ll, (self.theta.copy(), self.delta.copy(), self.alpha.copy(), self.beta.copy(), self.tau.copy()))
        if best[1]: self.theta, self.delta, self.alpha, self.beta, self.tau = best[1]
        self._fin(); return self

    def _up(self, arr, obs, n, dmp, rid, sign):
        p = self._probs(); e = p @ self.cats; v = np.clip((p @ (self.cats**2)) - e**2, 0.001, None)
        for i in range(n):
            idx = obs[i]; arr[i] += dmp * (sign * self.x[idx].sum() - sign * e[idx].sum()) / (v[idx].sum() + rid)
        arr -= arr.mean(); arr[:] = np.clip(arr, -20, 20)

    def _up_theta(self, d, r): self._up(self.theta, self.obs_s, self.n_s, d, r, 1)
    def _up_delta(self, d, r): self._up(self.delta, self.obs_r, self.n_r, d, r, -1)
    def _up_alpha(self, d, r): self._up(self.alpha, self.obs_c, self.n_c, d, r, -1)
    def _up_beta(self, d, r):
        if self.n_i > 1: self._up(self.beta, self.obs_i, self.n_i, d, r, -1)

    def _up_tau(self, dmp, rid):
        p = self._probs()
        for k in range(self.K):
            o, e = (self.x >= k + 1).sum(), p[:, k + 1:].sum()
            pg = p[:, k + 1:].sum(axis=1); info = (pg * (1 - pg)).sum() + rid
            self.tau[k] += dmp * (e - o) / info
        self.tau -= self.tau[0]; self.tau[:] = np.clip(self.tau, -15, 25)

    def _fin(self):
        p = self._probs(); self.ll_final = self._ll(p)
        e = p @ self.cats; self.var_o = np.clip((p @ (self.cats**2)) - e**2, 0.001, None)
        self.exp_scores = self.min_s + e; self.resid = self.scores - self.exp_scores; self.z = self.resid / np.sqrt(self.var_o)
        self.var_exp = max(0, (np.var(self.scores, ddof=1) - np.var(self.resid, ddof=1)) / np.var(self.scores, ddof=1) * 100)

    def _facet(self, obs, param, labels, n):
        rows, ms, ses = [], [], []
        for p in range(n):
            idx = obs[p]
            if len(idx) == 0: continue
            total, nm = self.scores[idx].sum(), len(idx)
            zz, vv = self.z[idx], self.var_o[idx]
            w = vv.sum(); infit = (zz**2 * vv).sum() / max(w, 1e-10); outfit = (zz**2).sum() / max(nm, 1)
            se = 1.0 / np.sqrt(max(vv.sum(), 1e-10))
            rows.append({"label": labels[p], "total": int(total), "obs_avg": round(self.scores[idx].mean(), 2), "fair_avg": round(self.exp_scores[idx].mean(), 2), "meas": round(param[p], 3), "se": round(se, 3), "infit": round(infit, 3), "outfit": round(outfit, 3)})
            ms.append(param[p]); ses.append(se)
        ma, sa = np.array(ms), np.array(ses)
        vo = np.var(ma, ddof=1) if len(ma) > 1 else 0.001; me = np.mean(sa**2); vt = max(vo - me, 0.001)
        return rows, np.sqrt(vt / me) if me > 0 else 0, vt / (vt + me)

    def report(self):
        if not hasattr(self, 'exp_scores'): raise RuntimeError("请先运行 fit()")
        r = {"summary": {"N": self.N, "n_s": self.n_s, "n_r": self.n_r, "n_c": self.n_c, "n_i": self.n_i, "score_range": f"{self.min_s}-{self.max_s}", "obs_mean": round(float(self.scores.mean()), 3), "exp_mean": round(float(self.exp_scores.mean()), 3), "resid_sd": round(float(self.resid.std(ddof=1)), 4), "stres_sd": round(float(self.z.std(ddof=1)), 4), "var_exp": round(self.var_exp, 2), "ll": round(self.ll_final, 2)}, "facets": {}}
        for name, obs, param, labels, n in [("students", self.obs_s, self.theta, self.s_labels[:self.n_s], self.n_s), ("raters", self.obs_r, self.delta, self.r_labels[:self.n_r], self.n_r), ("criteria", self.obs_c, self.alpha, self.c_labels[:self.n_c], self.n_c)]:
            rows, sep, rel = self._facet(obs, param, labels, n)
            r["facets"][name] = {"rows": rows, "separation": round(sep, 2), "reliability": round(rel, 3)}
        if self.n_i > 1:
            rows, sep, rel = self._facet(self.obs_i, self.beta, self.i_labels[:self.n_i], self.n_i)
            r["facets"]["items"] = {"rows": rows, "separation": round(sep, 2), "reliability": round(rel, 3)}
        return r

    def print(self):
        r = self.report(); s = r["summary"]
        print(f"\n{'='*72}\nMFRMSight — 多面Rasch模型分析报告\n{'='*72}")
        print(f"{s['N']}条 | {s['n_s']}学生×{s['n_r']}评分者×{s['n_c']}标准×{s['n_i']}题目 | {s['score_range']}分")
        print(f"方差解释: {s['var_exp']}% | LL: {s['ll']:.0f}")
        print(f"ObsMean={s['obs_mean']:.3f} ExpMean={s['exp_mean']:.3f} ResidSD={s['resid_sd']:.4f} StResSD={s['stres_sd']:.4f}")
        for fn, fd in r["facets"].items():
            if not fd["rows"]: continue
            print(f"\n{'─'*72}\n{fn} — Sep={fd['separation']:.2f} Rel={fd['reliability']:.3f}\n{'─'*72}")
            print(f"{'':<16} {'Total':>7} {'ObsAvg':>7} {'FairAvg':>7} {'Meas':>7} {'SE':>6} {'Infit':>6} {'Outfit':>6}")
            print("-" * 72)
            for row in fd["rows"]:
                print(f"{row['label']:<16} {row['total']:>7.0f} {row['obs_avg']:>7.2f} {row['fair_avg']:>7.2f} {row['meas']:>7.3f} {row['se']:>6.3f} {row['infit']:>6.3f} {row['outfit']:>6.3f}")

    def to_excel(self, path):
        import pandas as pd
        r = self.report(); dfs = {n: pd.DataFrame(f["rows"]) for n, f in r["facets"].items()}
        dfs["summary"] = pd.DataFrame([r["summary"]])
        with pd.ExcelWriter(path) as w:
            for n, d in dfs.items(): d.to_excel(w, sheet_name=n, index=False)
        print(f"[OK] {path}")

    def to_word(self, path):
        try:
            from docx import Document
            from docx.shared import Pt
            from docx.enum.text import WD_ALIGN_PARAGRAPH
        except ImportError:
            print("[!] pip install python-docx"); return
        r = self.report(); doc = Document()
        t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = t.add_run("MFRM Rating Scale 模型分析报告"); run.bold = True; run.font.size = Pt(18)
        s = r["summary"]
        doc.add_paragraph(f"反应数: {s['N']} | {s['n_s']}学生×{s['n_r']}评分者×{s['n_c']}标准×{s['n_i']}题目\n分数: {s['score_range']} | Rasch解释方差: {s['var_exp']}%\n观察均值: {s['obs_mean']:.3f} | 残差SD: {s['resid_sd']:.3f} | StRes SD: {s['stres_sd']:.3f}")
        for fn, fd in r["facets"].items():
            if not fd["rows"]: continue
            doc.add_heading(f"{fn} (Sep={fd['separation']:.2f}, Rel={fd['reliability']:.3f})", level=2)
            h = ["元素", "总分", "ObsAvg", "FairAvg", "Meas", "SE", "Infit", "Outfit"]
            tbl = doc.add_table(rows=1 + len(fd["rows"]), cols=8); tbl.style = "Table Grid"
            for i, hh in enumerate(h): tbl.rows[0].cells[i].text = hh
            for ri, row in enumerate(fd["rows"]):
                for ci, k in enumerate(["label", "total", "obs_avg", "fair_avg", "meas", "se", "infit", "outfit"]):
                    v = row[k]; tbl.rows[ri + 1].cells[ci].text = f"{v:.1f}" if isinstance(v, float) and k != "label" else f"{v:.0f}" if k == "total" else str(v)
        doc.save(path); print(f"[OK] {path}")
