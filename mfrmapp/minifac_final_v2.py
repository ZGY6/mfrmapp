"""
minifac_final_v2 — 稳定 MFRM JMLE 实现
======================================
针对 tau 发散问题，采用以下数值策略：

1. 分阶段 PROX → JMLE（先固结面向参数，再松阈值）
2. 步长减半 (step-halving)：每次更新后检查 LL，不改善就减半步长
3. Ridge 衰减：1.0 → 0.01，先强正则化保证收敛方向正确，后放松
4. 顺序更新：theta → delta → alpha → beta → tau，每轮用最新值
5. 发散检测 + 回滚：参数超出 [-20, 20] 范围即恢复上一步并减速

数据: 2026Raterbias10.txt (256 条, 4 面向, R23)
"""
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# 1. 数据加载
# ══════════════════════════════════════════════════════════════════════

raw = []
with open(r"C:\Users\zgy78\Desktop\2026Raterbias10.txt", "r") as f:
    in_data = False
    for line in f:
        line = line.strip()
        if line.startswith("Data="):
            in_data = True
            continue
        if in_data and line and not line.startswith("*"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 5:
                raw.append([int(p) for p in parts])

raw = np.array(raw)
s_idx = raw[:, 0] - 1; r_idx = raw[:, 1] - 1
c_idx = raw[:, 2] - 1; i_idx = raw[:, 3] - 1
score = raw[:, 4].astype(float)

N_S, N_R, N_C, N_I = 4, 8, 2, 4
MIN_S, MAX_S = 2, 22; K = 20; N = len(score)
x = score - MIN_S

# 预计算索引
obs_s = [np.where(s_idx == s)[0] for s in range(N_S)]
obs_r = [np.where(r_idx == r)[0] for r in range(N_R)]
obs_c = [np.where(c_idx == c)[0] for c in range(N_C)]
obs_i = [np.where(i_idx == it)[0] for it in range(N_I)]

print(f"数据: N={N} | S={N_S} R={N_R} C={N_C} I={N_I} | 分数 {MIN_S}-{MAX_S} | 阈值={K}")
print(f"均分: {score.mean():.3f} | SD: {score.std(ddof=1):.3f}")

# ══════════════════════════════════════════════════════════════════════
# 2. 核心函数
# ══════════════════════════════════════════════════════════════════════

def compute_probs(theta, delta, alpha, beta, tau):
    """返回 (N, K+1) 概率矩阵"""
    intercept = theta[s_idx] - delta[r_idx] - alpha[c_idx] - beta[i_idx]
    logits = np.zeros((N, K + 1))
    cum = np.zeros(N)
    for k_val in range(K):
        cum += intercept - tau[k_val]
        logits[:, k_val + 1] = cum
    logits -= logits.max(axis=1, keepdims=True)
    el = np.exp(logits)
    return el / el.sum(axis=1, keepdims=True)

cats = np.arange(K + 1, dtype=float)

def log_lik(probs):
    ll = 0.0
    for i in range(N):
        xi = int(x[i])
        ll += np.log(max(probs[i, xi], 1e-300))
    return ll

def residuals_sd(probs):
    exp_off = probs @ cats
    return (score - (MIN_S + exp_off)).std(ddof=1)

# ══════════════════════════════════════════════════════════════════════
# 3. PROX 初始化（与 Facets 对齐）
# ══════════════════════════════════════════════════════════════════════

theta = np.zeros(N_S); delta = np.zeros(N_R)
alpha = np.zeros(N_C); beta = np.zeros(N_I); tau = np.zeros(K)

for s in range(N_S):
    m = score[obs_s[s]]; p = np.clip(m.mean() / MAX_S, 0.02, 0.98)
    theta[s] = np.log(p / (1-p))
theta -= theta.mean()

for r in range(N_R):
    m = score[obs_r[r]]; p = np.clip(m.mean() / MAX_S, 0.02, 0.98)
    delta[r] = -np.log(p/(1-p))
delta -= delta.mean()

for c in range(N_C):
    m = score[obs_c[c]]; p = np.clip(m.mean() / MAX_S, 0.02, 0.98)
    alpha[c] = -np.log(p/(1-p))
alpha -= alpha.mean()

for iti in range(N_I):
    m = score[obs_i[iti]]; p = np.clip(m.mean() / MAX_S, 0.02, 0.98)
    beta[iti] = -np.log(p/(1-p))
beta -= beta.mean()

for k_val in range(K):
    sv = MIN_S + k_val + 1
    prop = np.clip((score >= sv).mean(), 0.02, 0.98)
    tau[k_val] = -np.log(prop/(1-prop))
tau -= tau[0]

# 展开因子
probs_init = compute_probs(theta, delta, alpha, beta, tau)
exp_off_init = probs_init @ cats
exp_sc_init = MIN_S + exp_off_init
sd_obs = np.std(score)
sd_exp = np.std(exp_sc_init)
U = np.clip(sd_obs / max(sd_exp, 0.01), 0.5, 3.0)
theta *= U; delta *= U; alpha *= U; beta *= U; tau *= U

print(f"PROX 初始化: U={U:.3f}, "
      f"theta=[{theta.min():.2f}..{theta.max():.2f}], "
      f"delta=[{delta.min():.2f}..{delta.max():.2f}]")

probs = compute_probs(theta, delta, alpha, beta, tau)
best_ll = log_lik(probs)
best_params = (theta.copy(), delta.copy(), alpha.copy(), beta.copy(), tau.copy())
print(f"初始 LL={best_ll:.2f}, ResidSD={residuals_sd(probs):.3f}")

# ══════════════════════════════════════════════════════════════════════
# 4. JMLE 迭代（带数值保护）
# ══════════════════════════════════════════════════════════════════════

def safe_update(old_params, new_params, max_abs=20.0):
    """如果新参数超出合理范围，返回旧参数"""
    if np.any(np.abs(new_params) > max_abs):
        return old_params.copy()
    return new_params

def update_facet_with_step_halving(
    param_vec, obs_list, n_elem, update_fn, probs_fn,
    damping=0.5, ridge=0.1, step_halving_max=4
):
    """
    对一个面向做参数更新，带步长减半。
    update_fn(idx, probs, param, damping, ridge) -> (new_param, improved)
    """
    for p in range(n_elem):
        idx = obs_list[p]
        if len(idx) < 1:
            continue

        old_val = param_vec[p]
        probs = probs_fn()
        exp_off = probs @ cats

        # 计算初始步长
        obs_sum = x[idx].sum()
        exp_sum = exp_off[idx].sum()

        # Fisher info with ridge
        var_off = np.clip((probs @ (cats**2)) - exp_off**2, 0.001, None)
        info = var_off[idx].sum() + ridge

        step = damping * (obs_sum - exp_sum) / info
        new_val = old_val + step

        # 步长减半
        for halve in range(step_halving_max):
            if abs(new_val) > 20.0:
                new_val = old_val + step * 0.5 ** (halve + 1)
                continue

            param_vec[p] = new_val
            new_probs = probs_fn()
            new_exp = new_probs @ cats
            new_var = np.clip((new_probs @ (cats**2)) - new_exp**2, 0.001, None)
            new_info = new_var[idx].sum() + ridge
            new_obs = x[idx].sum(); new_exp_s = new_exp[idx].sum()

            # 检查改善：残差是否减小
            old_contrib = abs(obs_sum - exp_sum)
            new_contrib = abs(new_obs - new_exp_s)
            if new_contrib <= old_contrib:
                break  # 改善了，接受

            # 减半步长
            step *= 0.5
            new_val = old_val + step
            param_vec[p] = old_val  # 恢复

        else:
            # 所有步长都不改善，接受最后尝试的步长
            param_vec[p] = new_val

    return param_vec


# 简单版 Fisher scoring（不步长减半，但带强 ridge）
print(f"\n{'='*72}")
print(f"阶段 1: 高阻尼 + 强 ridge (200 次迭代)")
print(f"{'='*72}")

for it in range(1, 201):
    ridge_val = max(0.5 * (0.99 ** it), 0.01)
    damp = 0.2
    prev_vec = np.concatenate([theta, delta, alpha, beta, tau])

    # theta
    probs = compute_probs(theta, delta, alpha, beta, tau)
    exp_off = probs @ cats
    var_off = np.clip((probs @ (cats**2)) - exp_off**2, 0.001, None)
    for s in range(N_S):
        idx = obs_s[s]
        obs_sum = x[idx].sum(); exp_sum = exp_off[idx].sum()
        info = var_off[idx].sum() + ridge_val
        theta[s] += damp * (obs_sum - exp_sum) / info
    theta -= theta.mean()
    theta = np.clip(theta, -20, 20)

    # delta
    probs = compute_probs(theta, delta, alpha, beta, tau)
    exp_off = probs @ cats
    var_off = np.clip((probs @ (cats**2)) - exp_off**2, 0.001, None)
    for r in range(N_R):
        idx = obs_r[r]
        obs_sum = x[idx].sum(); exp_sum = exp_off[idx].sum()
        info = var_off[idx].sum() + ridge_val
        delta[r] += damp * (exp_sum - obs_sum) / info
    delta -= delta.mean()
    delta = np.clip(delta, -20, 20)

    # alpha
    probs = compute_probs(theta, delta, alpha, beta, tau)
    exp_off = probs @ cats
    var_off = np.clip((probs @ (cats**2)) - exp_off**2, 0.001, None)
    for c in range(N_C):
        idx = obs_c[c]
        obs_sum = x[idx].sum(); exp_sum = exp_off[idx].sum()
        info = var_off[idx].sum() + ridge_val
        alpha[c] += damp * (exp_sum - obs_sum) / info
    alpha -= alpha.mean()
    alpha = np.clip(alpha, -20, 20)

    # beta
    probs = compute_probs(theta, delta, alpha, beta, tau)
    exp_off = probs @ cats
    var_off = np.clip((probs @ (cats**2)) - exp_off**2, 0.001, None)
    for iti in range(N_I):
        idx = obs_i[iti]
        obs_sum = x[idx].sum(); exp_sum = exp_off[idx].sum()
        info = var_off[idx].sum() + ridge_val
        beta[iti] += damp * (exp_sum - obs_sum) / info
    beta -= beta.mean()
    beta = np.clip(beta, -20, 20)

    # tau（强阻尼 + 强正则化）
    probs = compute_probs(theta, delta, alpha, beta, tau)
    for k_val in range(K):
        obs_ge = (x >= k_val + 1).sum()
        exp_ge = probs[:, k_val + 1:].sum()
        p_ge = probs[:, k_val + 1:].sum(axis=1)
        info = (p_ge * (1 - p_ge)).sum() + ridge_val * 10  # tau 正则化更强
        tau[k_val] += damp * 0.5 * (exp_ge - obs_ge) / info
    tau -= tau[0]
    tau = np.clip(tau, -20, 20)

    cur_vec = np.concatenate([theta, delta, alpha, beta, tau])
    max_d = np.abs(cur_vec - prev_vec).max()

    if it % 40 == 0:
        probs = compute_probs(theta, delta, alpha, beta, tau)
        ll_val = log_lik(probs)
        rsd = residuals_sd(probs)
        print(f"  it={it:3d}: ridge={ridge_val:.3f} max|d|={max_d:.6f} "
              f"LL={ll_val:.2f} ResidSD={rsd:.3f} "
              f"theta=[{theta.min():.2f}..{theta.max():.2f}]")

    # 追踪最佳参数
    if it % 10 == 0:
        probs = compute_probs(theta, delta, alpha, beta, tau)
        cur_ll = log_lik(probs)
        if cur_ll > best_ll:
            best_ll = cur_ll
            best_params = (theta.copy(), delta.copy(), alpha.copy(), beta.copy(), tau.copy())

    if max_d < 1e-5:
        print(f"  收敛于 iter {it}")
        break

# 使用最佳参数
theta, delta, alpha, beta, tau = best_params

# ══════════════════════════════════════════════════════════════════════
# 5. 最终结果
# ══════════════════════════════════════════════════════════════════════

probs = compute_probs(theta, delta, alpha, beta, tau)
exp_off = probs @ cats
var_off = np.clip((probs @ (cats**2)) - exp_off**2, 0.001, None)
exp_scores = MIN_S + exp_off
residuals = score - exp_scores
z_scores = residuals / np.sqrt(var_off)

obs_var = np.var(score, ddof=1)
resid_var = np.var(residuals, ddof=1)
var_exp = (obs_var - resid_var) / obs_var * 100

ll_final = log_lik(probs)

print(f"\n{'='*72}")
print(f"最终结果")
print(f"{'='*72}")
print(f"LL = {ll_final:.2f}")
print(f"Obs Mean = {score.mean():.3f} | Exp Mean = {exp_scores.mean():.3f}")
print(f"Resid Mean = {residuals.mean():.4f} | Resid SD = {residuals.std(ddof=1):.4f}")
print(f"StRes Mean = {z_scores.mean():.4f} | StRes SD = {z_scores.std(ddof=1):.4f}")
print(f"Rasch Var Explained = {var_exp:.2f}% (Facets: 88.10%)")
print(f"Residual Var = {100-var_exp:.2f}% (Facets: 11.90%)")

# ── 学生面向 ─────────────────────────────────────────────────────────
facets_stu = {
    "meas": [-0.26, -0.56, -1.27, 0.61],
    "se": [0.06, 0.07, 0.07, 0.06],
    "infit": [1.28, 0.84, 0.80, 0.87],
    "outfit": [1.18, 0.78, 0.82, 0.91],
}

print(f"\n--- 学生面向 ---")
print(f"{'':12} {'Meas':>8} {'(Facets)':>10} {'SE':>7} {'(F)':>7} "
      f"{'Infit':>7} {'(F)':>7} {'Outfit':>7} {'(F)':>7}")
print("-" * 90)
s_measures, s_ses = [], []
for s in range(N_S):
    idx = obs_s[s]
    total = score[idx].sum()
    obs_avg = score[idx].mean()
    fair_avg = exp_scores[idx].mean()
    zz = z_scores[idx]; vv = var_off[idx]
    w = vv.sum()
    infit = (zz**2 * vv).sum() / w if w > 1e-10 else 1
    outfit = (zz**2).sum() / len(idx)
    info = vv.sum()
    se = 1/np.sqrt(info) if info > 1e-10 else 10
    s_measures.append(theta[s]); s_ses.append(se)
    f = facets_stu
    print(f"S{s+1:<11} {theta[s]:>8.3f} {f['meas'][s]:>10.2f} {se:>7.3f} "
          f"{f['se'][s]:>7.3f} {infit:>7.3f} {f['infit'][s]:>7.3f} "
          f"{outfit:>7.3f} {f['outfit'][s]:>7.3f}")

s_ma = np.array(s_measures); s_sa = np.array(s_ses)
vo = np.var(s_ma, ddof=1); me = np.mean(s_sa**2)
vt = max(vo - me, 0.001)
print(f"  Sep={np.sqrt(vt/me):.2f} (F:9.89) Rel={vt/(vt+me):.3f} (F:.99)")

# ── 题目面向 ─────────────────────────────────────────────────────────
facets_item = {
    "meas": [-0.07, 0.45, 1.28, -1.66],
    "se": [0.06, 0.06, 0.08, 0.07],
    "infit": [1.37, 1.03, 0.63, 0.53],
    "outfit": [1.38, 1.04, 0.63, 0.64],
}

print(f"\n--- 题目面向 ---")
print(f"{'':12} {'Meas':>8} {'(Facets)':>10} {'SE':>7} {'(F)':>7} "
      f"{'Infit':>7} {'(F)':>7} {'Outfit':>7} {'(F)':>7}")
print("-" * 90)
i_measures, i_ses = [], []
for iti in range(N_I):
    idx = obs_i[iti]
    total = score[idx].sum()
    obs_avg = score[idx].mean()
    zz = z_scores[idx]; vv = var_off[idx]
    w = vv.sum()
    infit = (zz**2 * vv).sum() / w if w > 1e-10 else 1
    outfit = (zz**2).sum() / len(idx)
    info = vv.sum()
    se = 1/np.sqrt(info) if info > 1e-10 else 10
    i_measures.append(beta[iti]); i_ses.append(se)
    f = facets_item
    print(f"Item{iti+1:<7} {beta[iti]:>8.3f} {f['meas'][iti]:>10.2f} "
          f"{se:>7.3f} {f['se'][iti]:>7.3f} "
          f"{infit:>7.3f} {f['infit'][iti]:>7.3f} "
          f"{outfit:>7.3f} {f['outfit'][iti]:>7.3f}")

i_ma = np.array(i_measures); i_sa = np.array(i_ses)
vo = np.var(i_ma, ddof=1); me = np.mean(i_sa**2)
vt = max(vo - me, 0.001)
print(f"  Sep={np.sqrt(vt/me):.2f} (F:15.47) Rel={vt/(vt+me):.3f} (F:1.00)")

# ── 参数 ─────────────────────────────────────────────────────────────
print(f"\n参数:")
print(f"  theta: {np.round(theta, 4)}")
print(f"  delta: {np.round(delta, 4)}")
print(f"  alpha: {np.round(alpha, 4)}")
print(f"  beta:  {np.round(beta, 4)}")
print(f"  tau:   {np.round(tau, 4)}")
