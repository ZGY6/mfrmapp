# MFRMSight 更新日志 (Changelog)

## v1.0.9 (2026-06-09) — 当前版本

**新增:**
- (待补充)

**修复:**
- (待补充)

---

## v1.0.3 (2026-06-08) — Gradio 6 稳定版

**新增:**
- 4 步渐进式 Web 界面: 上传选面 → 分析结果 → 交互分析 → 报告+图表
- 偏差交互分析 (`bias_interaction`): 任意两面交互对偏差量计算，|z|≥2 标记显著
- 一键生成 Word 专业报告 (`generate_word_report`): 10 章结构 + APA 三线表 + 嵌入图表
- 3 张 mat plot lib 统计图: 垂直标尺图 / 拟合分布图 / 等级概率曲线+ICC
- AIC/BIC 模型拟合指标 (`aic_bic`)
- 动态面名称提取 (`extract_dimensions`): 从文件 Labels 段读取 + 英文自动翻译
- 维度选择过滤 (`filter_data`): 按勾选的标准/题目过滤数据
- 5 种评分者偏差自动诊断 (`_diagnose_bias`): 宽松/严格/集中趋势/随机/晕轮
- 类别功能诊断 (`_diagnose_categories`): 阈值无序检测 + 等级使用分析
- 异常反应标记 (`_anomalous_responses`): |StRes|≥3 自动标记
- 排名对比表: 原始分排名 vs MFRM 校正排名

**修复:**
- BUG: Step4 图表不显示 — `gr.Image` 需要 numpy 数组，不能用文件路径/base64
- BUG: Gradio 6 兼容 — 移除 `.then()` 链、`visible` 切换、废弃参数
- BUG: Gradio 6 静默崩溃 — 输出数量必须严格等于 outputs 数
- BUG: `parse_facets_txt` 被意外覆盖，从 git 恢复
- 版本号统一: `app_gradio.py` / `engine.py` / `pyproject.toml` / `cli.py` → v1.0.3

---

## v1.0.0 (2026-06-08) — 交互式分析 + 专业报告

**新增:**
- 4 步渐进式 Gradio Web 界面
- `extract_dimensions`: 动态面名称提取 + 中英文翻译
- `filter_data`: 维度选择过滤 + ID 重新映射
- `bias_interaction`: 任意两面交互对偏差分析
- `generate_report`: 10 章节中文 Markdown 报告
- `generate_word_report`: 10 章节 Word 报告 (python-docx + APA 三线表)
- `aic_bic`: AIC/BIC 模型比较指标
- `chart_ruler_map` / `chart_category_curves` / `chart_fit_distribution`: 3 张统计图
- `_diagnose_bias`: 5 种评分者偏差自动检测
- `_diagnose_categories`: 类别功能诊断
- `_anomalous_responses`: 异常响应标记

---

## v0.9.0 (2026-06-08) — Facets 输出文件解析

**新增:**
- `parse_facets_out()`: 解析 Facets .out.txt 输出文件，提取 Table 5/7/8/4
- `generate_report()`: 从解析结果生成 10 章节中文 Markdown 报告
- Facets 输入格式兼容: `;` 注释符、tab-range token (`1\t-\t2`)、`Models=` 复数写法
- 元素编号范围语法支持: `1-4` (Labels 段)

**验证:**
- 2026Raterbias10.out.txt 与 Facets 参考值完全匹配: Students Sep=9.89, Student4 meas=0.61, VarExp=88.10%

---

## v0.8.0 (2026-06-08) — 全部 Bug 修复 + 48 测试覆盖

**新增:**
- 包版测试: `mfrmapp/tests/test_engine.py` (16 用例)
- 单文件版测试: `tests/test_engine.py` (32 用例)
- 总计 48 个 pytest 用例全通过
- 稀疏数据检测与警告 (每个等级 < 8 观测发出 UserWarning)
- Excel 多行标题自动合并 (`_merge_header_rows`)
- README 添加 Gradio Web 使用说明 + 在线地址
- `.gitignore` 移除测试数据排除
- `pyproject.toml` 添加 `web`/`test` 可选依赖
- 评分者图标更换为自定义图片

**修复 (BUG-009~018):**

| Bug | 概述 |
|-----|------|
| BUG-009 | 标签解析器重写 — 区分 facet 声明行与元素标签行 |
| BUG-010 | JMLE 收敛 — ExpMean 从 8.80 → 10.62 (差距从 17% → 0) |
| BUG-011 | Noncentered=1 — 正确解析约束面，不再对所有面强制居中 |
| BUG-012 | PROX 初始化 — Newton-Raphson 反演替代粗糙 logit 近似 |
| BUG-013 | Sequential JMLE — 循环内每元素后重算概率矩阵 |
| BUG-014 | Items 标签正确 (连带修复) |
| BUG-015 | 包版测试补充 |
| BUG-016 | 两引擎参数差异 (JMLE 方法局限，已文档化) |
| BUG-017 | README 缺少 Web 使用说明 |
| BUG-018 | `.gitignore`/依赖/`render.yaml` 修复 |

---

## v0.7.0 (2026-06-08) — Render.com 部署 + 端口清理

**新增:**
- Gradio 端口冲突自动检测与清理 (Windows: `netstat -ano` → `taskkill`)
- Excel 列名映射智能推断 (`_guess_facet_from_name` 重写)
- `mfrm_web.py` 纯 HTTP Web 版 (手机可直接打开)
- Render.com Blueprint 部署配置 (`render.yaml`)

**修复:**
| Bug | 概述 |
|-----|------|
| BUG-001 | Gradio 端口冲突 — 启动前自动 kill 旧进程 + 换端口重试 |
| BUG-002 | Excel 列名推断 — 双模式 (强信号关键词 + 弱信号连字符结构) |
| BUG-003 | Measure 系统性差异 — 根因由 BUG-010~012 消除 |

---

## v0.6.0~v0.6.2 (2026-06-07~08) — 核心算法修复

**新增:**
- Minifac .txt 完整适配: `parse_minifac_txt()` 支持 Header/Labels/Data 状态机解析
- Model 字符串解析: `parse_model_string()` 支持 R/D/P/M 类型识别
- Rating Scale 块解析: 命名量表 + 类别标签
- Data 段索引展开: 范围 token (`2_4`)、R 复制前缀 (`R3`)
- Gradio Web 界面 + 交互式面向询问

**修复:**
- BUG-009: 标签解析器 — 用 `*` 分隔符区分 facet 声明和元素标签
- BUG-012: PROX 初始化 — 两阶段 N-R 反演 (核心)
- BUG-011: Noncentered — 解析 `Noncentered=` 头 + `_center_prox()` 居中逻辑

---

## v0.1.0~v0.5.0 (2026-06-07) — 项目启动与基础建设

**新增:**
- 基于 Andrich Rating Scale Model (1978) 的 Fisher-scoring JMLE 估计器
- PROX 初始化 → 两阶段 JMLE 迭代
- Facets .txt 和 Excel .xlsx 数据解析
- 命令行 + Gradio Web + 纯 HTTP Web 三种入口
- 报告输出: 控制台打印 + Excel + Word

**踩坑记录 (v0.1~v0.3):**
1. 参数发散 — theta 炸到 ±150 万 logit → 添加 ridge 正则化 + 截断
2. U 因子过度放大 — 3x 拉伸 → 温和校正 U=1+0.5×(ratio-1)
3. tau 阈值发散 — 梯度符号弄反 → 修正 ∂LL/∂τ_k = exp_ge - obs_ge
4. 稀疏数据 — 64 条配 20 阈值无法收敛 → 确认 < 8 观测/等级为硬限制

---

## 部署历史

| 版本 | 平台 | 状态 |
|------|------|------|
| v0.2.0 | GitHub | 初始发布 |
| v0.7.0 | Render.com | 首次部署成功 |
| v0.8.0 | Render.com | Bug 全面修复版 |
| v1.0.3 | Render.com | 4 步交互 + 报告 + 图表 (当前) |
| v1.0.7 | 本机 | 正在迭代 |
