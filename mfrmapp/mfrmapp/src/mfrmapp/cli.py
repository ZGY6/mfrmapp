"""MFRMSight CLI — 命令行入口"""
import argparse, sys
from pathlib import Path
from .engine import parse_facets_txt, parse_excel, MFRMEngine, parse_facets_out


def main():
    ap = argparse.ArgumentParser(
        description="MFRMSight v1.0.12 — 多面Rasch模型分析工具",
        epilog="示例: mfrmapp data.txt -o result.xlsx")
    ap.add_argument("input", help="输入文件 (.txt / .xlsx / .out.txt)")
    ap.add_argument("-o", "--output", help="输出文件 (.xlsx / .docx)")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()

    p = Path(a.input)
    try:
        # .out.txt: Facets 输出解读模式
        if p.suffix == ".txt" and ".out" in p.name.lower():
            r = parse_facets_out(str(p))
            _print_facets_report(r)
            return

        if p.suffix == ".txt":
            d = parse_facets_txt(str(p))
        elif p.suffix in (".xlsx", ".xls"):
            d = parse_excel(str(p))
        else:
            print(f"错误: 不支持的文件格式 {p.suffix}", file=sys.stderr)
            sys.exit(1)

        eng = MFRMEngine(d).fit()
        if not a.quiet: eng.print()

        if a.output:
            o = Path(a.output)
            if o.suffix == ".xlsx": eng.to_excel(str(o))
            elif o.suffix == ".docx": eng.to_word(str(o))
            else: eng.to_excel(str(o) + ".xlsx")

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback; traceback.print_exc()
        sys.exit(1)


def _print_facets_report(r: dict):
    """打印 Facets .out.txt 解读报告。"""
    s = r["summary"]
    print(f"\n{'='*72}")
    print(f"Facets 输出文件解读报告")
    print(f"{'='*72}")
    print(f"反应数: {s.get('N', '?')}")
    print(f"ObsMean={s.get('obs_mean', '?')} ExpMean={s.get('exp_mean', '?')}")
    print(f"方差解释: {s.get('var_exp', '?')}%")
    print(f"Chi-squared={s.get('chi_sq', '?')} p={s.get('chi_prob', '?')}")

    print(f"\n{'─'*72}")
    print(f"{'面向':<12} {'Separation':>10} {'Reliability':>12}")
    print("-" * 40)
    for fn, fd in r["facets"].items():
        print(f"{fn:<12} {fd['separation']:>10.2f} {fd['reliability']:>12.3f}")

    cat = r["categories"]
    if cat["rows"]:
        print(f"\n{'─'*72}")
        print(f"等级类别: {len(cat['rows'])} 档, 阈值有序: {'是' if cat['tau_ordered'] else '否'}")
        if not cat["tau_ordered"]:
            disordered = [(i, cat["tau"][i], cat["tau"][i+1])
                         for i in range(len(cat["tau"])-1) if cat["tau"][i] > cat["tau"][i+1]]
            for i, t1, t2 in disordered:
                print(f"  [!] tau[{i+1}]={t1:.2f} > tau[{i+2}]={t2:.2f}")

    anom = r["anomalous"]
    if anom:
        print(f"\n[!!] {len(anom)} 条异常反应 (|StRes| >= 3)")
    else:
        print(f"\n[OK] 未检出异常反应")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
