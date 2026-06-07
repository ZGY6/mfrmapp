"""MFRMSight CLI — 命令行入口"""
import argparse, sys
from pathlib import Path
from .engine import parse_facets_txt, parse_excel, MFRMEngine


def main():
    ap = argparse.ArgumentParser(
        description="MFRMSight v0.2 — 多面Rasch模型分析工具",
        epilog="示例: mfrmapp data.txt -o result.xlsx")
    ap.add_argument("input", help="输入文件 (.txt 或 .xlsx)")
    ap.add_argument("-o", "--output", help="输出文件 (.xlsx / .docx)")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()

    p = Path(a.input)
    try:
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


if __name__ == "__main__":
    main()
