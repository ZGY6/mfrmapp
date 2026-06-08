"""MFRMSight Web 启动器"""
import os, sys


def main():
    import streamlit.web.cli as stcli
    from pathlib import Path

    app_path = Path(__file__).parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
