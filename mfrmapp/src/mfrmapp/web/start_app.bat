@echo off
echo ========================================
echo   MFRMSight — 多面Rasch模型分析工具
echo ========================================
echo.
echo 正在启动...
echo.
echo 在浏览器打开: http://localhost:8501
echo 手机访问: http://YOUR_IP:8501
echo.
echo 按 Ctrl+C 停止
echo ========================================
echo.
cd /d "%~dp0..\.."
streamlit run src/mfrmapp/web/app.py --server.headless true
