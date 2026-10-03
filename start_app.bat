@echo off
title BasQ VQE Studio Launcher
echo ====================================================
echo Starting BasQ Quantum Molecular VQE Studio...
echo Opening in your browser at http://localhost:8501
echo ====================================================
python -m streamlit run app.py --server.port 8501
pause
