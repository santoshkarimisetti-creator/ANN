@echo off
title ANN Deep Learning Suite Server
echo Starting ANN Deep Learning Web Application...
if exist "C:\Users\jagan\AppData\Local\Programs\Python\Python310\python.exe" (
    "C:\Users\jagan\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run app.py
) else (
    python -m streamlit run app.py
)
pause
