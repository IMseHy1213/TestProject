@echo off
chcp 949 >nul
cd /d "%~dp0"
title 토지·공장 매물 추천 - 이 창을 닫으면 프로그램이 꺼집니다
if exist ".venv\Scripts\python.exe" goto run
echo 아직 설치가 되지 않았습니다. 이 폴더의 설치하기.bat 를 먼저 실행하세요.
pause
exit /b 1

:run
echo 토지·공장 매물 추천 프로그램을 켭니다. 잠시 뒤 브라우저가 열립니다.
echo 이 검은 창은 닫지 말고 작게 내려 두세요. 창을 닫으면 프로그램이 꺼집니다.
echo.
".venv\Scripts\python.exe" main.py mobile %*
if errorlevel 3 if not errorlevel 4 goto already
echo.
echo 프로그램이 꺼졌습니다.
pause
exit /b 0

:already
echo.
echo 이미 켜져 있어서 브라우저만 열었습니다. 이 창은 5초 뒤 닫힙니다.
timeout /t 5 >nul
exit /b 0
