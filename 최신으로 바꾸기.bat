@echo off
chcp 949 >nul
cd /d "%~dp0"
title 토지·공장 매물 추천 - 최신 코드 받기
if exist ".venv\Scripts\python.exe" goto run
echo 아직 설치가 되지 않았습니다. 이 폴더의 설치하기.bat 를 먼저 실행하세요.
pause
exit /b 1

:run
".venv\Scripts\python.exe" main.py update
if errorlevel 1 goto end
echo.
echo 필요한 패키지 확인 중 ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt --disable-pip-version-check -q
echo.
echo 다 됐습니다. 프로그램이 켜져 있으면 그 검은 창을 닫고 "매물 추천 켜기" 로 다시 켜세요.
:end
echo.
pause
