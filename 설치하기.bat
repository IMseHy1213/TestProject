@echo off
chcp 949 >nul
setlocal
cd /d "%~dp0"
title 토지·공장 매물 추천 - 설치
echo ==============================================================
echo   토지·공장 매물 추천 프로그램 설치  [윈도 10 · 11]
echo   처음 한 번만 하면 됩니다. 인터넷이 연결되어 있어야 합니다.
echo ==============================================================
echo.

rem ---------- 1. 파이썬 3.10 이상 찾기 ----------
echo [1/5] 파이썬 확인 중 ...
call :findpy
if defined PY goto havepy
echo   파이썬이 없거나 너무 옛날 버전입니다. 파이썬 3.12 를 설치합니다.
where winget >nul 2>nul
if errorlevel 1 goto nowinget
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
call :findpy
if defined PY goto havepy
echo.
echo   파이썬 설치가 끝났지만 이 창에서는 아직 보이지 않습니다.
echo   이 창을 닫고 설치하기.bat 를 한 번 더 실행해 주세요.
goto fail

:nowinget
echo.
echo   이 PC 에서는 자동 설치를 할 수 없습니다. 열리는 홈페이지에서 직접 설치해 주세요.
echo     1. Download Python 3.x 버튼을 눌러 받은 파일 실행
echo     2. 첫 화면 아래 "Add python.exe to PATH" 에 꼭 체크 후 Install Now
echo     3. 설치가 끝나면 이 창을 닫고 설치하기.bat 를 다시 실행
start "" "https://www.python.org/downloads/windows/"
goto fail

:havepy
echo   OK  %PY%

rem ---------- 2. 가상환경 .venv ----------
echo.
echo [2/5] 프로그램 전용 파이썬 환경 준비 중 ...
if not exist ".venv\Scripts\python.exe" goto makevenv
".venv\Scripts\python.exe" -c "import sys" >nul 2>nul
if not errorlevel 1 goto venvok
echo   다른 PC 에서 복사해 온 환경이라 이 PC 에 맞게 새로 만듭니다.
rmdir /s /q ".venv"
:makevenv
"%PY%" -m venv .venv
if errorlevel 1 goto venvfail
:venvok
echo   OK  .venv

rem ---------- 3. 필요한 패키지 ----------
echo.
echo [3/5] 필요한 패키지 설치 중 ... 처음에는 3~10분 걸립니다.
".venv\Scripts\python.exe" -m pip install --upgrade pip --disable-pip-version-check -q
".venv\Scripts\python.exe" -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 goto pipfail
echo   OK  패키지 설치 완료

rem ---------- 4. Git : 최신 코드 받기에 필요 ----------
echo.
echo [4/5] Git 확인 중 ... 최신 코드로 바꿀 때 필요합니다.
where git >nul 2>nul
if not errorlevel 1 goto havegit
where winget >nul 2>nul
if errorlevel 1 goto nogit
choice /c YN /m "  Git 이 없습니다. 지금 설치할까요"
if errorlevel 2 goto nogit
winget install -e --id Git.Git --accept-package-agreements --accept-source-agreements
goto gitdone
:nogit
echo   Git 없이도 프로그램은 쓸 수 있습니다. 최신 코드로 바꾸려면 나중에 https://git-scm.com 에서 설치하세요.
goto gitdone
:havegit
echo   OK  Git
:gitdone

rem ---------- 5. 바탕화면 아이콘 · 자동 실행 · 절전 ----------
echo.
echo [5/5] 바탕화면 아이콘 만들기
".venv\Scripts\python.exe" main.py windows-setup
echo.
echo ==============================================================
echo   설치 완료!  바탕화면의 "매물 추천 켜기" 를 더블클릭하세요.
echo ==============================================================
pause
exit /b 0

:venvfail
echo.
echo   [오류] 프로그램 전용 환경을 만들지 못했습니다. 위의 메시지를 캡처해서 보내 주세요.
goto fail
:pipfail
echo.
echo   [오류] 패키지 설치에 실패했습니다. 인터넷 연결을 확인하고 설치하기.bat 를 다시 실행하세요.
echo   계속 안 되면 위의 메시지를 캡처해서 보내 주세요.
goto fail
:fail
echo.
pause
exit /b 1

rem ---------- 파이썬 3.10 이상 찾기: 찾으면 PY 에 python.exe 경로 ----------
:findpy
set "PY="
for /f "delims=" %%P in ('py -3 -c "import sys;sys.version_info>=(3,10) and print(sys.executable)" 2^>nul') do set "PY=%%P"
if defined PY goto :eof
for /f "delims=" %%P in ('python -c "import sys;sys.version_info>=(3,10) and print(sys.executable)" 2^>nul') do set "PY=%%P"
if defined PY goto :eof
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*" "%ProgramFiles%\Python3*") do if exist "%%~D\python.exe" call :trypy "%%~D\python.exe"
goto :eof

:trypy
if defined PY goto :eof
for /f "delims=" %%P in ('call "%~1" -c "import sys;sys.version_info>=(3,10) and print(sys.executable)" 2^>nul') do set "PY=%%P"
goto :eof
