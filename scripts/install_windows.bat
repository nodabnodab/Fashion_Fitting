@echo off
chcp 65001 > nul
echo.
echo ============================================
echo   AI 가상 피팅 프로젝트 - 환경 설치 스크립트
echo ============================================
echo.

REM Python 버전 확인
python --version 2>nul
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    echo https://www.python.org/downloads/release/python-31011/ 에서 3.10.x 버전을 설치하세요.
    pause
    exit /b 1
)

echo [1/5] 가상환경 생성 중...
python -m venv venv
if errorlevel 1 (
    echo [오류] 가상환경 생성 실패
    pause
    exit /b 1
)

echo [2/5] 가상환경 활성화...
call venv\Scripts\activate.bat

echo [3/5] pip 업그레이드...
python -m pip install --upgrade pip

echo [4/5] 핵심 패키지 설치 중... (시간이 걸립니다)
echo.

REM PyTorch (CUDA 12.1 버전 - NVIDIA GPU용)
REM GPU가 없다면 아래 줄을 주석처리하고 그 다음 줄의 주석을 해제하세요
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
REM pip install torch torchvision torchaudio  ^(CPU 전용 - GPU 없을 때)^

REM 핵심 AI 패키지
pip install diffusers==0.27.2
pip install transformers==4.40.0
pip install accelerate==0.30.0
pip install huggingface_hub

REM 이미지 처리
pip install Pillow opencv-python-headless
pip install numpy scipy

REM YOLO + Segment Anything
pip install ultralytics
pip install segment-anything-py

REM IP-Adapter 관련
pip install einops

REM Web UI
pip install gradio==4.36.0

REM LLM API (Gemini)
pip install google-generativeai

REM 기타 유틸리티
pip install python-dotenv
pip install tqdm
pip install matplotlib
pip install jupyter notebook ipykernel

echo.
echo [5/5] 설치 완료 확인 중...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA 사용 가능: {torch.cuda.is_available()}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"

echo.
echo ============================================
echo   설치 완료!
echo   다음 명령으로 앱을 실행하세요:
echo   scripts\run_app.bat
echo ============================================
pause
