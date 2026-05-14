@echo off
chcp 65001 > nul
echo.
echo ============================================
echo   LoRA 학습 시작
echo   사진을 data\input_faces\ 폴더에 넣으세요
echo   (권장: 15~20장, 다양한 각도)
echo ============================================
echo.
call venv\Scripts\activate.bat
python src\lora_trainer.py --input_dir data\input_faces --output_dir models\lora --steps 1000
pause
