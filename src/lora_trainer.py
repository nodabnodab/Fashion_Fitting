"""
lora_trainer.py - LoRA 학습 유틸리티

주의: LoRA 학습은 kohya-ss/sd-scripts 외부 도구를 활용합니다.
이 파일은 데이터 전처리와 학습 설정을 도와주는 헬퍼입니다.

LoRA 학습 전체 과정:
1. 사진 수집 (15~20장, 다양한 각도)
2. 이 스크립트로 이미지 전처리 (리사이즈, 캡셔닝)
3. kohya-ss로 학습 실행
"""

import os
import argparse
from pathlib import Path
from PIL import Image


def preprocess_training_images(
    input_dir: str,
    output_dir: str,
    target_size: int = 512,
    trigger_word: str = "ohwx person",
):
    """
    LoRA 학습용 이미지 전처리

    1. 이미지를 정사각형으로 리사이즈 (SD 기본 해상도)
    2. 각 이미지에 캡션 파일(.txt) 생성

    Args:
        input_dir: 원본 사진 폴더 (15~20장의 인물 사진)
        output_dir: 전처리된 이미지 저장 폴더
        target_size: 학습 이미지 크기 (512 또는 768)
        trigger_word: LoRA 트리거 워드 (생성 시 이 단어로 호출)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    supported_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in supported_exts]

    if not image_files:
        print(f"[LoRA Trainer] 이미지를 찾을 수 없습니다: {input_dir}")
        return 0

    print(f"[LoRA Trainer] {len(image_files)}개 이미지 전처리 시작...")

    processed = 0
    for img_file in image_files:
        try:
            img = Image.open(img_file).convert("RGB")

            # 정사각형 크롭 후 리사이즈 (중앙 크롭)
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            img_cropped = img.crop((left, top, left + min_dim, top + min_dim))
            img_resized = img_cropped.resize((target_size, target_size), Image.LANCZOS)

            # 이미지 저장
            out_img_path = output_path / f"{img_file.stem}_proc.png"
            img_resized.save(out_img_path, "PNG")

            # 캡션 파일 생성 (SD 학습에 필수)
            caption = f"{trigger_word}, a photo of a person"
            out_txt_path = output_path / f"{img_file.stem}_proc.txt"
            out_txt_path.write_text(caption, encoding="utf-8")

            processed += 1
            print(f"  [{processed}/{len(image_files)}] {img_file.name} → {out_img_path.name}")

        except Exception as e:
            print(f"  ⚠️  {img_file.name} 처리 실패: {e}")

    print(f"\n[LoRA Trainer] 전처리 완료: {processed}개 이미지")
    print(f"  출력 폴더: {output_path}")
    print(f"  트리거 워드: {trigger_word}")
    return processed


def print_kohya_guide(output_dir: str, trigger_word: str = "ohwx person"):
    """kohya-ss를 이용한 LoRA 학습 가이드 출력"""
    print("\n" + "=" * 60)
    print("  kohya-ss LoRA 학습 가이드")
    print("=" * 60)
    print("""
📌 다음 단계로 LoRA를 학습하세요:

1. kohya-ss 설치 (최초 1회):
   git clone https://github.com/bmaltais/kohya_ss.git
   cd kohya_ss
   setup.bat

2. kohya-ss GUI 실행:
   gui.bat

3. GUI 설정 (LoRA 탭):
   - Training data dir: {output_dir}
   - Output dir: models/lora/
   - Model: Stable Diffusion 1.5 체크포인트
   - Network type: LoRA
   - Network rank (Lora dim): 32
   - Network alpha: 16
   - Learning rate: 1e-4
   - Max train steps: 1000~1500

4. 학습 완료 후:
   models/lora/ 에 .safetensors 파일 생성됨

5. 앱에서 LoRA 경로 입력:
   models/lora/last.safetensors

💡 트리거 워드: "{trigger_word}"
   이 단어를 프롬프트에 포함하면 학습한 인물이 등장합니다.
""".format(output_dir=output_dir, trigger_word=trigger_word))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA 학습 전처리")
    parser.add_argument("--input_dir", default="data/input_faces", help="원본 사진 폴더")
    parser.add_argument("--output_dir", default="data/training_processed", help="전처리 출력 폴더")
    parser.add_argument("--steps", type=int, default=1000, help="학습 스텝 수 (참고용)")
    parser.add_argument("--trigger", default="ohwx person", help="LoRA 트리거 워드")
    args = parser.parse_args()

    count = preprocess_training_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        trigger_word=args.trigger,
    )

    if count > 0:
        print_kohya_guide(args.output_dir, args.trigger)
