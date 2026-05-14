"""
pipeline.py - 전체 가상 피팅 파이프라인 통합 모듈

역할: 마스킹 → LoRA 인페인팅 → 결과 저장까지 전 과정을 하나의 흐름으로 연결합니다.

파이프라인 흐름:
[인물 사진] → [YOLO/SAM 마스킹] → [LLM 프롬프트 변환] → [SD Inpainting + LoRA] → [결과 저장]
"""

import os
import time
from pathlib import Path
from PIL import Image
from typing import Optional, Literal
from datetime import datetime

from src.masking import FashionMasker
from src.inpainting import FashionInpainter
from src.prompt_generator import PromptGenerator


class VirtualFittingPipeline:
    """
    AI 가상 피팅 통합 파이프라인

    사용 예시:
        pipeline = VirtualFittingPipeline()
        pipeline.setup(lora_path="models/lora/my_person.safetensors")

        result = pipeline.run(
            person_image_path="data/test_images/me.jpg",
            style_description="시원한 여름 린넨 셔츠",
            target_region="upper_body"
        )
    """

    def __init__(self, output_dir: str = "data/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.masker = FashionMasker()
        self.inpainter = FashionInpainter()
        self.prompt_gen = PromptGenerator()

        self._pipeline_ready = False

    def setup(
        self,
        lora_path: Optional[str] = None,
        lora_scale: float = 0.8,
    ):
        """
        파이프라인 초기화 - 모델 로드

        Args:
            lora_path: LoRA 파일 경로 (없으면 일반 Inpainting)
            lora_scale: LoRA 강도 (0.0~1.0)
        """
        print("=" * 50)
        print("  AI 가상 피팅 파이프라인 초기화")
        print("=" * 50)

        # Inpainting 파이프라인 로드
        self.inpainter.load_pipeline()

        # LoRA 로드 (있으면)
        if lora_path and os.path.exists(lora_path):
            self.inpainter.load_lora(lora_path, lora_scale)
        elif lora_path:
            print(f"[Pipeline] LoRA 파일 없음: {lora_path}")
            print("  → LoRA 없이 실행합니다 (얼굴 일관성 보장 안됨)")

        self._pipeline_ready = True
        print("파이프라인 준비 완료!")

    def run(
        self,
        person_image_path: str,
        style_description: str,
        target_region: Literal["upper_body", "lower_body", "full_body"] = "upper_body",
        style_preset: Optional[str] = None,
        custom_mask_path: Optional[str] = None,
        seed: Optional[int] = None,
        num_steps: int = 30,
        save_intermediates: bool = True,
    ) -> dict:
        """
        가상 피팅 실행

        Args:
            person_image_path: 인물 사진 경로
            style_description: 원하는 스타일 설명 (한국어 가능)
                예: "시원한 여름 린넨 셔츠"
            target_region: 교체할 의상 부위
            style_preset: 스타일 프리셋 ("luxury", "casual", "streetwear", "minimal")
            custom_mask_path: 수동 마스크 이미지 경로 (자동 마스킹 대신 사용)
            seed: 재현을 위한 시드값
            num_steps: 추론 스텝 수
            save_intermediates: 중간 결과물 저장 여부

        Returns:
            dict: {
                "result_path": 최종 이미지 경로,
                "mask_path": 마스크 이미지 경로,
                "prompt": 사용된 SD 프롬프트,
                "elapsed_time": 소요 시간(초)
            }
        """
        if not self._pipeline_ready:
            raise RuntimeError("setup()을 먼저 호출하세요.")

        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n{'='*50}")
        print(f"  가상 피팅 시작")
        print(f"  스타일: {style_description}")
        print(f"  부위: {target_region}")
        print(f"{'='*50}\n")

        # Step 1: 이미지 로드
        print("[Step 1/4] 이미지 로드...")
        person_image = Image.open(person_image_path).convert("RGB")
        print(f"  원본 크기: {person_image.size}")

        # Step 2: 마스크 생성
        print("\n[Step 2/4] 의상 영역 마스킹...")
        if custom_mask_path:
            print(f"  수동 마스크 사용: {custom_mask_path}")
            mask_image = Image.open(custom_mask_path).convert("L")
        else:
            mask_image = self.masker.get_mask(person_image, target=target_region)
            if mask_image is None:
                print("  ⚠️  자동 마스킹 실패. 전신 영역으로 대체합니다.")
                # 폴백: 이미지 중앙 영역을 마스크로 사용
                import numpy as np
                w, h = person_image.size
                mask_arr = np.zeros((h, w), dtype=np.uint8)
                mask_arr[h//4:3*h//4, w//6:5*w//6] = 255
                mask_image = Image.fromarray(mask_arr)

        # 중간 결과 저장
        if save_intermediates:
            mask_path = str(self.output_dir / f"mask_{timestamp}.png")
            mask_image.save(mask_path)

            preview = self.masker.apply_mask_preview(person_image, mask_image)
            preview_path = str(self.output_dir / f"mask_preview_{timestamp}.png")
            preview.save(preview_path)
            print(f"  마스크 저장: {mask_path}")

        # Step 3: 프롬프트 생성
        print("\n[Step 3/4] 스타일 프롬프트 생성...")
        sd_prompt = self.prompt_gen.generate(style_description, style_preset)
        negative_prompt = self.prompt_gen.generate_negative_prompt()
        print(f"  프롬프트: {sd_prompt[:80]}...")

        # Step 4: Inpainting 실행
        print("\n[Step 4/4] 가상 피팅 생성 중...")
        result_image = self.inpainter.try_on(
            person_image=person_image,
            mask_image=mask_image,
            clothing_prompt=sd_prompt,
            negative_prompt=negative_prompt,
            num_steps=num_steps,
            seed=seed,
        )

        # 결과 저장
        result_path = str(self.output_dir / f"result_{timestamp}.png")
        result_image.save(result_path)

        elapsed = time.time() - start_time
        print(f"\n✅ 완료! 소요 시간: {elapsed:.1f}초")
        print(f"  결과 저장: {result_path}")

        return {
            "result_path": result_path,
            "mask_path": mask_path if save_intermediates else None,
            "prompt": sd_prompt,
            "elapsed_time": elapsed,
        }

    def run_lookbook(
        self,
        person_image_path: str,
        style_descriptions: list[str],
        target_region: Literal["upper_body", "lower_body", "full_body"] = "upper_body",
        style_preset: Optional[str] = None,
        reuse_mask: bool = True,
    ) -> list[dict]:
        """
        여러 스타일로 룩북 일괄 생성

        Args:
            style_descriptions: 여러 스타일 설명 리스트
            reuse_mask: 첫 번째 마스크를 재사용 (속도 향상)

        Returns:
            각 피팅 결과 dict 리스트
        """
        print(f"\n룩북 생성 시작: {len(style_descriptions)}개 스타일")
        results = []

        cached_mask = None
        for i, style in enumerate(style_descriptions):
            print(f"\n[{i+1}/{len(style_descriptions)}] {style}")

            # 첫 번째 마스크 재사용으로 속도 향상
            if reuse_mask and i > 0 and cached_mask:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    cached_mask.save(f.name)
                    result = self.run(
                        person_image_path=person_image_path,
                        style_description=style,
                        target_region=target_region,
                        style_preset=style_preset,
                        custom_mask_path=f.name,
                    )
            else:
                result = self.run(
                    person_image_path=person_image_path,
                    style_description=style,
                    target_region=target_region,
                    style_preset=style_preset,
                )

            # 첫 번째 마스크 캐시
            if i == 0 and result.get("mask_path"):
                cached_mask = Image.open(result["mask_path"])

            results.append(result)

        print(f"\n\n룩북 생성 완료! {len(results)}장")
        return results


# ─────────────────────────────────────────────
# 직접 실행 테스트용
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python -m src.pipeline <인물사진 경로> [스타일 설명]")
        print("예시: python -m src.pipeline data/test_images/me.jpg '시원한 여름 린넨 셔츠'")
        sys.exit(1)

    image_path = sys.argv[1]
    style = sys.argv[2] if len(sys.argv) > 2 else "casual summer fashion, light colors"

    pipeline = VirtualFittingPipeline()
    pipeline.setup(lora_path="models/lora/my_person.safetensors")

    result = pipeline.run(
        person_image_path=image_path,
        style_description=style,
        target_region="upper_body",
        seed=42,
    )

    print(f"\n결과: {result['result_path']}")
