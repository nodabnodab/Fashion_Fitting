"""
inpainting.py - Stable Diffusion Inpainting 모듈

역할: 마스크 영역(옷)을 새로운 의상으로 교체합니다.
LoRA 가중치를 로드하여 인물의 얼굴/체형을 유지합니다.
"""

import os
import torch
from PIL import Image
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class FashionInpainter:
    """
    SD Inpainting + LoRA 기반 가상 피팅 클래스

    사용 예시:
        inpainter = FashionInpainter()
        inpainter.load_pipeline()
        result = inpainter.try_on(
            person_image=person_img,
            mask_image=mask_img,
            clothing_prompt="a white linen shirt, summer fashion",
            lora_path="models/lora/my_person.safetensors"
        )
    """

    # 권장 기본 모델 (HuggingFace에서 자동 다운로드)
    DEFAULT_MODEL = "runwayml/stable-diffusion-inpainting"
    # SDXL 사용 시:
    # DEFAULT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_path = model_path or self.DEFAULT_MODEL
        self.pipe = None
        self._lora_loaded = None

        print(f"[FashionInpainter] 디바이스: {self.device}")
        if self.device == "cpu":
            print("  ⚠️  GPU 없음: 이미지 생성에 수분 이상 소요될 수 있습니다.")

    def load_pipeline(self, enable_xformers: bool = True):
        """
        Inpainting 파이프라인 로드

        Args:
            enable_xformers: 메모리 최적화 (VRAM 부족 시 효과적)
        """
        from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel

        print(f"[FashionInpainter] 모델 로딩: {self.model_path}")
        print("  (처음 실행 시 HuggingFace에서 다운로드됩니다. 수GB 용량)")

        # ControlNet Canny 로드 (약 1.5GB)
        print("[FashionInpainter] ControlNet Canny 모델 로딩 중...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            self.model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,  # 포트폴리오용으로 비활성화
        )
        self.pipe = self.pipe.to(self.device)

        # 메모리 최적화
        if enable_xformers and self.device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("[FashionInpainter] xformers 메모리 최적화 활성화")
            except Exception:
                print("[FashionInpainter] xformers 없음 (선택사항)")

        if self.device == "cuda":
            self.pipe.enable_attention_slicing()

        print("[FashionInpainter] 파이프라인 로드 완료!")

    def load_lora(self, lora_path: str, lora_scale: float = 0.8):
        """
        LoRA 가중치 로드 - 인물의 얼굴/체형 고유성 유지

        Args:
            lora_path: LoRA .safetensors 파일 경로
            lora_scale: LoRA 강도 (0.0~1.0, 높을수록 LoRA 특성 강함)
        """
        if self.pipe is None:
            raise RuntimeError("load_pipeline()을 먼저 호출하세요.")

        if not os.path.exists(lora_path):
            print(f"[FashionInpainter] LoRA 파일 없음: {lora_path}")
            print("  → scripts/train_lora.bat 으로 먼저 학습하세요")
            return

        print(f"[FashionInpainter] LoRA 로딩: {lora_path} (scale={lora_scale})")
        self.pipe.load_lora_weights(lora_path)
        self.pipe.fuse_lora(lora_scale=lora_scale)
        self._lora_loaded = lora_path
        print("[FashionInpainter] LoRA 로드 완료!")

    def try_on(
        self,
        person_image: Image.Image,
        mask_image: Image.Image,
        clothing_prompt: str,
        negative_prompt: str = "",
        num_steps: int = 30,
        guidance_scale: float = 7.5,
        strength: float = 0.85,
        controlnet_scale: float = 0.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        가상 피팅 실행 - 마스크 영역에 새 옷을 입힙니다.

        Args:
            person_image: 인물 원본 이미지 (PIL RGB)
            mask_image: 의상 영역 마스크 (PIL L, 흰=교체영역)
            clothing_prompt: 입힐 옷 설명 (영문 권장)
                예: "a white linen shirt, high fashion, studio lighting"
            negative_prompt: 제외할 요소
                예: "blurry, distorted, low quality"
            num_steps: 추론 스텝 수 (많을수록 품질 좋지만 느림)
            guidance_scale: 프롬프트 준수도 (7~12 권장)
            seed: 재현 가능한 결과를 위한 시드

        Returns:
            PIL Image (완성된 피팅 이미지)
        """
        if self.pipe is None:
            raise RuntimeError("load_pipeline()을 먼저 호출하세요.")

        # 표준 크기로 리사이즈 (SD 1.5 기준 512x512)
        target_size = (512, 512)
        person_resized = person_image.resize(target_size).convert("RGB")
        mask_resized = mask_image.resize(target_size).convert("L")

        # ControlNet Canny 이미지 생성 (원본 옷의 주름/형태 보존용)
        import cv2
        import numpy as np
        person_np = np.array(person_resized)
        
        # Canny Edge 추출
        canny_np = cv2.Canny(person_np, 100, 200)
        
        # 마스크 밖의 엣지는 지움 (배경 노이즈가 옷으로 변형되는 것 방지)
        mask_np = np.array(mask_resized)
        canny_np = canny_np * (mask_np > 128)
        
        # 3채널 이미지로 변환
        canny_image = Image.fromarray(cv2.cvtColor(canny_np, cv2.COLOR_GRAY2RGB))

        # 기본 negative prompt
        default_negative = (
            "blurry, distorted face, extra limbs, deformed, "
            "low quality, bad anatomy, watermark"
        )
        full_negative = f"{default_negative}, {negative_prompt}" if negative_prompt else default_negative

        # LoRA가 로드된 경우 인물 트리거 워드 추가 (학습 시 설정한 키워드)
        lora_trigger = ""
        if self._lora_loaded:
            lora_trigger = "ohwx person, "  # LoRA 학습 시 기본 트리거 워드

        full_prompt = f"{lora_trigger}{clothing_prompt}, high quality, detailed fabric texture"

        # 재현 가능한 생성
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        print(f"[FashionInpainter] 이미지 생성 중...")
        print(f"  프롬프트: {full_prompt}")
        print(f"  스텝: {num_steps}, 가이던스: {guidance_scale}")

        with torch.autocast(self.device) if self.device == "cuda" else torch.no_grad():
            result = self.pipe(
                prompt=full_prompt,
                negative_prompt=full_negative,
                image=person_resized,
                mask_image=mask_resized,
                control_image=canny_image,
                controlnet_conditioning_scale=controlnet_scale,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=generator,
            )

        output_image = result.images[0]

        # 원본 크기로 복원
        if person_image.size != target_size:
            output_image = output_image.resize(person_image.size, Image.LANCZOS)

        # 100% 완벽한 원본 보존을 위한 수동 마스크 합성 (Composite)
        # 생성된 이미지는 마스크 영역(옷)에만 적용하고, 나머지 영역(얼굴/배경)은 원본 픽셀을 그대로 덮어씌움
        from PIL import ImageFilter
        blur_mask = mask_image.filter(ImageFilter.GaussianBlur(radius=3))
        output_image = Image.composite(output_image, person_image.convert("RGB"), blur_mask)

        print("[FashionInpainter] 생성 및 수동 마스크 합성 완료!")
        return output_image

    def batch_try_on(
        self,
        person_image: Image.Image,
        mask_image: Image.Image,
        clothing_prompts: list[str],
        output_dir: str = "data/results",
        **kwargs,
    ) -> list[Image.Image]:
        """
        여러 의상 프롬프트로 룩북 일괄 생성

        Args:
            clothing_prompts: 의상 설명 리스트
            output_dir: 결과 저장 폴더

        Returns:
            생성된 이미지 리스트
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []

        for i, prompt in enumerate(clothing_prompts):
            print(f"\n[{i+1}/{len(clothing_prompts)}] 생성 중: {prompt[:50]}...")
            result = self.try_on(person_image, mask_image, prompt, **kwargs)
            results.append(result)

            save_path = os.path.join(output_dir, f"outfit_{i+1:02d}.png")
            result.save(save_path)
            print(f"  저장: {save_path}")

        print(f"\n룩북 생성 완료! {len(results)}장 저장됨: {output_dir}")
        return results


# ─────────────────────────────────────────────
# 직접 실행 테스트용
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=== Inpainting 테스트 ===")
    print("1. 기본 인페인팅 테스트 (LoRA 없음)")

    inpainter = FashionInpainter()
    inpainter.load_pipeline()

    # 테스트 이미지 생성 (실제로는 인물 사진 사용)
    test_person = Image.new("RGB", (512, 512), color=(200, 180, 160))
    test_mask = Image.new("L", (512, 512), color=0)
    # 상의 영역만 마스킹 (중간 부분)
    import numpy as np
    mask_arr = np.array(test_mask)
    mask_arr[150:350, 100:400] = 255
    test_mask = Image.fromarray(mask_arr)

    result = inpainter.try_on(
        person_image=test_person,
        mask_image=test_mask,
        clothing_prompt="a blue denim jacket, casual fashion",
        num_steps=30,
        seed=42,
    )

    os.makedirs("data/results", exist_ok=True)
    result.save("data/results/test_inpainting.png")
    print("테스트 결과 저장: data/results/test_inpainting.png")
