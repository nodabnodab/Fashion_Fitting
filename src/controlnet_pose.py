"""
controlnet_pose.py - ControlNet 포즈 추출 및 적용 모듈

역할: 인물의 포즈(팔 위치, 몸 방향 등)를 고정합니다.
      옷을 갈아입혀도 포즈가 바뀌지 않도록 합니다.

주요 기능:
- OpenPose로 관절 키포인트 추출
- Canny Edge로 옷 형태/주름 고정
- ControlNet 조건부 이미지 생성
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional


class PoseController:
    """
    ControlNet + OpenPose 기반 포즈 고정 클래스

    사용 예시:
        controller = PoseController()
        pose_image = controller.extract_pose(person_image)
        result = controller.generate_with_pose(
            pose_image=pose_image,
            prompt="a person wearing a white shirt",
        )
    """

    def __init__(self, device: str = "auto"):
        import torch
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.pipe = None
        self.pose_detector = None

    def extract_pose(self, image: Image.Image) -> Image.Image:
        """
        OpenPose로 인물 포즈 키포인트 추출

        Args:
            image: PIL Image (인물 사진)

        Returns:
            포즈 스켈레톤 이미지 (ControlNet 입력용)
        """
        try:
            from controlnet_aux import OpenposeDetector
            if self.pose_detector is None:
                print("[PoseController] OpenPose 모델 로딩...")
                self.pose_detector = OpenposeDetector.from_pretrained(
                    "lllyasviel/ControlNet"
                )
            pose_image = self.pose_detector(image)
            print("[PoseController] 포즈 추출 완료")
            return pose_image

        except ImportError:
            print("[PoseController] controlnet_aux 없음")
            print("  → pip install controlnet-aux 로 설치하세요")
            return self._extract_pose_canny(image)

    def _extract_pose_canny(self, image: Image.Image) -> Image.Image:
        """OpenPose 없을 때 Canny Edge로 대체"""
        import cv2
        img_array = np.array(image.convert("L"))
        edges = cv2.Canny(img_array, 100, 200)
        return Image.fromarray(edges).convert("RGB")

    def load_controlnet_pipeline(
        self,
        base_model: str = "runwayml/stable-diffusion-v1-5",
        controlnet_model: str = "lllyasviel/sd-controlnet-openpose",
    ):
        """
        ControlNet 파이프라인 로드

        Args:
            base_model: 기본 SD 모델
            controlnet_model: ControlNet 모델
                - OpenPose: "lllyasviel/sd-controlnet-openpose"
                - Canny: "lllyasviel/sd-controlnet-canny"
        """
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

        print(f"[PoseController] ControlNet 로딩: {controlnet_model}")

        controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
        ).to(self.device)

        print("[PoseController] ControlNet 파이프라인 로드 완료!")

    def generate_with_pose(
        self,
        pose_image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        controlnet_conditioning_scale: float = 0.95,
        num_steps: int = 30,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        포즈를 고정한 채 새로운 이미지 생성

        Args:
            pose_image: 포즈 스켈레톤 이미지
            prompt: 생성 프롬프트
            controlnet_conditioning_scale: ControlNet 강도 (0.0~1.0)
            num_steps: 추론 스텝

        Returns:
            생성된 이미지
        """
        if self.pipe is None:
            raise RuntimeError("load_controlnet_pipeline()을 먼저 호출하세요.")

        generator = None
        if seed is not None:
            generator = torch.Generator(self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or "blurry, distorted, low quality",
            image=pose_image,
            num_inference_steps=num_steps,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        )

        return result.images[0]
