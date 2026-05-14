"""
masking.py - SegFormer Human Parsing 기반 자동 의상 마스킹 모듈

역할: 입력 이미지에서 픽셀 단위로 18개 신체/의상 부위를 분류(파싱)합니다.
얼굴, 머리카락, 피부는 철통 방어하여 마스크에서 완전히 제외하고,
오직 사용자가 교체 원하는 의상 영역만 정밀하게 마스크로 추출합니다.

사용 모델: mattmdjaga/segformer_b2_clothes (HuggingFace)
"""

import numpy as np
from PIL import Image
import cv2
from typing import Optional, Literal


# SegFormer 파싱 모델의 클래스 인덱스 정의 (18개 카테고리)
PARSING_CLASSES = {
    0:  "Background",
    1:  "Hat",
    2:  "Hair",
    3:  "Sunglasses",
    4:  "Upper-clothes",
    5:  "Skirt",
    6:  "Pants",
    7:  "Dress",
    8:  "Belt",
    9:  "Left-shoe",
    10: "Right-shoe",
    11: "Face",
    12: "Left-leg",
    13: "Right-leg",
    14: "Left-arm",
    15: "Right-arm",
    16: "Bag",
    17: "Scarf",
}

# 교체 타겟에 따라 마스크로 추출할 클래스 인덱스
TARGET_CLASSES = {
    "upper_body": [4, 7],       # 상의, 드레스
    "lower_body": [5, 6],       # 치마, 바지
    "full_body":  [4, 5, 6, 7], # 상의 + 하의 + 드레스
}


class FashionMasker:
    """
    SegFormer Human Parsing 기반 패션 마스킹 클래스.

    기존 YOLO+SAM의 좌표 기반 접근법을 완전히 폐기하고,
    픽셀 단위로 신체 부위를 분류하는 Semantic Segmentation으로 대체합니다.

    사용 예시:
        masker = FashionMasker()
        mask = masker.get_mask(image, target="upper_body")
        preview = masker.apply_mask_preview(image, mask)
    """

    PARSING_MODEL_ID = "mattmdjaga/segformer_b2_clothes"

    def __init__(self, device: str = "auto"):
        import torch
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.processor = None
        self.model = None
        self._models_loaded = False

        print(f"[FashionMasker] 디바이스: {self.device}")

    def _load_models(self):
        """모델을 처음 필요할 때만 로드 (지연 로딩)"""
        if self._models_loaded:
            return

        print("[FashionMasker] SegFormer Human Parsing 모델 로딩 중...")
        print(f"  모델: {self.PARSING_MODEL_ID}")
        print("  (처음 실행 시 HuggingFace에서 자동 다운로드)")

        try:
            from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
            import torch

            self.processor = SegformerImageProcessor.from_pretrained(self.PARSING_MODEL_ID)
            self.model = AutoModelForSemanticSegmentation.from_pretrained(self.PARSING_MODEL_ID)
            self.model.to(self.device)
            self.model.eval()
            print("[FashionMasker] SegFormer 모델 로드 완료!")

        except Exception as e:
            print(f"[FashionMasker] SegFormer 로드 실패: {e}")
            self.model = None

        self._models_loaded = True

    def _parse_image(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        이미지를 Human Parsing하여 각 픽셀의 카테고리 레이블 맵을 반환합니다.

        Returns:
            np.ndarray: (H, W) 크기의 정수 배열, 값은 0~17의 클래스 인덱스
                        실패 시 None 반환
        """
        if self.model is None:
            return None

        import torch

        # SegFormer 입력 전처리
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 로짓에서 픽셀별 클래스 레이블 추출
        logits = outputs.logits.cpu()  # (1, num_labels, H/4, W/4)

        # 원본 이미지 크기로 업스케일
        upsampled = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # (H, W)
            mode="bilinear",
            align_corners=False,
        )
        label_map = upsampled.argmax(dim=1).squeeze().numpy()  # (H, W)
        return label_map

    def get_mask(
        self,
        image: Image.Image,
        target: Literal["upper_body", "lower_body", "full_body"] = "upper_body",
        use_sam: bool = False,  # 하위 호환성을 위해 인수는 유지하지만 무시됨
    ) -> Optional[Image.Image]:
        """
        SegFormer Human Parsing으로 의상 마스크를 생성합니다.

        - 얼굴(11), 머리카락(2), 피부(팔/다리 12~15)는 절대 마스크에 포함하지 않습니다.
        - 오직 target에 해당하는 의상 클래스 픽셀만 흰색(255)으로 마스킹합니다.

        Args:
            image: PIL Image (RGB)
            target: "upper_body" | "lower_body" | "full_body"

        Returns:
            PIL Image (L 모드, 흰색=교체 대상 영역)
        """
        self._load_models()

        if self.model is None:
            print("[FashionMasker] SegFormer 사용 불가 - 빈 마스크 반환")
            return None

        print(f"[FashionMasker] Human Parsing 시작 ({target})...")
        label_map = self._parse_image(image.convert("RGB"))

        if label_map is None:
            print("[FashionMasker] Parsing 실패")
            return None

        # 타겟 의상 클래스만 추출
        target_class_ids = TARGET_CLASSES.get(target, [4])
        mask = np.zeros(label_map.shape, dtype=np.uint8)
        for cls_id in target_class_ids:
            mask[label_map == cls_id] = 255

        # 안전장치: 얼굴(11), 머리카락(2) 픽셀은 마스크에서 무조건 제거
        # (SegFormer가 이미 분리해주지만, 이중 방어)
        forbidden_classes = [2, 11]  # Hair, Face
        for cls_id in forbidden_classes:
            mask[label_map == cls_id] = 0

        # 마스크가 너무 작으면 (감지 실패) 경고
        mask_ratio = np.sum(mask > 0) / mask.size
        if mask_ratio < 0.01:
            print(f"[FashionMasker] 경고: 의상 마스크 면적이 너무 작음 ({mask_ratio:.2%})")
            print("  → 다른 각도의 사진이나 더 큰 의상 사진을 사용해보세요.")

        # 마스크 후처리: 노이즈 제거 (Morphological Opening)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # 엣지 부드럽게 (가장자리 픽셀 한두겹 제거로 경계선 깔끔하게)
        mask = cv2.erode(mask, kernel, iterations=1)

        print(f"[FashionMasker] Human Parsing 완료 (마스크 면적: {mask_ratio:.2%})")
        return Image.fromarray(mask).convert("L")

    def get_detected_labels(self, image: Image.Image) -> dict:
        """
        이미지에서 감지된 모든 의상/신체 부위 레이블을 반환합니다. (디버깅용)

        Returns:
            dict: {클래스명: 면적_퍼센트} 형태의 딕셔너리
        """
        self._load_models()
        if self.model is None:
            return {}

        label_map = self._parse_image(image.convert("RGB"))
        if label_map is None:
            return {}

        total = label_map.size
        result = {}
        for cls_id, cls_name in PARSING_CLASSES.items():
            count = np.sum(label_map == cls_id)
            if count > 0:
                result[cls_name] = round(count / total * 100, 2)

        return dict(sorted(result.items(), key=lambda x: -x[1]))

    def apply_mask_preview(
        self, image: Image.Image, mask: Image.Image, alpha: float = 0.5
    ) -> Image.Image:
        """마스크 영역을 반투명 빨간색으로 시각화 (디버깅용)"""
        img_array = np.array(image.convert("RGB"))
        mask_array = np.array(mask)

        overlay = img_array.copy()
        overlay[mask_array > 128] = [255, 50, 50]  # 빨간색으로 표시

        blended = cv2.addWeighted(img_array, 1 - alpha, overlay, alpha, 0)
        return Image.fromarray(blended)


if __name__ == "__main__":
    """빠른 테스트 실행"""
    print("=== FashionMasker SegFormer 테스트 ===")
    masker = FashionMasker()

    test_image = Image.new("RGB", (512, 512), color=(200, 150, 100))
    print("\n감지된 레이블:")
    labels = masker.get_detected_labels(test_image)
    for name, pct in labels.items():
        print(f"  {name}: {pct:.2f}%")
