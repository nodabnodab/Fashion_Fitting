"""
masking.py - YOLO + SAM 기반 자동 의상 마스킹 모듈

역할: 입력 이미지에서 상의/하의/전신 영역을 자동으로 감지하고 마스크를 생성합니다.
이 마스크를 Inpainting에 전달하면 해당 영역의 옷만 자연스럽게 교체됩니다.
"""

import numpy as np
from PIL import Image
import cv2
from typing import Optional, Literal


# YOLOv8로 감지할 의상 클래스 ID (COCO 기준)
# 실제 패션 특화 모델 사용 시 클래스 ID가 달라질 수 있음
FASHION_CLASSES = {
    "upper_body": ["shirt", "jacket", "hoodie", "top"],
    "lower_body": ["pants", "skirt", "shorts"],
    "full_body": ["dress", "jumpsuit"],
}


class FashionMasker:
    """
    YOLO + SAM을 조합한 패션 의상 마스킹 클래스

    사용 예시:
        masker = FashionMasker()
        mask = masker.get_mask(image, target="upper_body")
        masked_image = masker.apply_mask_preview(image, mask)
    """

    def __init__(self, device: str = "auto"):
        """
        Args:
            device: "auto" | "cuda" | "cpu"
                    auto면 GPU 있으면 GPU, 없으면 CPU 사용
        """
        import torch
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.yolo_model = None
        self.sam_model = None
        self._models_loaded = False

        print(f"[FashionMasker] 디바이스: {self.device}")

    def _load_models(self):
        """모델을 처음 필요할 때만 로드 (지연 로딩)"""
        if self._models_loaded:
            return

        print("[FashionMasker] 모델 로딩 중...")

        try:
            from ultralytics import YOLO
            # YOLOv8 nano 모델 (가벼움, 포트폴리오용으로 충분)
            # 추후 패션 특화 모델로 교체 가능
            self.yolo_model = YOLO("yolov8n-seg.pt")
            print("[FashionMasker] YOLOv8 로드 완료")
        except Exception as e:
            print(f"[FashionMasker] YOLO 로드 실패: {e}")
            print("  → 'pip install ultralytics' 로 설치하세요")

        try:
            from segment_anything import sam_model_registry, SamPredictor
            import os
            sam_checkpoint = "models/sam_vit_b_01ec64.pth"  # SAM ViT-B (가벼운 버전)
            if os.path.exists(sam_checkpoint):
                sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
                sam.to(device=self.device)
                self.sam_model = SamPredictor(sam)
                print("[FashionMasker] SAM 로드 완료")
            else:
                print(f"[FashionMasker] SAM 체크포인트 없음: {sam_checkpoint}")
                print("  → notebooks/01_setup_test.ipynb 에서 다운로드 방법 확인")
        except Exception as e:
            print(f"[FashionMasker] SAM 로드 실패: {e}")

        self._models_loaded = True

    def get_mask_from_yolo(
        self,
        image: Image.Image,
        target: Literal["upper_body", "lower_body", "full_body"] = "upper_body",
        confidence: float = 0.25,
    ) -> Optional[np.ndarray]:
        """
        [백업 로직] YOLOv8 세그멘테이션으로 인물 전체 영역 감지
        
        YOLOv8 COCO 기본 모델은 'person'만 감지 가능하므로, 
        인물의 대략적인 전체 실루엣을 가져옵니다.
        """
        self._load_models()
        if self.yolo_model is None:
            return None

        img_array = np.array(image)
        results = self.yolo_model(img_array, conf=confidence, verbose=False)

        h, w = img_array.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        detected = False

        for result in results:
            if result.masks is None:
                continue
            for seg_mask, cls_id in zip(result.masks.data, result.boxes.cls):
                class_name = self.yolo_model.names[int(cls_id)]
                # 의상 클래스가 직접 없으므로 인물을 탐색
                if class_name == "person":
                    mask_resized = cv2.resize(
                        seg_mask.cpu().numpy().astype(np.uint8),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    combined_mask = np.maximum(combined_mask, mask_resized * 255)
                    detected = True

        if not detected:
            return None

        return combined_mask

    def get_mask(
        self,
        image: Image.Image,
        target: Literal["upper_body", "lower_body", "full_body"] = "upper_body",
        use_sam: bool = True,
    ) -> Optional[Image.Image]:
        """
        메인 인터페이스: YOLOv8로 인물 위치를 잡고 SAM 포인트 프롬프트를 사용해 의상을 정밀 격리합니다.

        Args:
            image: PIL Image (RGB)
            target: 마스킹할 부위 ('upper_body', 'lower_body', 'full_body')
            use_sam: SAM 사용 유무 (의상 격리를 위해 필수 권장)

        Returns:
            PIL Image 마스크 (L 모드, 흰색=피팅 대상)
        """
        self._load_models()
        if self.yolo_model is None:
            print("[FashionMasker] YOLO 모델 로드 실패.")
            return None

        img_array = np.array(image.convert("RGB"))
        h, w = img_array.shape[:2]

        # 1. YOLO로 가장 뚜렷한 'person' 탐지
        results = self.yolo_model(img_array, conf=0.25, verbose=False)
        person_box = None
        max_area = 0

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = self.yolo_model.names[cls_id]
                if class_name == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    # 범위를 화면 크기로 제한
                    x1, x2 = max(0, x1), min(w, x2)
                    y1, y2 = max(0, y1), min(h, y2)
                    
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        person_box = [x1, y1, x2, y2]

        if person_box is None:
            print("[FashionMasker] 인물을 찾지 못했습니다. 기본 전신 영역을 가정합니다.")
            person_box = [int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)]

        x1, y1, x2, y2 = person_box
        bw, bh = x2 - x1, y2 - y1

        # 2. 타겟 영역에 따른 핵심 힌트 포인트(Point Prompt) 계산
        prompt_points = []
        prompt_labels = []

        if target == "upper_body":
            # 가슴 정중앙 부근 포인트 2개로 안정성 확보
            prompt_points.append([x1 + int(bw * 0.5), y1 + int(bh * 0.35)])
            prompt_points.append([x1 + int(bw * 0.5), y1 + int(bh * 0.45)])
            prompt_labels.extend([1, 1])
            # 머리 부분 제외 힌트 (Negative prompt)
            prompt_points.append([x1 + int(bw * 0.5), y1 + int(bh * 0.1)])
            prompt_labels.append(0)

        elif target == "lower_body":
            # 허벅지-무릎 사이 정중앙
            prompt_points.append([x1 + int(bw * 0.5), y1 + int(bh * 0.75)])
            prompt_points.append([x1 + int(bw * 0.5), y1 + int(bh * 0.85)])
            prompt_labels.extend([1, 1])
            # 상체 제외 힌트
            prompt_points.append([x1 + int(bw * 0.5), y1 + int(bh * 0.35)])
            prompt_labels.append(0)

        else:  # full_body
            # 상체, 하체 골고루 힌트
            prompt_points.append([x1 + int(bw * 0.5), y1 + int(bh * 0.4)])
            prompt_points.append([x1 + int(bw * 0.5), y1 + int(bh * 0.75)])
            prompt_labels.extend([1, 1])

        # 3. SAM 적용
        if use_sam and self.sam_model is not None:
            print(f"[FashionMasker] SAM 포인트 프롬프트 구동 중 ({target})...")
            self.sam_model.set_image(img_array)
            
            input_points = np.array(prompt_points)
            input_labels = np.array(prompt_labels)
            
            # SAM 박스 힌트 제공하여 인물 외부 침범 최소화
            input_box = np.array([x1, y1, x2, y2])

            masks, scores, _ = self.sam_model.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=input_box[None, :] if target != "full_body" else None,
                multimask_output=True,
            )
            
            # 스코어 기반 최적 마스크 추출
            best_mask = masks[np.argmax(scores)]
            final_mask = (best_mask * 255).astype(np.uint8)
            print(f"[FashionMasker] SAM 마스킹 완료 (신뢰도: {max(scores):.3f})")
            return Image.fromarray(final_mask).convert("L")
        
        else:
            # SAM이 없거나 비활성화 시 YOLO 마스크 대체 시도
            print("[FashionMasker] SAM을 사용할 수 없어 YOLO 인물 실루엣을 대체 활용합니다.")
            yolo_mask = self.get_mask_from_yolo(image, confidence=0.25)
            if yolo_mask is not None:
                return Image.fromarray(yolo_mask).convert("L")
            return None

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

    def _is_target_class(self, class_name: str, target: str) -> bool:
        """YOLO 클래스명이 목표 의상 부위에 해당하는지 확인 (레거시 호환)"""
        return class_name == "person"


# ─────────────────────────────────────────────
# 직접 실행 테스트용
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python masking.py <이미지 경로>")
        print("예시: python masking.py data/test_images/person.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    image = Image.open(image_path).convert("RGB")

    masker = FashionMasker()
    mask = masker.get_mask(image, target="upper_body")

    if mask:
        preview = masker.apply_mask_preview(image, mask)
        preview.save("data/results/mask_preview.jpg")
        print("마스크 미리보기 저장: data/results/mask_preview.jpg")
    else:
        print("마스크 생성 실패")
