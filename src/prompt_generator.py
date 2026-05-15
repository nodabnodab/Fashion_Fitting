"""
prompt_generator.py - LLM 기반 자연어 → SD 프롬프트 변환 모듈

역할: 사용자가 "시원한 여름 휴가 느낌"처럼 자연어로 입력하면,
      Stable Diffusion이 이해할 수 있는 상세한 프롬프트로 변환합니다.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


SYSTEM_PROMPT = """당신은 패션 화보 촬영 디렉터이자 Stable Diffusion 프롬프트 전문가입니다.

사용자가 원하는 패션 스타일이나 분위기를 자연어로 설명하면,
Stable Diffusion Inpainting에 최적화된 영문 프롬프트로 변환해주세요.

규칙:
1. 반드시 영어로 작성 (SD는 영문 프롬프트가 더 효과적)
2. 의상 소재, 색상, 스타일, 배경, 조명을 구체적으로 포함
3. 패션 잡지 화보 퀄리티를 목표로
4. 쉼표로 구분된 키워드 나열 형식
5. 프롬프트만 출력 (설명 없이)

예시 입력: "여름 휴가 느낌의 시원한 옷"
예시 출력: "light linen shirt, sky blue color, breathable fabric, beach background, golden hour lighting, summer vacation, fashion magazine editorial, high quality"
"""


class PromptGenerator:
    """
    Gemini API를 활용한 자연어 → SD 프롬프트 변환기

    사용 예시:
        gen = PromptGenerator()
        prompt = gen.generate("가을 캠퍼스 룩, 따뜻하고 편안한 느낌")
        # 출력: "oversized beige knit sweater, warm autumn palette, ..."
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = None

        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            print("[PromptGenerator] ⚠️  Gemini API 키 없음")
            print("  → .env 파일에 GEMINI_API_KEY를 설정하세요")
            print("  → https://aistudio.google.com/ 에서 무료 발급")
            print("  → 키 없으면 규칙 기반 변환(fallback)을 사용합니다")
        else:
            self._init_gemini()

    def _init_gemini(self):
        """Gemini 모델 초기화 (google-genai SDK 사용)"""
        try:
            from google import genai
            self._genai_client = genai.Client(api_key=self.api_key)
            self.model = "gemini-2.0-flash"  # 모델명 문자열로 보관
            print("[PromptGenerator] Gemini API 연결 완료 (google-genai)")
        except Exception as e:
            print(f"[PromptGenerator] Gemini 초기화 실패: {e}")
            self.model = None
            self._genai_client = None

    def generate(self, user_input: str, style_preset: Optional[str] = None) -> str:
        """
        자연어 입력을 SD 프롬프트로 변환

        Args:
            user_input: 사용자의 자연어 스타일 설명 (한국어/영어 모두 가능)
            style_preset: 사전 정의된 스타일 ("luxury", "casual", "streetwear", "minimal")

        Returns:
            SD 프롬프트 문자열
        """
        # 스타일 프리셋 적용
        preset_suffix = self._get_preset_suffix(style_preset)
        full_input = f"{user_input} {preset_suffix}".strip()

        if self.model:
            return self._generate_with_gemini(full_input)
        else:
            return self._generate_fallback(full_input)

    def _generate_with_gemini(self, user_input: str) -> str:
        """Gemini API로 프롬프트 생성 (google-genai SDK)"""
        try:
            response = self._genai_client.models.generate_content(
                model=self.model,
                contents=f"{SYSTEM_PROMPT}\n\n사용자 입력: {user_input}",
            )
            generated = response.text.strip()
            print(f"[PromptGenerator] 생성된 프롬프트: {generated[:100]}...")
            return generated
        except Exception as e:
            print(f"[PromptGenerator] Gemini 호출 실패: {e}")
            return self._generate_fallback(user_input)

    def _generate_fallback(self, user_input: str) -> str:
        """
        API 없을 때 규칙 기반 폴백 변환
        간단한 키워드 매핑으로 기본적인 프롬프트 생성
        """
        keywords = []
        user_lower = user_input.lower()

        # 계절 감지
        season_map = {
            "여름": "summer fashion, light fabric, breathable",
            "겨울": "winter fashion, thick fabric, warm",
            "봄": "spring fashion, pastel colors, fresh",
            "가을": "autumn fashion, earth tones, layered",
            "summer": "summer fashion, light fabric, breathable",
            "winter": "winter fashion, thick fabric, warm",
        }
        for k, v in season_map.items():
            if k in user_lower:
                keywords.append(v)

        # 스타일 감지
        style_map = {
            "캐주얼": "casual style, relaxed fit",
            "포멀": "formal business attire",
            "스트릿": "streetwear, urban fashion",
            "미니멀": "minimalist fashion, clean lines",
            "럭셔리": "luxury fashion, high-end designer",
            "casual": "casual style, relaxed fit",
            "formal": "formal business attire",
            "luxury": "luxury fashion, high-end designer",
        }
        for k, v in style_map.items():
            if k in user_lower:
                keywords.append(v)

        # 색상 감지 (API 실패 시 대비)
        color_map = {
            "빨간": "red", "빨강": "red", "레드": "red",
            "파란": "blue", "파랑": "blue", "블루": "blue",
            "초록": "green", "그린": "green",
            "노란": "yellow", "노랑": "yellow", "옐로우": "yellow",
            "검은": "black", "검정": "black", "블랙": "black",
            "흰색": "white", "하얀": "white", "화이트": "white",
            "회색": "grey", "그레이": "grey",
            "핑크": "pink", "분홍": "pink",
            "보라": "purple", "퍼플": "purple",
            "갈색": "brown", "브라운": "brown",
            "베이지": "beige",
        }
        for k, v in color_map.items():
            if k in user_lower:
                keywords.append(v)

        # 기본 품질 태그
        keywords.extend([
            "high quality", "detailed fabric texture",
            "fashion magazine editorial", "professional lighting",
            "8k resolution"
        ])

        # 원본 입력도 포함
        keywords.insert(0, user_input)

        return ", ".join(keywords)

    def _get_preset_suffix(self, preset: Optional[str]) -> str:
        """스타일 프리셋 문자열 반환"""
        presets = {
            "luxury": "luxury brand, haute couture, Vogue magazine quality",
            "casual": "everyday casual, relaxed, comfortable",
            "streetwear": "streetwear, urban style, hypebeast",
            "minimal": "minimalist, clean, monochrome, architectural fashion",
        }
        return presets.get(preset, "") if preset else ""

    def generate_negative_prompt(self, user_input: str = "") -> str:
        """
        negative prompt 생성 (제외할 요소들)
        일반적으로 공통 negative prompt를 사용하므로 간단하게 유지
        """
        base_negative = (
            "blurry, distorted, deformed, ugly, bad anatomy, "
            "extra limbs, missing limbs, watermark, text, logo, "
            "low quality, jpeg artifacts, grainy"
        )
        return base_negative


# 빠른 사용을 위한 편의 함수
def generate_prompt(user_input: str, style_preset: Optional[str] = None) -> str:
    """모듈 레벨 편의 함수"""
    gen = PromptGenerator()
    return gen.generate(user_input, style_preset)


# ─────────────────────────────────────────────
# 직접 실행 테스트용
# ─────────────────────────────────────────────
if __name__ == "__main__":
    gen = PromptGenerator()

    test_inputs = [
        ("시원한 여름 휴가 느낌", None),
        ("가을 캠퍼스 룩, 따뜻하고 편안한", "casual"),
        ("면접용 정장, 깔끔하고 전문적", "formal"),
        ("명품 브랜드 화보 느낌", "luxury"),
    ]

    print("\n=== 프롬프트 생성 테스트 ===\n")
    for user_input, preset in test_inputs:
        print(f"입력: {user_input}")
        result = gen.generate(user_input, preset)
        print(f"출력: {result}")
        print("-" * 60)
