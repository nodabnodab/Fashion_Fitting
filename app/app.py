"""
app.py - Gradio 기반 Web UI

실행 방법:
    python app/app.py

그러면 http://localhost:7860 에서 데모 앱을 열 수 있습니다.
"""

import os
import sys
import io

# 윈도우 터미널(cp949) 이모지 출력 시 발생하는 UnicodeEncodeError 방지 (전역 적용)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='backslashreplace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='backslashreplace')

import gradio as gr
from PIL import Image
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import VirtualFittingPipeline
from src.masking import FashionMasker
from src.prompt_generator import PromptGenerator


# ─── 전역 파이프라인 (한 번만 로드) ───────────────────────────
pipeline = None
masker = FashionMasker()
prompt_gen = PromptGenerator()


def initialize_pipeline(lora_path: str = ""):
    """파이프라인 초기화 (첫 실행 또는 LoRA 변경 시)"""
    global pipeline
    try:
        pipeline = VirtualFittingPipeline()
        lora = lora_path.strip() if lora_path else None
        if lora and not os.path.exists(lora):
            lora = None
            msg_prefix = f"⚠️ LoRA 파일 없음, LoRA 미적용. "
        else:
            msg_prefix = ""
        pipeline.setup(lora_path=lora)
        return f"{msg_prefix}✅ 파이프라인 초기화 완료!"
    except Exception as e:
        import traceback
        pipeline = None
        return f"❌ 초기화 실패: {e}\n{traceback.format_exc()}"


def run_fitting(
    person_image,
    style_description,
    target_region,
    style_preset,
    num_steps,
    strength,
    controlnet_scale,
    seed,
    lora_path,
):
    """가상 피팅 실행 함수"""
    global pipeline

    if person_image is None:
        return None, None, "❌ 인물 사진을 업로드해주세요."

    if not style_description.strip():
        return None, None, "❌ 스타일 설명을 입력해주세요."

    # 파이프라인이 없거나 준비 안 된 경우 초기화
    if pipeline is None or not pipeline._pipeline_ready:
        msg = initialize_pipeline(lora_path)
        if "실패" in msg:
            return None, None, msg

    try:
        # 인물 이미지를 임시 파일로 저장
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            person_img = Image.fromarray(person_image).convert("RGB")
            person_img.save(f.name)
            temp_path = f.name

        # 파이프라인 실행
        preset = style_preset if style_preset != "없음" else None
        result = pipeline.run(
            person_image_path=temp_path,
            style_description=style_description,
            target_region=target_region,
            style_preset=preset,
            seed=int(seed) if seed else None,
            num_steps=int(num_steps),
            strength=float(strength),
            controlnet_scale=float(controlnet_scale),
        )

        # 결과 이미지 로드
        result_img = Image.open(result["result_path"])
        mask_img = Image.open(result["mask_path"]) if result.get("mask_path") else None

        status = (
            f"✅ 생성 완료! 소요: {result['elapsed_time']:.1f}초\n"
            f"📝 사용된 프롬프트:\n{result['prompt']}"
        )

        return result_img, mask_img, status

    except Exception as e:
        import traceback
        error_msg = f"❌ 오류 발생:\n{str(e)}\n\n{traceback.format_exc()}"
        return None, None, error_msg

    finally:
        if 'temp_path' in locals():
            os.unlink(temp_path)


def generate_prompt_preview(style_description, style_preset):
    """프롬프트 미리보기"""
    if not style_description.strip():
        return "스타일 설명을 입력하면 여기에 SD 프롬프트가 미리보기됩니다."
    preset = style_preset if style_preset != "없음" else None
    return prompt_gen.generate(style_description, preset)


# ─── UI 구성 ───────────────────────────────────────────────────
CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
}
.header-title {
    text-align: center;
    font-size: 2em;
    font-weight: bold;
    margin-bottom: 0.5em;
}
.status-box {
    font-family: monospace;
    font-size: 0.85em;
}
"""

with gr.Blocks(
    title="AI 가상 피팅 & 룩북 생성기"
) as demo:

    gr.HTML("""
    <div class="header-title">
        👗 AI 가상 피팅 & 룩북 생성기
    </div>
    <p style="text-align:center; color: #666;">
        LoRA로 인물 고유성을 유지하며, 자연어로 원하는 스타일을 입혀보세요.
    </p>
    """)

    with gr.Tabs():

        # ── Tab 1: 가상 피팅 ──────────────────────────────────
        with gr.TabItem("👕 가상 피팅", id="fitting"):
            with gr.Row():
                # 왼쪽: 입력
                with gr.Column(scale=1):
                    gr.Markdown("### 📸 인물 사진 업로드")
                    person_input = gr.Image(
                        label="인물 사진 (정면 사진 권장)",
                        type="numpy",
                        height=300,
                        elem_id="person_input",
                    )

                    gr.Markdown("### ✍️ 스타일 설정")
                    style_input = gr.Textbox(
                        label="원하는 스타일 설명",
                        placeholder="예: 시원한 여름 린넨 셔츠, 화이트 컬러\n예: 가을 캠퍼스 룩, 따뜻하고 캐주얼하게",
                        lines=3,
                        elem_id="style_input",
                    )

                    with gr.Row():
                        target_region = gr.Dropdown(
                            label="의상 부위",
                            choices=["upper_body", "lower_body", "full_body"],
                            value="upper_body",
                            elem_id="target_region",
                        )
                        style_preset = gr.Dropdown(
                            label="스타일 프리셋",
                            choices=["없음", "luxury", "casual", "streetwear", "minimal"],
                            value="없음",
                            elem_id="style_preset",
                        )

                    prompt_preview = gr.Textbox(
                        label="🔍 SD 프롬프트 미리보기",
                        interactive=False,
                        lines=2,
                        elem_id="prompt_preview",
                    )

                    with gr.Accordion("⚙️ 고급 설정", open=False):
                        num_steps = gr.Slider(
                            minimum=10, maximum=50, value=30, step=5,
                            label="추론 스텝 수 (많을수록 품질 좋음, 느림)",
                            elem_id="num_steps",
                        )
                        strength = gr.Slider(
                            minimum=0.1, maximum=1.0, value=0.85, step=0.05,
                            label="변형 강도 (Denoising Strength)",
                            info="색상만 바꾸려면 0.4~0.5, 아예 새로운 옷을 입히려면 0.85"
                        )
                        controlnet_scale = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                            label="형태 보존 강도 (ControlNet Canny)",
                            info="원래 옷의 주름과 형태를 얼마나 유지할지 결정 (0.0 = 미적용)"
                        )
                        seed_input = gr.Number(
                            label="시드 (같은 시드 = 같은 결과, 비워두면 랜덤)",
                            value=None,
                            elem_id="seed_input",
                        )
                        lora_path = gr.Textbox(
                            label="LoRA 파일 경로 (선택)",
                            placeholder="예: models/lora/my_person.safetensors",
                            elem_id="lora_path",
                        )

                    run_btn = gr.Button(
                        "🚀 가상 피팅 시작",
                        variant="primary",
                        size="lg",
                        elem_id="run_btn",
                    )

                # 오른쪽: 출력
                with gr.Column(scale=1):
                    gr.Markdown("### 🎨 결과")
                    result_output = gr.Image(
                        label="피팅 결과",
                        height=350,
                        elem_id="result_output",
                    )
                    mask_output = gr.Image(
                        label="의상 마스크 (자동 감지 영역)",
                        height=150,
                        elem_id="mask_output",
                    )
                    status_output = gr.Textbox(
                        label="상태",
                        lines=4,
                        interactive=False,
                        elem_id="status_output",
                        elem_classes=["status-box"],
                    )

        # ── Tab 2: 룩북 생성 ──────────────────────────────────
        with gr.TabItem("📚 룩북 일괄 생성", id="lookbook"):
            gr.Markdown("""
            ### 여러 스타일을 한 번에 생성합니다
            한 줄에 하나씩 스타일을 입력하세요.
            """)

            with gr.Row():
                with gr.Column():
                    lookbook_image = gr.Image(
                        label="인물 사진",
                        type="numpy",
                        height=250,
                        elem_id="lookbook_image",
                    )
                    lookbook_styles = gr.Textbox(
                        label="스타일 목록 (한 줄에 하나씩)",
                        placeholder="여름 린넨 셔츠, 화이트\n가을 니트 스웨터, 베이지\n캐주얼 청바지 재킷\n블랙 포멀 자켓",
                        lines=6,
                        elem_id="lookbook_styles",
                    )
                    lookbook_btn = gr.Button(
                        "📸 룩북 생성",
                        variant="primary",
                        elem_id="lookbook_btn",
                    )

                with gr.Column():
                    lookbook_gallery = gr.Gallery(
                        label="룩북 결과",
                        columns=2,
                        height=400,
                        elem_id="lookbook_gallery",
                    )
                    lookbook_status = gr.Textbox(
                        label="상태",
                        lines=3,
                        interactive=False,
                        elem_id="lookbook_status",
                    )

        # ── Tab 3: 설정 & 도움말 ──────────────────────────────
        with gr.TabItem("⚙️ 설정 & 도움말", id="help"):
            gr.Markdown("""
            ## 사용 방법

            ### 1. 가상 피팅
            1. **인물 사진** 업로드 (정면, 전신 사진 권장)
            2. **스타일 설명** 입력 (한국어 가능)
            3. **의상 부위** 선택 (상의/하의/전신)
            4. **가상 피팅 시작** 클릭

            ### 2. LoRA 활용 (얼굴 일관성)
            `scripts/train_lora.bat`을 실행해 LoRA를 먼저 학습하세요.
            - 사진 15~20장을 `data/input_faces/`에 넣고 실행
            - 학습 후 `models/lora/` 폴더에 `.safetensors` 파일 생성
            - 고급 설정에서 해당 파일 경로 입력

            ### 3. 팁
            - **좋은 결과를 위한 조건**: 정면 사진, 단색 배경, 좋은 조명
            - **스텝 수**: 빠른 테스트는 10~15, 최종 결과물은 30~50
            - **시드 고정**: 같은 시드를 쓰면 재현 가능한 결과 생성

            ## 기술 스택
            | 기술 | 역할 |
            |------|------|
            | Stable Diffusion | 이미지 생성 엔진 |
            | LoRA | 인물 고유성 유지 |
            | YOLO + SAM | 자동 의상 마스킹 |
            | Gemini API | 자연어 → 프롬프트 변환 |
            | Gradio | 웹 UI |
            """)

            with gr.Row():
                init_btn = gr.Button("🔄 파이프라인 재초기화", elem_id="init_btn")
                init_lora = gr.Textbox(
                    label="LoRA 경로 (선택)",
                    placeholder="models/lora/my_person.safetensors",
                    elem_id="init_lora",
                )
            init_status = gr.Textbox(label="초기화 상태", interactive=False, elem_id="init_status")

    # ─── 이벤트 연결 ────────────────────────────────────────────

    # 프롬프트 미리보기 (실시간)
    style_input.change(
        fn=generate_prompt_preview,
        inputs=[style_input, style_preset],
        outputs=prompt_preview,
    )
    style_preset.change(
        fn=generate_prompt_preview,
        inputs=[style_input, style_preset],
        outputs=prompt_preview,
    )

    # 가상 피팅 실행
    run_btn.click(
        fn=run_fitting,
        inputs=[person_input, style_input, target_region, style_preset,
                num_steps, strength, controlnet_scale, seed_input, lora_path],
        outputs=[result_output, mask_output, status_output],
    )

    # 룩북 생성
    def run_lookbook_ui(person_image, styles_text, progress=gr.Progress()):
        global pipeline
        if pipeline is None:
            initialize_pipeline()

        if person_image is None:
            return [], "❌ 인물 사진을 업로드해주세요."

        styles = [s.strip() for s in styles_text.strip().split("\n") if s.strip()]
        if not styles:
            return [], "❌ 스타일을 입력해주세요."

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            Image.fromarray(person_image).save(f.name)
            temp_path = f.name

        results_imgs = []
        for i, style in enumerate(progress.tqdm(styles, desc="룩북 생성 중")):
            try:
                result = pipeline.run(
                    person_image_path=temp_path,
                    style_description=style,
                    target_region="upper_body",
                )
                results_imgs.append(result["result_path"])
            except Exception as e:
                print(f"스타일 '{style}' 생성 실패: {e}")

        os.unlink(temp_path)
        return results_imgs, f"✅ {len(results_imgs)}/{len(styles)} 생성 완료!"

    lookbook_btn.click(
        fn=run_lookbook_ui,
        inputs=[lookbook_image, lookbook_styles],
        outputs=[lookbook_gallery, lookbook_status],
    )

    # 파이프라인 재초기화
    init_btn.click(
        fn=initialize_pipeline,
        inputs=[init_lora],
        outputs=[init_status],
    )


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  AI 가상 피팅 앱 시작")
    print("  http://localhost:7860 에서 접속하세요")
    print("=" * 50 + "\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="violet"),
        css=CSS
    )
