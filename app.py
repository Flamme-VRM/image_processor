#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
PREMIUM IMAGE PROCESSOR — Gradio Web UI
Drag & Drop ZIP → Remove Watermarks → AI Upscale → Download
╚══════════════════════════════════════════════════════════════════════╝

Usage:
    pip install gradio
    python app.py

Then open http://localhost:7860 in your browser.
"""

import os
import sys
import io
import time
import shutil
import zipfile
import tempfile
import logging
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

import gradio as gr

# Import pipeline
sys.path.insert(0, str(Path(__file__).parent))
from image_processor import ImageProcessingPipeline, log

# Log capture handler

class LogCapture(logging.Handler):
    """Custom logging handler that captures log messages into a list."""

    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    def emit(self, record):
        msg = self.format(record)
        self.records.append(msg)

    def get_log_text(self) -> str:
        return "\n".join(self.records)

    def clear(self):
        self.records.clear()


def process_wrapper(
    zip_file,
    mask_file,
    remove_watermark: bool,
    do_upscale: bool,
    scale_factor: str,
    output_format: str,
    jpg_quality: int,
    use_gpu: bool,
    inpaint_radius: int,
    inpaint_method: str,
    use_redis: bool = False,
    num_workers: int = 0,
):
    """
    Main processing function called by the Gradio button.
    Returns: (log_text: str, result_file_path: str | None)
    """

    #  Validation 
    if zip_file is None:
        return "Ошибка: Загрузите ZIP-архив с изображениями!", None

    zip_path = zip_file.name if hasattr(zip_file, 'name') else str(zip_file)

    if not os.path.exists(zip_path):
        return "Ошибка: Файл ZIP не найден!", None

    if not zipfile.is_zipfile(zip_path):
        return "Ошибка: Загруженный файл не является валидным ZIP-архивом!", None

    #  Parse parameters 
    scale = int(scale_factor.replace("x", ""))
    fmt = output_format.lower()

    # Handle mask file
    mask_path = None
    if mask_file is not None:
        if hasattr(mask_file, 'name'):
            mask_path = mask_file.name
        elif isinstance(mask_file, str):
            mask_path = mask_file

    #  Set up log capture 
    log_capture = LogCapture()
    log_capture.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S"
    ))
    log.addHandler(log_capture)

    #  Create temp output directory 
    output_dir = tempfile.mkdtemp(prefix="gradio_output_")

    try:
        #  Run the pipeline 
        log_capture.records.append("Запуск обработки...\n")

        pipeline = ImageProcessingPipeline(
            zip_path=zip_path,
            output_dir=output_dir,
            mask_path=mask_path,
            scale=scale,
            output_format=fmt,
            jpg_quality=jpg_quality,
            use_gpu=use_gpu,
            skip_watermark=not remove_watermark,
            skip_upscale=not do_upscale,
            inpaint_radius=inpaint_radius,
            inpaint_method=inpaint_method,
            use_redis=use_redis,
            num_workers=num_workers,
        )
        pipeline.run()

        #  Collect results into a ZIP 
        result_files = list(Path(output_dir).glob("*"))
        result_files = [f for f in result_files if f.is_file()]

        if not result_files:
            log_capture.records.append("\nОбработка завершилась, но выходных файлов нет.")
            return log_capture.get_log_text(), None

        # Create result ZIP
        result_zip_path = os.path.join(tempfile.gettempdir(), "processed_results.zip")
        with zipfile.ZipFile(result_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in result_files:
                zf.write(str(file_path), file_path.name)

        file_count = len(result_files)
        total_size_mb = sum(f.stat().st_size for f in result_files) / (1024 * 1024)

        log_capture.records.append(f"\n{'='*50}")
        log_capture.records.append(f"Результат упакован в ZIP:")
        log_capture.records.append(f"   Файлов: {file_count}")
        log_capture.records.append(f"   Размер: {total_size_mb:.1f} MB")
        log_capture.records.append(f"{'='*50}")
        log_capture.records.append("Готово! Нажмите кнопку скачивания ниже.")

        return log_capture.get_log_text(), result_zip_path

    except Exception as e:
        log_capture.records.append(f"\nКритическая ошибка: {e}")
        import traceback
        log_capture.records.append(traceback.format_exc())
        return log_capture.get_log_text(), None

    finally:
        # Remove the log handler to avoid duplicates on next run
        log.removeHandler(log_capture)
        # Clean up temp output dir (ZIP is already created)
        try:
            shutil.rmtree(output_dir, ignore_errors=True)
        except Exception:
            pass


# UI Layout

CUSTOM_CSS = """
/* ─── Global ─── */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* ─── Header ─── */
.app-header {
    text-align: center;
    padding: 20px 0 10px 0;
    margin-bottom: 10px;
}
.app-header h1 {
    font-size: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}
.app-header p {
    opacity: 0.7;
    font-size: 0.95rem;
}

/* ─── Launch button ─── */
.launch-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    padding: 14px 0 !important;
    border: none !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    margin-top: 10px !important;
}
.launch-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
}

/* ─── Section labels ─── */
.section-title {
    font-weight: 600;
    font-size: 1.05rem;
    margin-bottom: 4px;
    padding: 6px 12px;
    border-left: 3px solid #667eea;
    background: rgba(102, 126, 234, 0.08);
    border-radius: 0 6px 6px 0;
}

/* ─── Log output ─── */
.log-output textarea {
    font-family: 'Cascadia Code', 'Fira Code', 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.5 !important;
    background: #1a1a2e !important;
    color: #e0e0e0 !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

/* ─── Cards / Groups ─── */
.gr-group {
    border-radius: 12px !important;
    border: 1px solid rgba(102, 126, 234, 0.2) !important;
}
"""


def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks UI."""

    with gr.Blocks(
        title="Premium Image Processor",
    ) as app:

        #  Header 
        with gr.Column(elem_classes="app-header"):
            gr.Markdown(
                "# Premium Image Processor\n"
                "Удаление водяных знаков + AI-апскейлинг из ZIP-архива"
            )

        #  Main layout: two columns 
        with gr.Row(equal_height=False):

            with gr.Column(scale=1):

                #  File inputs 
                gr.Markdown("### Входные файлы", elem_classes="section-title")
                with gr.Group():
                    zip_input = gr.File(
                        label="ZIP-архив с изображениями",
                        file_types=[".zip"],
                        type="filepath",
                    )
                    mask_input = gr.File(
                        label="Маска водяного знака (опционально)",
                        file_types=["image"],
                        type="filepath",
                    )
                    gr.Markdown(
                        "*Маска: чёрный фон + белые зоны водяного знака. "
                        "Без маски — автоопределение.*",
                    )

                #  Processing toggles 
                gr.Markdown("### Режим обработки", elem_classes="section-title")
                with gr.Group():
                    with gr.Row():
                        chk_watermark = gr.Checkbox(
                            label="Удалить водяные знаки",
                            value=True,
                        )
                        chk_upscale = gr.Checkbox(
                            label="AI-апскейл",
                            value=True,
                        )
                    chk_redis = gr.Checkbox(
                        label="Использовать Redis-кэширование (ускоряет повторы)",
                        value=False,
                        info="Кэширует результаты в Redis. Требует запущенный Redis-сервер.",
                    )

                #  Upscale settings 
                gr.Markdown("### Настройки апскейла", elem_classes="section-title")
                with gr.Group():
                    radio_scale = gr.Radio(
                        choices=["2x", "4x"],
                        value="4x",
                        label="Коэффициент увеличения",
                    )
                    chk_gpu = gr.Checkbox(
                        label="Использовать GPU (CUDA)",
                        value=True,
                    )

                #  Output settings 
                gr.Markdown("### Формат сохранения", elem_classes="section-title")
                with gr.Group():
                    dd_format = gr.Dropdown(
                        choices=["png", "jpg", "webp"],
                        value="webp",
                        label="Формат выходных файлов",
                    )
                    slider_quality = gr.Slider(
                        minimum=1, maximum=100, value=100, step=1,
                        label="Качество JPG (если формат = jpg)",
                    )

                #  Inpainting advanced settings 
                with gr.Accordion("Расширенные настройки инпейнтинга", open=False):
                    slider_radius = gr.Slider(
                        minimum=1, maximum=20, value=7, step=1,
                        label="Радиус инпейнтинга (px)",
                    )
                    dd_method = gr.Dropdown(
                        choices=["telea", "ns"],
                        value="telea",
                        label="Метод инпейнтинга",
                        info="telea — быстрый; ns (Navier-Stokes) — медленнее, иногда лучше",
                    )
                    slider_workers = gr.Slider(
                        minimum=0, maximum=8, value=0, step=1,
                        label="Потоки обработки (0 = авто)",
                        info="0 = макс. нагрузка (все потоки CPU, до 3 потоков GPU для 100% загрузки).",
                    )

                #  Launch button 
                btn_run = gr.Button(
                    " Запустить обработку",
                    variant="primary",
                    size="lg",
                    elem_classes="launch-btn",
                )

            with gr.Column(scale=1):

                gr.Markdown("### Логи и прогресс", elem_classes="section-title")
                log_output = gr.Textbox(
                    label="Лог обработки",
                    lines=22,
                    max_lines=40,
                    interactive=False,
                    placeholder="Здесь появятся логи после нажатия «Запустить обработку»...",
                    elem_classes="log-output",
                )

                gr.Markdown("### Скачать результат", elem_classes="section-title")
                result_file = gr.File(
                    label="Обработанные изображения (ZIP)",
                    interactive=False,
                )

                gr.Markdown(
                    "> **Совет:** Для максимального качества установите "
                    "[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) "
                    "(`pip install realesrgan basicsr`). "
                    "Без него используется PIL-фоллбэк."
                )

        #  Footer 
        gr.Markdown(
            "<center style='opacity:0.5; font-size:0.8rem; margin-top:20px;'>"
            "Premium Image Processor v1.0 • Для Scrollytelling Website"
            "</center>",
        )

        #  Wire up the button 
        btn_run.click(
            fn=process_wrapper,
            inputs=[
                zip_input,
                mask_input,
                chk_watermark,
                chk_upscale,
                radio_scale,
                dd_format,
                slider_quality,
                chk_gpu,
                slider_radius,
                dd_method,
                chk_redis,
                slider_workers,
            ],
            outputs=[log_output, result_file],
            show_progress="full",
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",   # Accessible from local network
        server_port=7860,
        share=False,             # Set True to get a public URL
        inbrowser=True,          # Auto-open browser
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CUSTOM_CSS,
    )
