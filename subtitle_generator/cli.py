import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .audio_processor import AudioDenoiser, AudioExtractor, AudioNormalizer
from .formatting import SRTFormatter, TXTFormatter, VTTFormatter
from .transcription import AudioTranscriber, TextSegmenter, WhisperModelLoader
from .utils import LanguageDetector, PunctuationFixer
from .utils.file_watcher import FileWatcher

console = Console()


def _collect_files(input_dir: Path, supported_extensions: set[str]) -> list[Path]:
    files = []
    for root, _, filenames in os.walk(input_dir):
        for fname in filenames:
            fpath = Path(root) / fname
            if fpath.suffix.lower() in supported_extensions:
                files.append(fpath)
    return sorted(files)


def _get_output_path(
    input_path: Path,
    input_dir: Path,
    output_dir: Path,
    ext: str
) -> Path:
    try:
        rel = input_path.relative_to(input_dir)
    except ValueError:
        rel = Path(input_path.name)
    return output_dir / rel.with_suffix(ext)


def _show_language_table(files_data: list[tuple[Path, str, float]]):
    table = Table(title="语言检测结果", show_lines=True)
    table.add_column("文件", style="cyan", overflow="fold")
    table.add_column("语言", style="magenta")
    table.add_column("置信度", justify="right")

    detector = LanguageDetector()
    for fpath, lang_code, prob in files_data:
        lang_name = detector.get_language_name(lang_code)
        level = detector.get_confidence_level(prob)
        prob_pct = f"{prob * 100:.1f}%"

        if level == "high":
            style = "green"
        elif level == "medium":
            style = "yellow"
        else:
            style = "red"

        table.add_row(fpath.name, lang_name, f"[{style}]{prob_pct}[/{style}]")

    console.print(table)


def _process_single_file(
    file_path: Path,
    input_dir: Path,
    output_dir: Path,
    extractor: AudioExtractor,
    denoiser: Optional[AudioDenoiser],
    normalizer: Optional[AudioNormalizer],
    transcriber: AudioTranscriber,
    segmenter: TextSegmenter,
    formatters: dict,
    punct_fixer: PunctuationFixer,
    language: Optional[str] = None,
    progress: Optional[Progress] = None,
    file_task_id=None,
    transcribe_task_id=None,
) -> tuple[str, float]:
    temp_files = []
    current_audio = None

    try:
        if progress and file_task_id is not None:
            progress.update(file_task_id, description=f"提取音轨: {file_path.name}")
        current_audio = extractor.extract(file_path)
        temp_files.append(current_audio)

        if denoiser is not None:
            if progress and file_task_id is not None:
                progress.update(file_task_id, description=f"降噪处理: {file_path.name}")
            denoised = denoiser.denoise(current_audio)
            temp_files.append(denoised)
            current_audio = denoised

        if normalizer is not None:
            if progress and file_task_id is not None:
                progress.update(file_task_id, description=f"音量归一化: {file_path.name}")
            normalized = normalizer.normalize(current_audio)
            temp_files.append(normalized)
            current_audio = normalized

        def transcribe_progress(p: float):
            if progress and transcribe_task_id is not None:
                progress.update(transcribe_task_id, completed=p * 100)

        if progress and file_task_id is not None:
            progress.update(file_task_id, description=f"转写中: {file_path.name}")

        result = transcriber.transcribe(current_audio, language=language, progress_callback=transcribe_progress)

        if progress and file_task_id is not None:
            progress.update(file_task_id, description=f"分段处理: {file_path.name}")

        for seg in result.segments:
            seg.text = punct_fixer.fix(seg.text)

        segmented = segmenter.segment(result.segments)

        if progress and file_task_id is not None:
            progress.update(file_task_id, description=f"生成字幕: {file_path.name}")

        for fmt_name, fmt_output_ext, formatter in formatters:
            out_path = _get_output_path(file_path, input_dir, output_dir, fmt_output_ext)
            formatter.format(segmented, out_path)

        return result.language, result.language_probability

    finally:
        for tf in temp_files:
            try:
                if tf and os.path.exists(tf) and not tf.endswith(str(file_path)):
                    os.remove(tf)
            except OSError:
                pass


@click.group()
@click.version_option(package_name="subtitle-generator", prog_name="subtitle-generator")
def cli():
    """音频/视频批量转写字幕工具 - 使用 Whisper 模型自动生成字幕"""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", "output_dir", type=click.Path(file_okay=False, path_type=Path), default="./subtitles", help="输出目录")
@click.option("--format", "-f", "formats", multiple=True, type=click.Choice(["srt", "vtt", "txt"]), default=["srt", "vtt", "txt"], help="输出格式（可多选）")
@click.option("--model", "-m", type=click.Choice(["tiny", "base", "small", "medium", "large-v3"]), default="base", help="Whisper 模型大小")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", help="计算设备")
@click.option("--compute-type", type=click.Choice(["auto", "int8", "int8_float32", "int8_float16", "int16", "float16", "float32"]), default="auto", help="计算精度")
@click.option("--language", "-l", type=str, default=None, help="强制指定语言（如 zh, en, ja），默认自动检测")
@click.option("--denoise/--no-denoise", default=True, help="是否启用音频降噪")
@click.option("--denoise-level", type=click.FloatRange(0.0, 1.0), default=0.5, help="降噪强度 0.0-1.0")
@click.option("--normalize/--no-normalize", default=True, help="是否启用音量归一化")
@click.option("--target-lufs", type=float, default=-16.0, help="归一化目标 LUFS 值")
@click.option("--beam-size", type=int, default=5, help="Beam search 大小")
@click.option("--vad-filter/--no-vad-filter", default=True, help="启用 VAD 语音活动检测")
@click.option("--txt-timestamp/--no-txt-timestamp", default=True, help="TXT 输出是否带时间戳")
@click.option("--max-chinese-chars", type=int, default=50, help="单行最大中文字符数")
@click.option("--max-english-chars", type=int, default=80, help="单行最大英文字符数")
def generate(
    input_file: Path,
    output_dir: Path,
    formats: list[str],
    model: str,
    device: str,
    compute_type: str,
    language: Optional[str],
    denoise: bool,
    denoise_level: float,
    normalize: bool,
    target_lufs: float,
    beam_size: int,
    vad_filter: bool,
    txt_timestamp: bool,
    max_chinese_chars: int,
    max_english_chars: int,
):
    """转写单个音视频文件"""
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]加载 Whisper {model} 模型...[/bold cyan]")
    model_loader = WhisperModelLoader(model_size=model, device=device, compute_type=compute_type)

    fmt_map = {
        "srt": (".srt", SRTFormatter()),
        "vtt": (".vtt", VTTFormatter()),
        "txt": (".txt", TXTFormatter(with_timestamp=txt_timestamp)),
    }
    active_formatters = [(f, *fmt_map[f]) for f in formats]

    extractor = AudioExtractor()
    denoiser = AudioDenoiser(level=denoise_level) if denoise else None
    normalizer = AudioNormalizer(target_lufs=target_lufs) if normalize else None
    transcriber = AudioTranscriber(model_loader=model_loader, beam_size=beam_size, vad_filter=vad_filter)
    segmenter = TextSegmenter(max_chinese_chars=max_chinese_chars, max_english_chars=max_english_chars)
    punct_fixer = PunctuationFixer()

    input_dir = input_file.parent

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        file_task = progress.add_task(f"处理: {input_file.name}", total=100)
        transcribe_task = progress.add_task("转写进度", total=100)

        lang_code, lang_prob = _process_single_file(
            file_path=input_file,
            input_dir=input_dir,
            output_dir=output_dir,
            extractor=extractor,
            denoiser=denoiser,
            normalizer=normalizer,
            transcriber=transcriber,
            segmenter=segmenter,
            formatters=active_formatters,
            punct_fixer=punct_fixer,
            language=language,
            progress=progress,
            file_task_id=file_task,
            transcribe_task_id=transcribe_task,
        )

        progress.update(file_task, completed=100, description=f"[green]完成: {input_file.name}[/green]")
        progress.update(transcribe_task, completed=100, visible=False)

    _show_language_table([(input_file, lang_code, lang_prob)])

    out_names = ", ".join(active_formatters[0][0] if len(formats) == 1 else ",".join(formats))
    console.print(f"[bold green]字幕已生成到: {output_dir}[/bold green]")


@cli.command()
@click.option("--input", "-i", "input_dir", type=click.Path(file_okay=False, path_type=Path), required=True, help="输入目录")
@click.option("--output", "-o", "output_dir", type=click.Path(file_okay=False, path_type=Path), default="./subtitles", help="输出目录")
@click.option("--format", "-f", "formats", multiple=True, type=click.Choice(["srt", "vtt", "txt"]), default=["srt", "vtt", "txt"], help="输出格式（可多选）")
@click.option("--model", "-m", type=click.Choice(["tiny", "base", "small", "medium", "large-v3"]), default="base", help="Whisper 模型大小")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="auto", help="计算设备")
@click.option("--compute-type", type=click.Choice(["auto", "int8", "int8_float32", "int8_float16", "int16", "float16", "float32"]), default="auto", help="计算精度")
@click.option("--language", "-l", type=str, default=None, help="强制指定语言（如 zh, en, ja）")
@click.option("--denoise/--no-denoise", default=True, help="是否启用音频降噪")
@click.option("--denoise-level", type=click.FloatRange(0.0, 1.0), default=0.5, help="降噪强度 0.0-1.0")
@click.option("--normalize/--no-normalize", default=True, help="是否启用音量归一化")
@click.option("--target-lufs", type=float, default=-16.0, help="归一化目标 LUFS 值")
@click.option("--beam-size", type=int, default=5, help="Beam search 大小")
@click.option("--vad-filter/--no-vad-filter", default=True, help="启用 VAD 语音活动检测")
@click.option("--txt-timestamp/--no-txt-timestamp", default=True, help="TXT 输出是否带时间戳")
@click.option("--max-chinese-chars", type=int, default=50, help="单行最大中文字符数")
@click.option("--max-english-chars", type=int, default=80, help="单行最大英文字符数")
@click.option("--watch/--no-watch", default=False, help="监听目录，自动处理新文件")
@click.option("--config", "-c", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help="YAML 配置文件（覆盖 CLI 参数）")
def batch(
    input_dir: Path,
    output_dir: Path,
    formats: list[str],
    model: str,
    device: str,
    compute_type: str,
    language: Optional[str],
    denoise: bool,
    denoise_level: float,
    normalize: bool,
    target_lufs: float,
    beam_size: int,
    vad_filter: bool,
    txt_timestamp: bool,
    max_chinese_chars: int,
    max_english_chars: int,
    watch: bool,
    config: Optional[Path],
):
    """批量转写目录中的音视频文件"""
    if config is not None:
        try:
            cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
            if cfg:
                m_cfg = cfg.get("model", {})
                p_cfg = cfg.get("preprocessing", {})
                t_cfg = cfg.get("transcription", {})
                f_cfg = cfg.get("formatting", {})
                w_cfg = cfg.get("watcher", {})

                input_dir = Path(cfg.get("input_directory", str(input_dir)))
                output_dir = Path(cfg.get("output_directory", str(output_dir)))
                model = m_cfg.get("size", model)
                device = m_cfg.get("device", device)
                compute_type = m_cfg.get("compute_type", compute_type)
                denoise = p_cfg.get("denoise", denoise)
                denoise_level = p_cfg.get("denoise_level", denoise_level)
                normalize = p_cfg.get("normalize", normalize)
                target_lufs = p_cfg.get("target_lufs", target_lufs)
                language = t_cfg.get("language", language)
                beam_size = t_cfg.get("beam_size", beam_size)
                vad_filter = t_cfg.get("vad_filter", vad_filter)
                cfg_formats = f_cfg.get("output_formats", None)
                if cfg_formats:
                    formats = tuple(cfg_formats)
                txt_timestamp = f_cfg.get("txt_with_timestamp", txt_timestamp)
                max_chinese_chars = f_cfg.get("max_chinese_chars", max_chinese_chars)
                max_english_chars = f_cfg.get("max_english_chars", max_english_chars)
                watch = w_cfg.get("enabled", watch)
        except Exception as e:
            console.print(f"[bold red]配置文件读取失败: {e}[/bold red]")
            sys.exit(1)

    if not input_dir.exists():
        console.print(f"[bold red]输入目录不存在: {input_dir}[/bold red]")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold cyan]加载 Whisper {model} 模型...[/bold cyan]")
    model_loader = WhisperModelLoader(model_size=model, device=device, compute_type=compute_type)

    fmt_map = {
        "srt": (".srt", SRTFormatter()),
        "vtt": (".vtt", VTTFormatter()),
        "txt": (".txt", TXTFormatter(with_timestamp=txt_timestamp)),
    }
    active_formatters = [(f, *fmt_map[f]) for f in formats]

    extractor = AudioExtractor()
    denoiser = AudioDenoiser(level=denoise_level) if denoise else None
    normalizer = AudioNormalizer(target_lufs=target_lufs) if normalize else None
    transcriber = AudioTranscriber(model_loader=model_loader, beam_size=beam_size, vad_filter=vad_filter)
    segmenter = TextSegmenter(max_chinese_chars=max_chinese_chars, max_english_chars=max_english_chars)
    punct_fixer = PunctuationFixer()

    language_results: list[tuple[Path, str, float]] = []

    def do_process(file_path: Path):
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            file_task = progress.add_task(f"处理: {file_path.name}", total=100)
            transcribe_task = progress.add_task("转写进度", total=100)

            lang_code, lang_prob = _process_single_file(
                file_path=file_path,
                input_dir=input_dir,
                output_dir=output_dir,
                extractor=extractor,
                denoiser=denoiser,
                normalizer=normalizer,
                transcriber=transcriber,
                segmenter=segmenter,
                formatters=active_formatters,
                punct_fixer=punct_fixer,
                language=language,
                progress=progress,
                file_task_id=file_task,
                transcribe_task_id=transcribe_task,
            )

            progress.update(file_task, completed=100, description=f"[green]完成: {file_path.name}[/green]")
            progress.update(transcribe_task, completed=100, visible=False)

        language_results.append((file_path, lang_code, lang_prob))
        console.print(f"[green]✓ {file_path.name}[/green]")

    files = _collect_files(input_dir, extractor.SUPPORTED_EXTENSIONS)

    if not files and not watch:
        console.print(f"[yellow]未在 {input_dir} 中找到支持的音视频文件[/yellow]")
        sys.exit(0)

    if files:
        console.print(f"[bold]找到 {len(files)} 个文件待处理[/bold]")

    for idx, fpath in enumerate(files, start=1):
        console.print(f"\n[bold cyan]处理 [{idx}/{len(files)}]: {fpath.name}[/bold cyan]")
        do_process(fpath)

    if language_results:
        console.print()
        _show_language_table(language_results)

    console.print(f"\n[bold green]所有字幕已生成到: {output_dir}[/bold green]")

    if watch:
        console.print(f"\n[bold yellow]正在监听目录: {input_dir}（Ctrl+C 停止）[/bold yellow]")

        def watch_callback(file_path_str: str):
            fpath = Path(file_path_str)
            console.print(f"\n[bold cyan]检测到新文件: {fpath.name}[/bold cyan]")
            try:
                do_process(fpath)
                _show_language_table([language_results[-1]] if language_results else [])
            except Exception as e:
                console.print(f"[bold red]处理失败 {fpath.name}: {e}[/bold red]")

        watcher = FileWatcher(
            directory=input_dir,
            callback=watch_callback,
            supported_extensions=extractor.SUPPORTED_EXTENSIONS,
        )
        try:
            watcher.start(scan_existing=False)
            watcher.wait()
        except KeyboardInterrupt:
            console.print("\n[yellow]已停止监听[/yellow]")
        finally:
            watcher.stop()


@cli.command("format")
@click.argument("subtitle_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), default=None, help="输出文件路径")
@click.option("--to", "target_format", type=click.Choice(["srt", "vtt", "txt"]), required=True, help="目标格式")
@click.option("--txt-timestamp/--no-txt-timestamp", default=True, help="TXT 输出是否带时间戳")
@click.option("--max-chinese-chars", type=int, default=50, help="单行最大中文字符数")
@click.option("--max-english-chars", type=int, default=80, help="单行最大英文字符数")
def format_cmd(
    subtitle_file: Path,
    output: Optional[Path],
    target_format: str,
    txt_timestamp: bool,
    max_chinese_chars: int,
    max_english_chars: int,
):
    """转换现有字幕文件格式（SRT/VTT/TXT 互转）"""
    try:
        import srt as srt_lib
    except ImportError:
        console.print("[bold red]缺少 srt 库，请安装: pip install srt[/bold red]")
        sys.exit(1)

    segments = []
    src_ext = subtitle_file.suffix.lower()

    try:
        if src_ext == ".srt":
            with open(subtitle_file, encoding="utf-8") as f:
                subs = list(srt_lib.parse(f.read()))
            for sub in subs:
                segments.append(type("Seg", (), {
                    "start": sub.start.total_seconds(),
                    "end": sub.end.total_seconds(),
                    "text": sub.content,
                })())
        elif src_ext == ".vtt":
            try:
                import webvtt
                vtt = webvtt.read(str(subtitle_file))
                for caption in vtt:
                    def parse_ts(ts: str) -> float:
                        parts = ts.replace(",", ".").split(":")
                        if len(parts) == 3:
                            h, m, s = parts
                        elif len(parts) == 2:
                            h = 0
                            m, s = parts
                        else:
                            return 0.0
                        return float(h) * 3600 + float(m) * 60 + float(s)
                    segments.append(type("Seg", (), {
                        "start": parse_ts(caption.start),
                        "end": parse_ts(caption.end),
                        "text": caption.text,
                    })())
            except ImportError:
                console.print("[bold red]缺少 webvtt-py 库，请安装: pip install webvtt-py[/bold red]")
                sys.exit(1)
        elif src_ext == ".txt":
            lines = subtitle_file.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                text = line.strip()
                if not text:
                    continue
                segments.append(type("Seg", (), {
                    "start": float(i) * 3.0,
                    "end": float(i) * 3.0 + 2.5,
                    "text": text,
                })())
        else:
            console.print(f"[bold red]不支持的输入格式: {src_ext}[/bold red]")
            sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]读取字幕文件失败: {e}[/bold red]")
        sys.exit(1)

    segmenter = TextSegmenter(max_chinese_chars=max_chinese_chars, max_english_chars=max_english_chars)
    segmented = segmenter.segment(segments)

    if output is None:
        output = subtitle_file.with_suffix(f".{target_format}")

    output.parent.mkdir(parents=True, exist_ok=True)

    if target_format == "srt":
        formatter = SRTFormatter()
    elif target_format == "vtt":
        formatter = VTTFormatter()
    else:
        formatter = TXTFormatter(with_timestamp=txt_timestamp)

    formatter.format(segmented, output)
    console.print(f"[bold green]已转换: {output}[/bold green]")


if __name__ == "__main__":
    cli()
