from setuptools import setup, find_packages

setup(
    name="subtitle-generator",
    version="1.0.0",
    description="音频/视频批量转写字幕命令行工具",
    author="Subtitle Generator",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "click==8.1.7",
        "rich==13.7.1",
        "faster-whisper==1.0.3",
        "ffmpeg-python==0.2.0",
        "srt==3.5.3",
        "webvtt-py==0.5.1",
        "watchdog==4.0.0",
        "pyloudnorm==0.1.1",
        "numpy==1.26.4",
        "soundfile==0.12.1",
        "PyYAML==6.0.1",
    ],
    entry_points={
        "console_scripts": [
            "subtitle-generator=subtitle_generator.cli:cli",
        ],
    },
    package_data={
        "subtitle_generator": ["templates/*.yaml"],
    },
    include_package_data=True,
)
