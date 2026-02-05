"""
Audio conversion utilities for Meeting Upload CLI
Uses ffmpeg for format conversion
"""

import subprocess
import shutil
from pathlib import Path
from typing import Tuple, Optional

from . import config


class AudioConversionError(Exception):
    """Raised when audio conversion fails"""
    pass


def check_ffmpeg() -> bool:
    """Check if ffmpeg is installed and available"""
    return shutil.which("ffmpeg") is not None


def get_audio_info(file_path: Path) -> dict:
    """
    Get audio file information using ffprobe
    
    Returns dict with: duration, size, bitrate, format
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration,size,bit_rate,format_name",
                "-of", "json",
                str(file_path)
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        import json
        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        
        return {
            "duration": float(fmt.get("duration", 0)),
            "size": int(fmt.get("size", 0)),
            "bitrate": int(fmt.get("bit_rate", 0)) if fmt.get("bit_rate") else None,
            "format": fmt.get("format_name", "unknown")
        }
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        return {"duration": 0, "size": 0, "bitrate": None, "format": "unknown"}


def format_duration(seconds: float) -> str:
    """Format duration as HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def needs_conversion(file_path: Path) -> bool:
    """Check if the audio file needs conversion to MP3"""
    suffix = file_path.suffix.lower()
    return suffix not in config.NATIVE_FORMATS


def is_supported(file_path: Path) -> bool:
    """Check if the audio format is supported"""
    suffix = file_path.suffix.lower()
    return suffix in config.SUPPORTED_FORMATS


def convert_to_mp3(
    input_path: Path,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[callable] = None
) -> Tuple[Path, dict]:
    """
    Convert audio file to MP3 format using ffmpeg
    
    Args:
        input_path: Path to input audio file
        output_dir: Directory for output file (default: same as input)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Tuple of (output_path, info_dict)
        
    Raises:
        AudioConversionError: If conversion fails
    """
    if not check_ffmpeg():
        raise AudioConversionError(
            "ffmpeg is not installed. Install with: brew install ffmpeg"
        )
    
    if not input_path.exists():
        raise AudioConversionError(f"Input file not found: {input_path}")
    
    if not is_supported(input_path):
        raise AudioConversionError(
            f"Unsupported format: {input_path.suffix}. "
            f"Supported: {', '.join(config.SUPPORTED_FORMATS)}"
        )
    
    # Determine output path
    if output_dir is None:
        output_dir = input_path.parent
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{input_path.stem}.mp3"
    
    # Get input file info
    input_info = get_audio_info(input_path)
    
    # If already MP3, just copy
    if input_path.suffix.lower() == ".mp3":
        if input_path != output_path:
            shutil.copy2(input_path, output_path)
        return output_path, input_info
    
    # Convert using ffmpeg
    try:
        cmd = [
            "ffmpeg",
            "-i", str(input_path),
            "-vn",  # No video
            "-ar", str(config.AUDIO_SAMPLE_RATE),
            "-ac", str(config.AUDIO_CHANNELS),
            "-b:a", config.AUDIO_BITRATE,
            "-f", "mp3",
            "-y",  # Overwrite
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
    except subprocess.CalledProcessError as e:
        raise AudioConversionError(f"Conversion failed: {e.stderr}")
    
    # Get output file info
    output_info = get_audio_info(output_path)
    output_info["input_format"] = input_info.get("format", "unknown")
    output_info["input_size"] = input_info.get("size", 0)
    
    return output_path, output_info


def prepare_audio(
    input_path: Path,
    temp_dir: Optional[Path] = None
) -> Tuple[Path, dict, bool]:
    """
    Prepare audio file for upload - convert if necessary
    
    Args:
        input_path: Path to input audio file
        temp_dir: Directory for converted files
        
    Returns:
        Tuple of (ready_path, info_dict, was_converted)
    """
    input_path = Path(input_path)
    
    if not is_supported(input_path):
        raise AudioConversionError(
            f"Unsupported format: {input_path.suffix}. "
            f"Supported: {', '.join(config.SUPPORTED_FORMATS)}"
        )
    
    if needs_conversion(input_path):
        # Convert to MP3
        if temp_dir is None:
            temp_dir = input_path.parent / "converted"
        
        output_path, info = convert_to_mp3(input_path, temp_dir)
        return output_path, info, True
    else:
        # Already in native format
        info = get_audio_info(input_path)
        return input_path, info, False
