"""
Transcript generation for Call Upload CLI
Generates formatted Markdown transcripts from call data
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

try:
    from . import config
except ImportError:
    import config


def format_timestamp(seconds: float) -> str:
    """Format seconds as [HH:MM:SS] timestamp"""
    if seconds is None:
        return "[00:00:00]"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def format_date(date_value: Any) -> str:
    """Format date for display"""
    if date_value is None:
        return "Unknown"
    
    if isinstance(date_value, str):
        # Handle quoted date strings from Snowflake
        date_value = date_value.strip('"')
        try:
            dt = datetime.strptime(date_value, "%Y-%m-%d")
            return dt.strftime("%B %d, %Y")
        except ValueError:
            return date_value
    
    if isinstance(date_value, datetime):
        return date_value.strftime("%B %d, %Y")
    
    return str(date_value)


def format_duration(minutes: Optional[float]) -> str:
    """Format duration in human-readable format"""
    if minutes is None:
        return "Unknown"
    
    if minutes < 60:
        return f"{int(minutes)} minutes"
    
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    
    if mins == 0:
        return f"{hours} hour{'s' if hours > 1 else ''}"
    
    return f"{hours} hour{'s' if hours > 1 else ''} {mins} minutes"


def generate_markdown(
    call_info: Dict[str, Any],
    contributions: List[Dict[str, Any]],
    include_timestamps: bool = True,
    include_summary: bool = True
) -> str:
    """
    Generate a formatted Markdown transcript
    
    Args:
        call_info: Call metadata dict
        contributions: List of contribution dicts with speaker info
        include_timestamps: Whether to include timestamps
        include_summary: Whether to include AI summary if available
        
    Returns:
        Formatted Markdown string
    """
    lines = []
    
    # Title
    title = call_info.get("title", "Call Transcript")
    call_date = format_date(call_info.get("call_date"))
    
    lines.append(f"# {title}")
    lines.append("")
    
    # Metadata
    lines.append(f"**Date:** {call_date}")
    
    duration = call_info.get("duration_minutes")
    if duration:
        lines.append(f"**Duration:** {format_duration(duration)}")
    
    speakers = call_info.get("speakers", [])
    if speakers:
        # Filter out generic speaker labels
        named_speakers = [s for s in speakers if s and not s.startswith("SPEAKER_")]
        if named_speakers:
            lines.append(f"**Speakers:** {', '.join(sorted(named_speakers))}")
    
    language = call_info.get("language")
    if language:
        lines.append(f"**Language:** {language}")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Summary (if available)
    summary = call_info.get("summary")
    if include_summary and summary:
        lines.append("## Summary")
        lines.append("")
        lines.append(summary)
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Transcript
    lines.append("## Transcript")
    lines.append("")
    
    current_speaker = None
    
    for contrib in contributions:
        speaker = contrib.get("SPEAKER_NAME", "Unknown")
        text = contrib.get("TEXT_CONTENT", "")
        start_time = contrib.get("START_TIME_SECONDS")
        
        if not text or not text.strip():
            continue
        
        # Format timestamp
        timestamp = format_timestamp(start_time) if include_timestamps else ""
        
        # Add speaker header if changed
        if speaker != current_speaker:
            current_speaker = speaker
            if include_timestamps:
                lines.append(f"**{timestamp} {speaker}:**")
            else:
                lines.append(f"**{speaker}:**")
        else:
            # Same speaker, just add timestamp if different segment
            if include_timestamps and start_time:
                lines.append(f"*{timestamp}*")
        
        # Add text content
        lines.append(text.strip())
        lines.append("")
    
    # Footer
    lines.append("---")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append(f"*Call ID: {call_info.get('call_id', 'Unknown')}*")
    
    return "\n".join(lines)


def generate_filename(call_info: Dict[str, Any]) -> str:
    """Generate a filename for the transcript"""
    title = call_info.get("title", "call")
    date = call_info.get("call_date")
    
    # Clean title for filename
    import re
    clean_title = re.sub(r'[^\w\s-]', '', title)
    clean_title = re.sub(r'\s+', '_', clean_title)
    clean_title = clean_title[:50]  # Limit length
    
    # Format date
    if date:
        if isinstance(date, str):
            date = date.strip('"')
        else:
            date = date.strftime("%Y-%m-%d")
    else:
        date = datetime.now().strftime("%Y-%m-%d")
    
    return f"{date}_{clean_title}.md"


def save_transcript(
    call_info: Dict[str, Any],
    contributions: List[Dict[str, Any]],
    output_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    include_timestamps: bool = True
) -> Path:
    """
    Generate and save a Markdown transcript
    
    Args:
        call_info: Call metadata dict
        contributions: List of contribution dicts
        output_dir: Output directory (default: config.DEFAULT_OUTPUT_DIR)
        filename: Output filename (default: auto-generated)
        include_timestamps: Whether to include timestamps
        
    Returns:
        Path to saved file
    """
    # Determine output directory
    if output_dir is None:
        output_dir = config.DEFAULT_OUTPUT_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    if filename is None:
        filename = generate_filename(call_info)
    
    output_path = output_dir / filename
    
    # Generate markdown
    markdown = generate_markdown(
        call_info,
        contributions,
        include_timestamps=include_timestamps
    )
    
    # Save file
    output_path.write_text(markdown, encoding="utf-8")
    
    return output_path


def export_call(
    call_id: str,
    output_dir: Optional[Path] = None,
    include_timestamps: bool = True
) -> Path:
    """
    Export a call transcript to Markdown
    
    Args:
        call_id: Call ID to export
        output_dir: Output directory
        include_timestamps: Whether to include timestamps
        
    Returns:
        Path to saved file
    """
    try:
        from . import snowflake_client
    except ImportError:
        import snowflake_client
    
    # Get transcript data
    call_info, contributions = snowflake_client.get_transcript(call_id)
    
    # Save transcript
    return save_transcript(
        call_info,
        contributions,
        output_dir=output_dir,
        include_timestamps=include_timestamps
    )
