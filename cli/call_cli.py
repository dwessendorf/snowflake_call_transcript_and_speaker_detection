#!/usr/bin/env python3
"""
Call Upload CLI
Upload call recordings, monitor transcription, and export transcripts.
"""

import sys
import time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

from . import config
from . import audio
from . import snowflake_client
from . import transcript

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="call-cli")
def cli():
    """Call Upload CLI - Process call recordings with Snowflake"""
    pass


@cli.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--title", "-t", help="Call title (default: filename)")
@click.option("--no-identify", is_flag=True, help="Skip speaker identification")
@click.option("--watch", "-w", is_flag=True, help="Watch until complete and auto-export")
def upload(audio_file: str, title: Optional[str], no_identify: bool, watch: bool):
    """Upload an audio file and start transcription.
    
    Supported formats: MP3, WAV, FLAC, M4A, MP4, MOV, AAC, OGG, WMA
    
    If the format needs conversion, it will be automatically converted to MP3.
    
    Examples:
    
        call-cli upload "Weekly Sync.m4a"
        
        call-cli upload call.mp3 --title "Q4 Planning"
        
        call-cli upload call.m4a --watch
    """
    audio_path = Path(audio_file)
    
    # Default title from filename
    if title is None:
        title = audio_path.stem
    
    console.print()
    console.print(Panel(f"[bold blue]Uploading:[/] {audio_path.name}", title="Call Upload CLI"))
    console.print()
    
    # Step 1: Check and convert audio
    console.print("[bold cyan][1/4][/] Checking audio format...")
    
    if not audio.is_supported(audio_path):
        console.print(f"[red]Error:[/] Unsupported format: {audio_path.suffix}")
        console.print(f"Supported: {', '.join(config.SUPPORTED_FORMATS)}")
        sys.exit(1)
    
    try:
        if audio.needs_conversion(audio_path):
            console.print(f"      → {audio_path.suffix.upper()} detected, converting to MP3...")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                progress.add_task("Converting...", total=None)
                ready_path, info, converted = audio.prepare_audio(audio_path)
            
            size_str = audio.format_size(info.get("size", 0))
            console.print(f"      [green]✓[/] Converted: {ready_path.name} ({size_str})")
        else:
            ready_path = audio_path
            info = audio.get_audio_info(audio_path)
            console.print(f"      [green]✓[/] Format OK: {audio_path.suffix.upper()}")
        
        # Show audio info
        duration = info.get("duration", 0)
        if duration:
            console.print(f"      Duration: {audio.format_duration(duration)}")
    
    except audio.AudioConversionError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)
    
    console.print()
    
    # Step 2: Upload to Snowflake
    console.print("[bold cyan][2/4][/] Uploading to Snowflake...")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task("Uploading...", total=None)
            stage_path = snowflake_client.upload_to_stage(ready_path)
        
        console.print(f"      [green]✓[/] Uploaded to {stage_path}")
    
    except snowflake_client.SnowflakeClientError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)
    
    console.print()
    
    # Step 3: Start transcription
    console.print("[bold cyan][3/4][/] Starting transcription...")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task("Processing...", total=None)
            call_id = snowflake_client.start_transcription(stage_path, title)
        
        console.print(f"      [green]✓[/] Call created: {call_id}")
    
    except snowflake_client.SnowflakeClientError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)
    
    console.print()
    
    # Step 4: Start speaker identification
    if not no_identify:
        console.print("[bold cyan][4/4][/] Processing speakers...")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                progress.add_task("Identifying speakers...", total=None)
                result = snowflake_client.start_speaker_identification(call_id)
            
            console.print(f"      [green]✓[/] Speaker identification started")
        
        except snowflake_client.SnowflakeClientError as e:
            console.print(f"[yellow]Warning:[/] Speaker identification failed: {e}")
    else:
        console.print("[bold cyan][4/4][/] Skipping speaker identification")
    
    console.print()
    
    # Summary
    console.print(Panel.fit(
        f"[bold green]Call ID:[/] {call_id}\n"
        f"[bold]Status:[/] in_progress\n\n"
        f"Run [cyan]call-cli watch {call_id}[/] to monitor and auto-export",
        title="Upload Complete"
    ))
    
    # Watch mode
    if watch:
        console.print()
        _do_watch(call_id)


@cli.command()
@click.argument("call_id", required=False)
@click.option("--all", "-a", "show_all", is_flag=True, help="Show all calls")
@click.option("--limit", "-n", default=10, help="Number of calls to show")
def status(call_id: Optional[str], show_all: bool, limit: int):
    """Check the status of calls.
    
    Examples:
    
        call-cli status                    # Show recent calls
        
        call-cli status CALL_123           # Show specific call
        
        call-cli status --all --limit 20   # Show more calls
    """
    console.print()
    
    if call_id:
        # Show specific call status
        try:
            status_info = snowflake_client.get_call_status(call_id)
            
            if status_info.get("status") == "not_found":
                console.print(f"[red]Error:[/] Call not found: {call_id}")
                sys.exit(1)
            
            _print_call_detail(status_info)
        
        except snowflake_client.SnowflakeClientError as e:
            console.print(f"[red]Error:[/] {e}")
            sys.exit(1)
    else:
        # List recent calls
        try:
            calls = snowflake_client.list_calls(limit=limit)
            
            if not calls:
                console.print("[yellow]No calls found.[/]")
                return
            
            _print_calls_table(calls)
        
        except snowflake_client.SnowflakeClientError as e:
            console.print(f"[red]Error:[/] {e}")
            sys.exit(1)


@cli.command()
@click.argument("call_id")
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--no-timestamps", is_flag=True, help="Exclude timestamps from transcript")
def export(call_id: str, output: Optional[str], no_timestamps: bool):
    """Export a call transcript to Markdown.
    
    The transcript will be saved to ~/Documents/CallTranscripts/ by default.
    
    Examples:
    
        call-cli export CALL_123
        
        call-cli export CALL_123 --output ~/Desktop
        
        call-cli export CALL_123 --no-timestamps
    """
    console.print()
    
    # Check if call is complete
    try:
        status_info = snowflake_client.get_call_status(call_id)
        
        if status_info.get("status") == "not_found":
            console.print(f"[red]Error:[/] Call not found: {call_id}")
            sys.exit(1)
        
        total = status_info.get("total_contributions", 0)
        identified = status_info.get("identified_contributions", 0)
        
        if total > 0 and identified < total:
            console.print(f"[yellow]Warning:[/] Not all speakers identified ({identified}/{total})")
            console.print("Some speakers may appear as SPEAKER_00, SPEAKER_01, etc.")
            console.print()
    
    except snowflake_client.SnowflakeClientError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)
    
    # Export transcript
    console.print(f"[bold cyan]Exporting transcript...[/]")
    
    try:
        output_dir = Path(output) if output else None
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            progress.add_task("Generating...", total=None)
            output_path = transcript.export_call(
                call_id,
                output_dir=output_dir,
                include_timestamps=not no_timestamps
            )
        
        console.print()
        console.print(Panel.fit(
            f"[bold green]Transcript saved:[/]\n{output_path}",
            title="Export Complete"
        ))
    
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


@cli.command()
@click.argument("call_id")
@click.option("--interval", "-i", default=10, help="Poll interval in seconds")
@click.option("--output", "-o", type=click.Path(), help="Output directory for transcript")
def watch(call_id: str, interval: int, output: Optional[str]):
    """Monitor a call until complete, then export transcript.
    
    This command polls the call status and automatically exports
    the transcript when all speakers have been identified.
    
    Examples:
    
        call-cli watch CALL_123
        
        call-cli watch CALL_123 --interval 30
        
        call-cli watch CALL_123 --output ~/Desktop
    """
    console.print()
    _do_watch(call_id, interval=interval, output_dir=output)


def _do_watch(call_id: str, interval: int = 10, output_dir: Optional[str] = None):
    """Internal watch implementation"""
    console.print(f"[bold cyan]Watching call:[/] {call_id}")
    console.print(f"Polling every {interval} seconds. Press Ctrl+C to stop.")
    console.print()
    
    try:
        attempts = 0
        while attempts < config.MAX_POLL_ATTEMPTS:
            status_info = snowflake_client.get_call_status(call_id)
            
            if status_info.get("status") == "not_found":
                console.print(f"[red]Error:[/] Call not found: {call_id}")
                return
            
            total = status_info.get("total_contributions", 0)
            identified = status_info.get("identified_contributions", 0)
            status = status_info.get("status")
            
            # Calculate progress
            if total > 0:
                pct = (identified / total) * 100
                progress_bar = _progress_bar(pct)
            else:
                pct = 0
                progress_bar = _progress_bar(0)
            
            # Status line
            status_color = {
                "completed": "green",
                "in_progress": "yellow",
                "transcribing": "cyan",
                "pending": "white"
            }.get(status, "white")
            
            console.print(
                f"\r[{status_color}]{status}[/] {progress_bar} "
                f"{identified}/{total} speakers identified ({pct:.0f}%)",
                end=""
            )
            
            # Check if complete
            if status == "completed" or (total > 0 and identified == total):
                console.print()
                console.print()
                console.print("[bold green]✓ All speakers identified![/]")
                console.print()
                
                # Export transcript
                console.print("[bold cyan]Exporting transcript...[/]")
                
                output_path = transcript.export_call(
                    call_id,
                    output_dir=Path(output_dir) if output_dir else None
                )
                
                console.print()
                console.print(Panel.fit(
                    f"[bold green]Transcript saved:[/]\n{output_path}",
                    title="Complete"
                ))
                return
            
            time.sleep(interval)
            attempts += 1
        
        console.print()
        console.print("[yellow]Timeout reached. Call still in progress.[/]")
        console.print(f"Run [cyan]call-cli watch {call_id}[/] to continue monitoring.")
    
    except KeyboardInterrupt:
        console.print()
        console.print()
        console.print("[yellow]Stopped watching.[/]")
        console.print(f"Run [cyan]call-cli watch {call_id}[/] to resume.")
    
    except snowflake_client.SnowflakeClientError as e:
        console.print()
        console.print(f"[red]Error:[/] {e}")


def _progress_bar(pct: float, width: int = 20) -> str:
    """Generate a text progress bar"""
    filled = int(width * pct / 100)
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}]"


def _print_call_detail(status_info: dict):
    """Print detailed call status"""
    status = status_info.get("status", "unknown")
    status_color = {
        "completed": "green",
        "in_progress": "yellow",
        "transcribing": "cyan",
        "pending": "white"
    }.get(status, "white")
    
    total = status_info.get("total_contributions", 0)
    identified = status_info.get("identified_contributions", 0)
    
    console.print(Panel(
        f"[bold]Title:[/] {status_info.get('title', 'Unknown')}\n"
        f"[bold]Call ID:[/] {status_info.get('call_id')}\n"
        f"[bold]Date:[/] {status_info.get('call_date', 'Unknown')}\n"
        f"[bold]Status:[/] [{status_color}]{status}[/]\n"
        f"[bold]Transcription:[/] {status_info.get('transcription_status', 'Unknown')}\n"
        f"[bold]Classification:[/] {status_info.get('classification_status', 'Unknown')}\n"
        f"[bold]Progress:[/] {identified}/{total} speakers identified",
        title="Call Status"
    ))
    
    if status == "completed":
        console.print()
        console.print(f"Run [cyan]call-cli export {status_info.get('call_id')}[/] to get transcript")


def _print_calls_table(calls: list):
    """Print calls in a table format"""
    table = Table(title="Recent Calls")
    
    table.add_column("Call ID", style="cyan", no_wrap=True)
    table.add_column("Title", max_width=30)
    table.add_column("Date")
    table.add_column("Status")
    table.add_column("Progress", justify="right")
    
    for c in calls:
        status = c.get("status", "unknown")
        status_color = {
            "completed": "green",
            "in_progress": "yellow",
            "transcribing": "cyan",
            "pending": "white"
        }.get(status, "white")
        
        table.add_row(
            c.get("call_id", ""),
            c.get("title", "")[:30],
            str(c.get("call_date", ""))[:10],
            f"[{status_color}]{status}[/]",
            c.get("progress", "0%")
        )
    
    console.print(table)


@cli.command()
def config_info():
    """Show current configuration."""
    console.print()
    console.print(Panel(
        f"[bold]Connection:[/] {config.SNOWFLAKE_CONNECTION_NAME}\n"
        f"[bold]Database:[/] {config.SNOWFLAKE_DATABASE}\n"
        f"[bold]Schema:[/] {config.SNOWFLAKE_SCHEMA}\n"
        f"[bold]Warehouse:[/] {config.SNOWFLAKE_WAREHOUSE}\n"
        f"[bold]Output Dir:[/] {config.DEFAULT_OUTPUT_DIR}\n"
        f"[bold]Speaker Threshold:[/] {config.SPEAKER_CONFIDENCE_THRESHOLD}",
        title="Configuration"
    ))


def main():
    """Main entry point"""
    cli()


if __name__ == "__main__":
    main()
