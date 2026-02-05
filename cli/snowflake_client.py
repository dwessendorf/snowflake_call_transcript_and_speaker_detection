"""
Snowflake client for Meeting Upload CLI
Handles uploads, transcription, and status queries
"""

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import snowflake.connector
from snowflake.connector import DictCursor

from . import config


class SnowflakeClientError(Exception):
    """Raised when Snowflake operations fail"""
    pass


class MeetingStatus:
    """Meeting processing status"""
    PENDING = "pending"
    TRANSCRIBING = "transcribing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


def get_connection() -> snowflake.connector.SnowflakeConnection:
    """Create Snowflake connection using key-based auth or named connection"""
    from pathlib import Path
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    import os
    
    # Disable OCSP certificate checking for S3 transfers
    os.environ['SF_OCSP_RESPONSE_CACHE_SERVER_ENABLED'] = 'false'
    
    # Try key-based auth first
    key_path = Path(config.SNOWFLAKE_PRIVATE_KEY_PATH)
    if key_path.exists():
        try:
            with open(key_path, "rb") as key_file:
                p_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,
                    backend=default_backend()
                )
            
            pkb = p_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            conn = snowflake.connector.connect(
                account=config.SNOWFLAKE_ACCOUNT,
                user=config.SNOWFLAKE_USER,
                private_key=pkb,
                database=config.SNOWFLAKE_DATABASE,
                schema=config.SNOWFLAKE_SCHEMA,
                warehouse=config.SNOWFLAKE_WAREHOUSE,
                insecure_mode=True  # Disable OCSP for file transfers
            )
            return conn
        except Exception as e:
            pass  # Fall back to named connection
    
    # Fallback to named connection
    try:
        conn = snowflake.connector.connect(
            connection_name=config.SNOWFLAKE_CONNECTION_NAME,
            ocsp_response_cache_filename="/tmp/ocsp_cache",
            insecure_mode=True
        )
        return conn
    except Exception as e:
        raise SnowflakeClientError(f"Failed to connect to Snowflake: {e}")


def generate_meeting_id(title: str) -> str:
    """Generate a unique meeting ID from title"""
    # Clean title for ID - keep shorter to fit contribution IDs
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    clean = re.sub(r'\s+', '_', clean).upper()[:18]  # Shorter to allow contrib suffix
    
    # Add date and short UUID
    date_str = datetime.now().strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:6].upper()
    
    return f"{config.MEETING_ID_PREFIX}_{date_str}_{clean}_{short_uuid}"


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for stage upload"""
    # Replace spaces with underscores, remove special chars
    clean = re.sub(r'[^\w\-.]', '_', filename)
    clean = re.sub(r'_+', '_', clean)
    return clean.lower()


def upload_to_stage(
    local_path: Path,
    stage_name: str = None,
    auto_compress: bool = False
) -> str:
    """
    Upload a file to Snowflake stage
    
    Args:
        local_path: Path to local file
        stage_name: Target stage (default: MEETING_RECORDINGS)
        auto_compress: Whether to auto-compress the file
        
    Returns:
        Stage path of uploaded file
    """
    if stage_name is None:
        stage_name = config.STAGE_RECORDINGS
    
    local_path = Path(local_path)
    if not local_path.exists():
        raise SnowflakeClientError(f"File not found: {local_path}")
    
    # Sanitize the filename
    stage_filename = sanitize_filename(local_path.name)
    
    # Use direct connection with key-based auth for uploads
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Upload file - PUT directly to stage, Snowflake will use the original filename
        # Don't append stage_filename to avoid nested directories
        put_cmd = f"PUT 'file://{local_path}' {stage_name}/"
        if not auto_compress:
            put_cmd += " AUTO_COMPRESS=FALSE"
        
        cursor.execute(put_cmd)
        result = cursor.fetchall()
        
        # Check upload status - result contains the actual uploaded filename
        if result and len(result[0]) > 6 and result[0][6] in ('UPLOADED', 'SKIPPED'):
            # Use the TARGET column which has the actual uploaded filename
            actual_filename = result[0][1] if len(result[0]) > 1 else stage_filename
            return f"{stage_name}/{actual_filename}"
        elif result and 'UPLOADED' in str(result):
            return f"{stage_name}/{stage_filename}"
        else:
            raise SnowflakeClientError(f"Upload failed: {result}")
            
    finally:
        conn.close()


def _upload_via_snowsql(local_path: Path, stage_name: str, stage_filename: str, auto_compress: bool) -> str:
    """Upload file using Snowflake CLI (bypasses Python SSL issues)"""
    import subprocess
    import shutil
    
    # Try Snowflake CLI (snow) first
    if shutil.which("snow"):
        # Build stage path without @ for snow CLI
        stage_target = stage_name.lstrip("@") + "/" + stage_filename
        
        result = subprocess.run(
            [
                "snow", "stage", "copy",
                str(local_path),
                f"@{stage_target}",
                "--connection", config.SNOWFLAKE_CONNECTION_NAME,
                "--overwrite"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            return f"{stage_name}/{stage_filename}"
        
        # Check stderr for success indicators
        if "copied" in result.stdout.lower() or "copied" in result.stderr.lower():
            return f"{stage_name}/{stage_filename}"
        
        raise SnowflakeClientError(f"Upload failed: {result.stderr or result.stdout}")
    
    # Fallback to snowsql
    if shutil.which("snowsql"):
        put_cmd = f"PUT 'file://{local_path}' {stage_name}/{stage_filename}"
        if not auto_compress:
            put_cmd += " AUTO_COMPRESS=FALSE"
        
        result = subprocess.run(
            [
                "snowsql",
                "-c", config.SNOWFLAKE_CONNECTION_NAME,
                "-d", config.SNOWFLAKE_DATABASE,
                "-s", config.SNOWFLAKE_SCHEMA,
                "-w", config.SNOWFLAKE_WAREHOUSE,
                "-q", put_cmd,
                "-o", "output_format=plain",
                "-o", "friendly=false"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            raise SnowflakeClientError(f"SnowSQL upload failed: {result.stderr}")
        
        if "UPLOADED" in result.stdout or "SKIPPED" in result.stdout:
            return f"{stage_name}/{stage_filename}"
        
        raise SnowflakeClientError(f"Upload failed: {result.stdout}")
    
    raise SnowflakeClientError("Neither 'snow' nor 'snowsql' CLI available for uploads")


def start_transcription(
    stage_path: str,
    meeting_title: str,
    meeting_date: Optional[datetime] = None
) -> str:
    """
    Start transcription for an uploaded audio file
    
    Args:
        stage_path: Path to file in stage
        meeting_title: Title for the meeting
        meeting_date: Date of meeting (default: today)
        
    Returns:
        Meeting ID
    """
    if meeting_date is None:
        meeting_date = datetime.now()
    
    meeting_id = generate_meeting_id(meeting_title)
    
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Set context
        cursor.execute(f"USE DATABASE {config.SNOWFLAKE_DATABASE}")
        cursor.execute(f"USE SCHEMA {config.SNOWFLAKE_SCHEMA}")
        cursor.execute(f"USE WAREHOUSE {config.SNOWFLAKE_WAREHOUSE}")
        
        # Call transcription procedure
        cursor.execute(f"CALL TRANSCRIBE_MEETING('{stage_path}')")
        result = cursor.fetchone()
        
        if result:
            import json
            try:
                data = json.loads(result[0]) if isinstance(result[0], str) else result[0]
                if data.get("status") == "success":
                    # Get transcription details
                    transcription = data.get("transcription", "")
                    num_speakers = data.get("num_speakers", 0)
                    duration_mins = data.get("estimated_duration_minutes", 0)
                    
                    # Save to MEETINGS table
                    insert_sql = f"""
                    INSERT INTO MEETINGS (
                        MEETING_ID, TITLE, MEETING_DATE, RECORDING_PATH, 
                        TRANSCRIPTION_STATUS, CLASSIFICATION_STATUS,
                        TOTAL_SPEAKERS, DURATION_MINUTES, CREATED_AT
                    ) VALUES (
                        '{meeting_id}', 
                        '{meeting_title.replace("'", "''")}', 
                        '{meeting_date.strftime("%Y-%m-%d")}',
                        '{stage_path}',
                        'completed',
                        'pending',
                        {num_speakers},
                        {int(duration_mins)},
                        CURRENT_TIMESTAMP
                    )
                    """
                    cursor.execute(insert_sql)
                    
                    # Save transcription to TRANSCRIPTIONS stage
                    transcription_path = f"{config.STAGE_TRANSCRIPTIONS}/{meeting_id}_transcript.json"
                    
                    # Parse and save contributions - merge consecutive segments from same speaker
                    if transcription:
                        trans_data = json.loads(transcription) if isinstance(transcription, str) else transcription
                        segments = trans_data.get("segments", [])
                        
                        # Merge consecutive segments from the same speaker
                        merged_segments = []
                        current_segment = None
                        
                        for seg in segments:
                            speaker = seg.get("speaker_label", "UNKNOWN")
                            
                            if current_segment is None:
                                # Start a new segment
                                current_segment = {
                                    "speaker_label": speaker,
                                    "start": seg.get("start", 0),
                                    "end": seg.get("end", 0),
                                    "text": seg.get("text", "")
                                }
                            elif speaker == current_segment["speaker_label"]:
                                # Same speaker - merge with current segment
                                current_segment["end"] = seg.get("end", current_segment["end"])
                                current_segment["text"] += " " + seg.get("text", "")
                            else:
                                # Different speaker - save current and start new
                                merged_segments.append(current_segment)
                                current_segment = {
                                    "speaker_label": speaker,
                                    "start": seg.get("start", 0),
                                    "end": seg.get("end", 0),
                                    "text": seg.get("text", "")
                                }
                        
                        # Don't forget the last segment
                        if current_segment:
                            merged_segments.append(current_segment)
                        
                        # Insert merged segments as contributions
                        for idx, seg in enumerate(merged_segments):
                            contrib_id = f"{meeting_id}_{idx:04d}"
                            start_time = seg.get("start", 0)
                            end_time = seg.get("end", 0)
                            duration = end_time - start_time if end_time > start_time else 0
                            text = seg.get("text", "").strip()
                            word_count = len(text.split())
                            
                            insert_contrib = f"""
                            INSERT INTO MEETING_CONTRIBUTIONS (
                                CONTRIBUTION_ID, MEETING_ID, SEGMENT_NUMBER,
                                DIARIZATION_LABEL, TEXT_CONTENT,
                                START_TIME_SECONDS, END_TIME_SECONDS, 
                                DURATION_SECONDS, WORD_COUNT,
                                CLASSIFICATION_STATUS, CREATED_AT
                            ) VALUES (
                                '{contrib_id}',
                                '{meeting_id}',
                                {idx},
                                '{seg.get("speaker_label", "UNKNOWN")}',
                                '{text.replace("'", "''")}',
                                {start_time},
                                {end_time},
                                {duration},
                                {word_count},
                                'pending',
                                CURRENT_TIMESTAMP
                            )
                            """
                            cursor.execute(insert_contrib)
                    
                    return meeting_id
                else:
                    raise SnowflakeClientError(f"Transcription failed: {data.get('message')}")
            except json.JSONDecodeError:
                # Result might be the meeting ID directly
                return str(result[0]) if result[0] else meeting_id
        
        return meeting_id
        
    finally:
        conn.close()


def start_speaker_identification(meeting_id: str, threshold: float = None) -> Dict[str, Any]:
    """
    Start speaker identification for a meeting
    
    Args:
        meeting_id: Meeting ID to process
        threshold: Confidence threshold for auto-matching
        
    Returns:
        Processing result dict
    """
    if threshold is None:
        threshold = config.SPEAKER_CONFIDENCE_THRESHOLD
    
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Set context
        cursor.execute(f"USE DATABASE {config.SNOWFLAKE_DATABASE}")
        cursor.execute(f"USE SCHEMA {config.SNOWFLAKE_SCHEMA}")
        cursor.execute(f"USE WAREHOUSE {config.SNOWFLAKE_WAREHOUSE}")
        
        # Call speaker identification procedure (groups by diarization, queues for review)
        cursor.execute(f"CALL IDENTIFY_MEETING_SPEAKERS_WITH_EMBEDDINGS('{meeting_id}', {threshold})")
        result = cursor.fetchone()
        
        identification_result = {"status": "started", "meeting_id": meeting_id}
        if result:
            import json
            try:
                identification_result = json.loads(result[0]) if isinstance(result[0], str) else result[0]
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Run automatic speaker detection using voiceprints
        try:
            cursor.execute(f"CALL AUTO_DETECT_SPEAKERS('{meeting_id}', {threshold})")
            auto_result = cursor.fetchone()
            if auto_result:
                import json
                try:
                    auto_data = json.loads(auto_result[0]) if isinstance(auto_result[0], str) else auto_result[0]
                    identification_result["auto_detection"] = auto_data
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception as e:
            # Auto-detection is optional, don't fail if it errors
            identification_result["auto_detection_error"] = str(e)
        
        return identification_result
        
    finally:
        conn.close()


def get_meeting_status(meeting_id: str) -> Dict[str, Any]:
    """
    Get the current status of a meeting
    
    Returns dict with: status, title, total_contributions, identified_contributions, etc.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor(DictCursor)
        
        # Set context
        cursor.execute(f"USE DATABASE {config.SNOWFLAKE_DATABASE}")
        cursor.execute(f"USE SCHEMA {config.SNOWFLAKE_SCHEMA}")
        
        # Get meeting info
        cursor.execute(f"""
            SELECT 
                m.MEETING_ID,
                m.TITLE,
                m.MEETING_DATE,
                m.TRANSCRIPTION_STATUS,
                m.CLASSIFICATION_STATUS,
                m.TOTAL_SPEAKERS,
                m.LANGUAGE,
                m.DURATION_MINUTES
            FROM MEETINGS m
            WHERE m.MEETING_ID = '{meeting_id}'
        """)
        meeting = cursor.fetchone()
        
        if not meeting:
            return {"status": "not_found", "meeting_id": meeting_id}
        
        # Get contribution counts
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total,
                COUNT(IDENTIFIED_SPEAKER_ID) as identified,
                COUNT(CASE WHEN CLASSIFICATION_STATUS = 'pending' THEN 1 END) as pending
            FROM MEETING_CONTRIBUTIONS
            WHERE MEETING_ID = '{meeting_id}'
        """)
        counts = cursor.fetchone()
        
        # Determine overall status
        if meeting.get("TRANSCRIPTION_STATUS") == "pending":
            status = MeetingStatus.TRANSCRIBING
        elif counts and counts["TOTAL"] > 0 and counts["IDENTIFIED"] == counts["TOTAL"]:
            status = MeetingStatus.COMPLETED
        elif meeting.get("CLASSIFICATION_STATUS") == "completed":
            status = MeetingStatus.COMPLETED
        elif counts and counts["IDENTIFIED"] > 0:
            status = MeetingStatus.IN_PROGRESS
        else:
            status = MeetingStatus.PENDING
        
        return {
            "status": status,
            "meeting_id": meeting_id,
            "title": meeting.get("TITLE"),
            "meeting_date": meeting.get("MEETING_DATE"),
            "transcription_status": meeting.get("TRANSCRIPTION_STATUS"),
            "classification_status": meeting.get("CLASSIFICATION_STATUS"),
            "total_speakers": meeting.get("TOTAL_SPEAKERS"),
            "language": meeting.get("LANGUAGE"),
            "duration_minutes": meeting.get("DURATION_MINUTES"),
            "total_contributions": counts["TOTAL"] if counts else 0,
            "identified_contributions": counts["IDENTIFIED"] if counts else 0,
            "pending_contributions": counts["PENDING"] if counts else 0
        }
        
    finally:
        conn.close()


def list_meetings(limit: int = 10, status_filter: str = None) -> List[Dict[str, Any]]:
    """
    List recent meetings with their status
    
    Args:
        limit: Maximum number of meetings to return
        status_filter: Optional status filter (pending, in_progress, completed)
        
    Returns:
        List of meeting status dicts
    """
    conn = get_connection()
    try:
        cursor = conn.cursor(DictCursor)
        
        # Set context
        cursor.execute(f"USE DATABASE {config.SNOWFLAKE_DATABASE}")
        cursor.execute(f"USE SCHEMA {config.SNOWFLAKE_SCHEMA}")
        
        # Build query
        query = f"""
            SELECT 
                m.MEETING_ID,
                m.TITLE,
                m.MEETING_DATE,
                m.TRANSCRIPTION_STATUS,
                m.CLASSIFICATION_STATUS,
                m.CREATED_AT,
                COUNT(mc.CONTRIBUTION_ID) as TOTAL_CONTRIBUTIONS,
                COUNT(mc.IDENTIFIED_SPEAKER_ID) as IDENTIFIED_CONTRIBUTIONS
            FROM MEETINGS m
            LEFT JOIN MEETING_CONTRIBUTIONS mc ON m.MEETING_ID = mc.MEETING_ID
        """
        
        if status_filter:
            query += f" WHERE m.CLASSIFICATION_STATUS = '{status_filter}'"
        
        query += f"""
            GROUP BY m.MEETING_ID, m.TITLE, m.MEETING_DATE, 
                     m.TRANSCRIPTION_STATUS, m.CLASSIFICATION_STATUS, m.CREATED_AT
            ORDER BY m.CREATED_AT DESC
            LIMIT {limit}
        """
        
        cursor.execute(query)
        meetings = cursor.fetchall()
        
        results = []
        for m in meetings:
            total = m.get("TOTAL_CONTRIBUTIONS", 0)
            identified = m.get("IDENTIFIED_CONTRIBUTIONS", 0)
            
            if m.get("TRANSCRIPTION_STATUS") == "pending":
                status = MeetingStatus.TRANSCRIBING
            elif total > 0 and identified == total:
                status = MeetingStatus.COMPLETED
            elif m.get("CLASSIFICATION_STATUS") == "completed":
                status = MeetingStatus.COMPLETED
            elif identified > 0:
                status = MeetingStatus.IN_PROGRESS
            else:
                status = MeetingStatus.PENDING
            
            results.append({
                "meeting_id": m.get("MEETING_ID"),
                "title": m.get("TITLE"),
                "meeting_date": m.get("MEETING_DATE"),
                "status": status,
                "total_contributions": total,
                "identified_contributions": identified,
                "progress": f"{(identified/total*100):.0f}%" if total > 0 else "0%"
            })
        
        return results
        
    finally:
        conn.close()


def get_transcript(meeting_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Get the full transcript for a meeting with speaker names
    
    Returns:
        Tuple of (meeting_info, contributions_list)
    """
    conn = get_connection()
    try:
        cursor = conn.cursor(DictCursor)
        
        # Set context
        cursor.execute(f"USE DATABASE {config.SNOWFLAKE_DATABASE}")
        cursor.execute(f"USE SCHEMA {config.SNOWFLAKE_SCHEMA}")
        
        # Get meeting info
        cursor.execute(f"""
            SELECT 
                MEETING_ID, TITLE, MEETING_DATE, DURATION_MINUTES,
                TOTAL_SPEAKERS, LANGUAGE, SUMMARY
            FROM MEETINGS
            WHERE MEETING_ID = '{meeting_id}'
        """)
        meeting = cursor.fetchone()
        
        if not meeting:
            raise SnowflakeClientError(f"Meeting not found: {meeting_id}")
        
        # Get contributions with speaker names
        cursor.execute(f"""
            SELECT 
                mc.SEGMENT_NUMBER,
                mc.DIARIZATION_LABEL,
                mc.IDENTIFIED_SPEAKER_ID,
                COALESCE(s.DISPLAY_NAME, mc.DIARIZATION_LABEL) as SPEAKER_NAME,
                mc.TEXT_CONTENT,
                mc.START_TIME_SECONDS,
                mc.END_TIME_SECONDS,
                mc.DURATION_SECONDS
            FROM MEETING_CONTRIBUTIONS mc
            LEFT JOIN SPEAKERS s ON mc.IDENTIFIED_SPEAKER_ID = s.SPEAKER_ID
            WHERE mc.MEETING_ID = '{meeting_id}'
            ORDER BY mc.SEGMENT_NUMBER
        """)
        contributions = cursor.fetchall()
        
        # Get unique speakers
        speakers = set()
        for c in contributions:
            speakers.add(c.get("SPEAKER_NAME"))
        
        meeting_info = {
            "meeting_id": meeting.get("MEETING_ID"),
            "title": meeting.get("TITLE"),
            "meeting_date": meeting.get("MEETING_DATE"),
            "duration_minutes": meeting.get("DURATION_MINUTES"),
            "total_speakers": meeting.get("TOTAL_SPEAKERS"),
            "language": meeting.get("LANGUAGE"),
            "summary": meeting.get("SUMMARY"),
            "speakers": sorted(list(speakers))
        }
        
        return meeting_info, list(contributions)
        
    finally:
        conn.close()


def is_meeting_complete(meeting_id: str) -> bool:
    """Check if all speakers in a meeting have been identified"""
    status = get_meeting_status(meeting_id)
    return status.get("status") == MeetingStatus.COMPLETED
