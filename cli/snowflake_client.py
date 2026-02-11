"""
Snowflake client for Call Upload CLI
Handles uploads, transcription, and status queries
"""

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import snowflake.connector
from snowflake.connector import DictCursor

try:
    from . import config
except ImportError:
    import config


class SnowflakeClientError(Exception):
    """Raised when Snowflake operations fail"""
    pass


class CallStatus:
    """Call processing status"""
    PENDING = "pending"
    TRANSCRIBING = "transcribing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


def get_connection() -> snowflake.connector.SnowflakeConnection:
    """Create Snowflake connection using keypair auth, connections.toml, or direct settings."""
    import os
    from pathlib import Path
    
    # Disable OCSP certificate checking for S3 transfers
    os.environ['SF_OCSP_RESPONSE_CACHE_SERVER_ENABLED'] = 'false'
    
    conn_params = {
        'database': config.SNOWFLAKE_DATABASE,
        'schema': config.SNOWFLAKE_SCHEMA,
        'warehouse': config.SNOWFLAKE_WAREHOUSE,
        'insecure_mode': True,
    }
    
    # Option 1: Use named connection from connections.toml
    conn_name = config.SNOWFLAKE_CONNECTION_NAME
    if conn_name:
        try:
            import toml
            connections_file = Path.home() / ".snowflake" / "connections.toml"
            if connections_file.exists():
                connections = toml.load(connections_file)
                if conn_name in connections:
                    conn_config = connections[conn_name]
                    conn_params['account'] = conn_config.get('account')
                    conn_params['user'] = conn_config.get('user')
                    if 'role' in conn_config:
                        conn_params['role'] = conn_config['role']
        except ImportError:
            pass  # toml not installed, use direct settings
    
    # Option 2: Use direct settings from config/environment
    if 'account' not in conn_params or not conn_params['account']:
        if config.SNOWFLAKE_ACCOUNT:
            conn_params['account'] = config.SNOWFLAKE_ACCOUNT
            conn_params['user'] = config.SNOWFLAKE_USER
        else:
            raise SnowflakeClientError(
                "No Snowflake connection configured. Set SNOWFLAKE_CONNECTION_NAME "
                "or SNOWFLAKE_ACCOUNT/SNOWFLAKE_USER environment variables."
            )
    
    # Authentication: Try keypair first, then password
    key_path = Path(config.SNOWFLAKE_PRIVATE_KEY_PATH)
    if key_path.exists():
        try:
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import serialization
            
            with open(key_path, 'rb') as f:
                private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None,
                    backend=default_backend()
                )
            conn_params['private_key'] = private_key
            return snowflake.connector.connect(**conn_params)
        except Exception:
            pass  # Fall back to password auth
    
    # Password authentication
    if config.SNOWFLAKE_PASSWORD:
        conn_params['password'] = config.SNOWFLAKE_PASSWORD
    else:
        raise SnowflakeClientError(
            "No authentication method available. Provide either:\n"
            "  - RSA key at ~/.snowflake/rsa_key.p8\n"
            "  - SNOWFLAKE_PASSWORD environment variable"
        )
    
    try:
        return snowflake.connector.connect(**conn_params)
    except Exception as e:
        raise SnowflakeClientError(f"Failed to connect to Snowflake: {e}")


def generate_call_id(title: str) -> str:
    """Generate a unique call ID from title"""
    # Clean title for ID - keep shorter to fit contribution IDs
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    clean = re.sub(r'\s+', '_', clean).upper()[:18]  # Shorter to allow contrib suffix
    
    # Add date and short UUID
    date_str = datetime.now().strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:6].upper()
    
    return f"{config.CALL_ID_PREFIX}_{date_str}_{clean}_{short_uuid}"


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
        stage_name: Target stage (default: CALL_RECORDINGS)
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
    call_title: str,
    call_date: Optional[datetime] = None
) -> str:
    """
    Start transcription for an uploaded audio file using Cortex AI_TRANSCRIBE.
    
    The workflow:
    1. Create a CALLS record with the stage path
    2. Call TRANSCRIBE_CALL procedure which uses AI_TRANSCRIBE with speaker diarization
    3. Contributions are automatically created from transcription segments
    
    Args:
        stage_path: Path to file in stage (e.g., @DB.SCHEMA.STAGE/file.mp3)
        call_title: Title for the call
        call_date: Date of call (default: today)
        
    Returns:
        Call ID
    """
    import json
    
    if call_date is None:
        call_date = datetime.now()
    
    call_id = generate_call_id(call_title)
    
    conn = get_connection()
    try:
        cursor = conn.cursor()
        
        # Set context
        cursor.execute(f"USE DATABASE {config.SNOWFLAKE_DATABASE}")
        cursor.execute(f"USE SCHEMA {config.SNOWFLAKE_SCHEMA}")
        cursor.execute(f"USE WAREHOUSE {config.SNOWFLAKE_WAREHOUSE}")
        
        # First, create the call record
        insert_sql = f"""
        INSERT INTO CALLS (
            CALL_ID, TITLE, CALL_DATE, RECORDING_PATH, 
            TRANSCRIPTION_STATUS, CLASSIFICATION_STATUS, CREATED_AT
        ) VALUES (
            '{call_id}', 
            '{call_title.replace("'", "''")}', 
            '{call_date.strftime("%Y-%m-%d")}',
            '{stage_path}',
            'pending',
            'pending',
            CURRENT_TIMESTAMP
        )
        """
        cursor.execute(insert_sql)
        
        # Commit the insert so the procedure can see it
        # (procedure runs in separate transaction context)
        conn.commit()
        
        # Call the TRANSCRIBE_CALL procedure which uses Cortex AI_TRANSCRIBE
        cursor.execute(f"CALL TRANSCRIBE_CALL('{call_id}')")
        result = cursor.fetchone()
        
        if result:
            try:
                data = json.loads(result[0]) if isinstance(result[0], str) else result[0]
                if data.get("status") == "success":
                    return call_id
                elif data.get("status") == "error":
                    raise SnowflakeClientError(f"Transcription failed: {data.get('message')}")
            except json.JSONDecodeError:
                # Result might be something else, continue
                pass
        
        return call_id
        
    finally:
        conn.close()


def start_speaker_identification(call_id: str, threshold: float = None) -> Dict[str, Any]:
    """
    Start speaker identification for a call
    
    Args:
        call_id: Call ID to process
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
        cursor.execute(f"CALL IDENTIFY_CALL_SPEAKERS_WITH_EMBEDDINGS('{call_id}', {threshold})")
        result = cursor.fetchone()
        
        identification_result = {"status": "started", "call_id": call_id}
        if result:
            import json
            try:
                identification_result = json.loads(result[0]) if isinstance(result[0], str) else result[0]
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Run automatic speaker detection using voiceprints
        try:
            cursor.execute(f"CALL AUTO_DETECT_SPEAKERS('{call_id}', {threshold})")
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


def get_call_status(call_id: str) -> Dict[str, Any]:
    """
    Get the current status of a call
    
    Returns dict with: status, title, total_contributions, identified_contributions, etc.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor(DictCursor)
        
        # Set context
        cursor.execute(f"USE DATABASE {config.SNOWFLAKE_DATABASE}")
        cursor.execute(f"USE SCHEMA {config.SNOWFLAKE_SCHEMA}")
        
        # Get call info
        cursor.execute(f"""
            SELECT 
                c.CALL_ID,
                c.TITLE,
                c.CALL_DATE,
                c.TRANSCRIPTION_STATUS,
                c.CLASSIFICATION_STATUS,
                c.TOTAL_SPEAKERS,
                c.LANGUAGE,
                c.DURATION_MINUTES
            FROM CALLS c
            WHERE c.CALL_ID = '{call_id}'
        """)
        call = cursor.fetchone()
        
        if not call:
            return {"status": "not_found", "call_id": call_id}
        
        # Get contribution counts
        cursor.execute(f"""
            SELECT 
                COUNT(*) as total,
                COUNT(IDENTIFIED_SPEAKER_ID) as identified,
                COUNT(CASE WHEN CLASSIFICATION_STATUS = 'pending' THEN 1 END) as pending
            FROM CALL_CONTRIBUTIONS
            WHERE CALL_ID = '{call_id}'
        """)
        counts = cursor.fetchone()
        
        # Determine overall status
        if call.get("TRANSCRIPTION_STATUS") == "pending":
            status = CallStatus.TRANSCRIBING
        elif counts and counts["TOTAL"] > 0 and counts["IDENTIFIED"] == counts["TOTAL"]:
            status = CallStatus.COMPLETED
        elif call.get("CLASSIFICATION_STATUS") == "completed":
            status = CallStatus.COMPLETED
        elif counts and counts["IDENTIFIED"] > 0:
            status = CallStatus.IN_PROGRESS
        else:
            status = CallStatus.PENDING
        
        return {
            "status": status,
            "call_id": call_id,
            "title": call.get("TITLE"),
            "call_date": call.get("CALL_DATE"),
            "transcription_status": call.get("TRANSCRIPTION_STATUS"),
            "classification_status": call.get("CLASSIFICATION_STATUS"),
            "total_speakers": call.get("TOTAL_SPEAKERS"),
            "language": call.get("LANGUAGE"),
            "duration_minutes": call.get("DURATION_MINUTES"),
            "total_contributions": counts["TOTAL"] if counts else 0,
            "identified_contributions": counts["IDENTIFIED"] if counts else 0,
            "pending_contributions": counts["PENDING"] if counts else 0
        }
        
    finally:
        conn.close()


def list_calls(limit: int = 10, status_filter: str = None) -> List[Dict[str, Any]]:
    """
    List recent calls with their status
    
    Args:
        limit: Maximum number of calls to return
        status_filter: Optional status filter (pending, in_progress, completed)
        
    Returns:
        List of call status dicts
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
                c.CALL_ID,
                c.TITLE,
                c.CALL_DATE,
                c.TRANSCRIPTION_STATUS,
                c.CLASSIFICATION_STATUS,
                c.CREATED_AT,
                COUNT(cc.CONTRIBUTION_ID) as TOTAL_CONTRIBUTIONS,
                COUNT(cc.IDENTIFIED_SPEAKER_ID) as IDENTIFIED_CONTRIBUTIONS
            FROM CALLS c
            LEFT JOIN CALL_CONTRIBUTIONS cc ON c.CALL_ID = cc.CALL_ID
        """
        
        if status_filter:
            query += f" WHERE c.CLASSIFICATION_STATUS = '{status_filter}'"
        
        query += f"""
            GROUP BY c.CALL_ID, c.TITLE, c.CALL_DATE, 
                     c.TRANSCRIPTION_STATUS, c.CLASSIFICATION_STATUS, c.CREATED_AT
            ORDER BY c.CREATED_AT DESC
            LIMIT {limit}
        """
        
        cursor.execute(query)
        calls = cursor.fetchall()
        
        results = []
        for c in calls:
            total = c.get("TOTAL_CONTRIBUTIONS", 0)
            identified = c.get("IDENTIFIED_CONTRIBUTIONS", 0)
            
            if c.get("TRANSCRIPTION_STATUS") == "pending":
                status = CallStatus.TRANSCRIBING
            elif total > 0 and identified == total:
                status = CallStatus.COMPLETED
            elif c.get("CLASSIFICATION_STATUS") == "completed":
                status = CallStatus.COMPLETED
            elif identified > 0:
                status = CallStatus.IN_PROGRESS
            else:
                status = CallStatus.PENDING
            
            results.append({
                "call_id": c.get("CALL_ID"),
                "title": c.get("TITLE"),
                "call_date": c.get("CALL_DATE"),
                "status": status,
                "total_contributions": total,
                "identified_contributions": identified,
                "progress": f"{(identified/total*100):.0f}%" if total > 0 else "0%"
            })
        
        return results
        
    finally:
        conn.close()


def get_transcript(call_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Get the full transcript for a call with speaker names
    
    Returns:
        Tuple of (call_info, contributions_list)
    """
    conn = get_connection()
    try:
        cursor = conn.cursor(DictCursor)
        
        # Set context
        cursor.execute(f"USE DATABASE {config.SNOWFLAKE_DATABASE}")
        cursor.execute(f"USE SCHEMA {config.SNOWFLAKE_SCHEMA}")
        
        # Get call info
        cursor.execute(f"""
            SELECT 
                CALL_ID, TITLE, CALL_DATE, DURATION_MINUTES,
                TOTAL_SPEAKERS, LANGUAGE, SUMMARY
            FROM CALLS
            WHERE CALL_ID = '{call_id}'
        """)
        call = cursor.fetchone()
        
        if not call:
            raise SnowflakeClientError(f"Call not found: {call_id}")
        
        # Get contributions with speaker names
        cursor.execute(f"""
            SELECT 
                cc.SEGMENT_NUMBER,
                cc.DIARIZATION_LABEL,
                cc.IDENTIFIED_SPEAKER_ID,
                COALESCE(s.DISPLAY_NAME, cc.DIARIZATION_LABEL) as SPEAKER_NAME,
                cc.TEXT_CONTENT,
                cc.START_TIME_SECONDS,
                cc.END_TIME_SECONDS,
                cc.DURATION_SECONDS
            FROM CALL_CONTRIBUTIONS cc
            LEFT JOIN SPEAKERS s ON cc.IDENTIFIED_SPEAKER_ID = s.SPEAKER_ID
            WHERE cc.CALL_ID = '{call_id}'
            ORDER BY cc.SEGMENT_NUMBER
        """)
        contributions = cursor.fetchall()
        
        # Get unique speakers
        speakers = set()
        for c in contributions:
            speakers.add(c.get("SPEAKER_NAME"))
        
        call_info = {
            "call_id": call.get("CALL_ID"),
            "title": call.get("TITLE"),
            "call_date": call.get("CALL_DATE"),
            "duration_minutes": call.get("DURATION_MINUTES"),
            "total_speakers": call.get("TOTAL_SPEAKERS"),
            "language": call.get("LANGUAGE"),
            "summary": call.get("SUMMARY"),
            "speakers": sorted(list(speakers))
        }
        
        return call_info, list(contributions)
        
    finally:
        conn.close()


def is_call_complete(call_id: str) -> bool:
    """Check if all speakers in a call have been identified"""
    status = get_call_status(call_id)
    return status.get("status") == CallStatus.COMPLETED
