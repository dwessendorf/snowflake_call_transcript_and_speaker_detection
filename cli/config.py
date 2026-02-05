"""
Configuration for Meeting Upload CLI

Update these settings to match your Snowflake environment.
You can also use environment variables to override these values.
"""

import os
from pathlib import Path

# ============================================================================
# Snowflake Connection Settings
# ============================================================================
# Option 1: Use a named connection from ~/.snowsql/config
SNOWFLAKE_CONNECTION_NAME = os.environ.get("SNOWFLAKE_CONNECTION_NAME", "")

# Option 2: Direct connection settings (used if SNOWFLAKE_CONNECTION_NAME is empty)
SNOWFLAKE_ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT", "")  # e.g., "xy12345.us-east-1"
SNOWFLAKE_USER = os.environ.get("SNOWFLAKE_USER", "")
SNOWFLAKE_PASSWORD = os.environ.get("SNOWFLAKE_PASSWORD", "")  # Or use key-based auth

# Key-based authentication (optional, more secure)
SNOWFLAKE_PRIVATE_KEY_PATH = os.environ.get(
    "SNOWFLAKE_PRIVATE_KEY_PATH", 
    str(Path.home() / ".snowflake" / "rsa_key.p8")
)

# ============================================================================
# Database Settings
# ============================================================================
SNOWFLAKE_DATABASE = os.environ.get("SNOWFLAKE_DATABASE", "MEETING_AGENT_DB")
SNOWFLAKE_SCHEMA = os.environ.get("SNOWFLAKE_SCHEMA", "MEETING_AGENT")
SNOWFLAKE_WAREHOUSE = os.environ.get("SNOWFLAKE_WAREHOUSE", "MEETING_AGENT_WH")

# ============================================================================
# Stages (auto-derived from database/schema)
# ============================================================================
STAGE_RECORDINGS = f"@{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.MEETING_RECORDINGS"
STAGE_TRANSCRIPTIONS = f"@{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.MEETING_TRANSCRIPTIONS"

# ============================================================================
# Output Directory
# ============================================================================
DEFAULT_OUTPUT_DIR = Path.home() / "Documents" / "MeetingTranscripts"

# ============================================================================
# Audio Format Settings
# ============================================================================
# Formats that don't need conversion
NATIVE_FORMATS = {".mp3", ".wav", ".flac"}

# Formats that need conversion to MP3
CONVERT_FORMATS = {".m4a", ".mp4", ".mov", ".aac", ".ogg", ".wma", ".webm", ".opus"}

# All supported formats
SUPPORTED_FORMATS = NATIVE_FORMATS | CONVERT_FORMATS

# Audio conversion settings (when using ffmpeg)
AUDIO_BITRATE = "192k"
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2

# ============================================================================
# Speaker Identification Settings
# ============================================================================
# Minimum confidence threshold for auto-classification
SPEAKER_CONFIDENCE_THRESHOLD = 0.75

# Minimum segment duration (seconds) for embedding extraction
MIN_DURATION_FOR_EMBEDDING = 5.0

# ============================================================================
# Polling Settings
# ============================================================================
# Interval between status checks when using --watch
POLL_INTERVAL_SECONDS = 10

# Maximum wait time (attempts * interval = 3600 seconds = 1 hour)
MAX_POLL_ATTEMPTS = 360

# ============================================================================
# Identifiers
# ============================================================================
MEETING_ID_PREFIX = "MTG"
