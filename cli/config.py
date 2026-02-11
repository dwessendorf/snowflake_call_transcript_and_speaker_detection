"""
Configuration for Call Upload CLI

Update these settings to match your Snowflake environment.
You can also use environment variables to override these values.
"""

import os
from pathlib import Path

# ============================================================================
# Snowflake Connection Settings
# ============================================================================
# Option 1: Use a named connection from ~/.snowflake/connections.toml (recommended)
# Set this to your connection name, or leave empty to use direct settings
SNOWFLAKE_CONNECTION_NAME = os.environ.get("SNOWFLAKE_CONNECTION_NAME", "")

# Option 2: Direct connection settings (used if SNOWFLAKE_CONNECTION_NAME is empty)
SNOWFLAKE_ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT", "")  # e.g., "myaccount.us-east-1"
SNOWFLAKE_USER = os.environ.get("SNOWFLAKE_USER", "")
SNOWFLAKE_PASSWORD = os.environ.get("SNOWFLAKE_PASSWORD", "")

# Key-pair authentication (recommended for automation)
# Generate key: openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out rsa_key.p8 -nocrypt
SNOWFLAKE_PRIVATE_KEY_PATH = os.environ.get(
    "SNOWFLAKE_PRIVATE_KEY_PATH", 
    str(Path.home() / ".snowflake" / "rsa_key.p8")
)

# ============================================================================
# Database Settings
# ============================================================================
SNOWFLAKE_DATABASE = os.environ.get("SNOWFLAKE_DATABASE", "CALL_TRANSCRIPTS_DB")
SNOWFLAKE_SCHEMA = os.environ.get("SNOWFLAKE_SCHEMA", "TRANSCRIPTS")
SNOWFLAKE_WAREHOUSE = os.environ.get("SNOWFLAKE_WAREHOUSE", "CALL_TRANSCRIPTS_WH")

# ============================================================================
# Stages (auto-derived from database/schema)
# ============================================================================
STAGE_RECORDINGS = f"@{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.CALL_RECORDINGS"
STAGE_TRANSCRIPTIONS = f"@{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.CALL_TRANSCRIPTIONS"

# ============================================================================
# Output Directory
# ============================================================================
DEFAULT_OUTPUT_DIR = Path.home() / "Documents" / "CallTranscripts"

# ============================================================================
# Audio Format Settings
# ============================================================================
# Cortex AI_TRANSCRIBE supported formats (no conversion needed)
NATIVE_FORMATS = {".mp3", ".wav", ".flac", ".ogg", ".webm"}

# Formats that need conversion to MP3
CONVERT_FORMATS = {".m4a", ".mp4", ".mov", ".aac", ".wma", ".opus"}

# All supported formats
SUPPORTED_FORMATS = NATIVE_FORMATS | CONVERT_FORMATS

# Audio conversion settings (when using ffmpeg)
AUDIO_BITRATE = "192k"
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2

# ============================================================================
# Speaker Identification Settings
# ============================================================================
# Minimum confidence threshold for auto-classification (0.0 to 1.0)
SPEAKER_CONFIDENCE_THRESHOLD = 0.6

# Minimum segment duration (seconds) for embedding extraction
MIN_DURATION_FOR_EMBEDDING = 5.0

# ============================================================================
# Polling Settings (for --watch mode)
# ============================================================================
POLL_INTERVAL_SECONDS = 10
MAX_POLL_ATTEMPTS = 360  # 360 * 10 = 3600 seconds = 1 hour

# ============================================================================
# Identifiers
# ============================================================================
CALL_ID_PREFIX = "CALL"
