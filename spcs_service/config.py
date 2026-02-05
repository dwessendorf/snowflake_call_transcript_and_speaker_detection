"""
Configuration for Speaker Identification Service
"""

import os

# Snowflake connection parameters (loaded from environment)
SNOWFLAKE_ACCOUNT = os.environ.get('SNOWFLAKE_ACCOUNT', '')
SNOWFLAKE_USER = os.environ.get('SNOWFLAKE_USER', '')
SNOWFLAKE_PASSWORD = os.environ.get('SNOWFLAKE_PASSWORD', '')
SNOWFLAKE_DATABASE = os.environ.get('SNOWFLAKE_DATABASE', 'CALL_TRANSCRIPTS_DB')
SNOWFLAKE_SCHEMA = os.environ.get('SNOWFLAKE_SCHEMA', 'TRANSCRIPTS')
SNOWFLAKE_WAREHOUSE = os.environ.get('SNOWFLAKE_WAREHOUSE', 'CALL_TRANSCRIPTS_WH')

# Model configuration
EMBEDDING_DIMENSION = 192  # pyannote.audio embedding size
CONFIDENCE_THRESHOLD = 0.75  # Default threshold for speaker matching

# Service configuration
SERVICE_PORT = 8080
WORKERS = 2
TIMEOUT = 300  # seconds

# Temporary storage
TEMP_DIR = '/tmp'
MAX_AUDIO_DURATION_SECONDS = 7200  # 2 hours max
