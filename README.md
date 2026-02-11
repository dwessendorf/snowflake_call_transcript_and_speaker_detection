# Call Transcript & Speaker Detection for Snowflake

A complete solution for transcribing call recordings and automatically identifying speakers using Snowflake's native AI capabilities.

## Features

- **Automatic Transcription**: Uses Cortex AI_TRANSCRIBE for speech-to-text with speaker diarization
- **Speaker Identification**: Machine learning-based voice embedding matching using SpeechBrain ECAPA-TDNN
- **Voice Enrollment**: Register known speakers from call segments for future identification
- **CLI Tool**: Upload, transcribe, and export calls from the command line
- **Model Registry**: Deploys speaker embedding model via Snowflake Model Registry

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Snowflake                               │
│  ┌───────────-────┐  ┌────────────────┐  ┌───────────────────┐  │
│  │ CALL_RECORDINGS│  │ Cortex         │  │ Model Registry    │  │
│  │ Stage (audio)  │→ │ AI_TRANSCRIBE  │→ │ Speaker Embedding │  │
│  └────────────-───┘  └────────────────┘  └───────────────────┘  │
│                              ↓                    ↓             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ CALLS, CALL_CONTRIBUTIONS, SPEAKER_VOICEPRINTS tables     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                    ┌─────────────────┐
                    │   CLI Tool      │
                    │ (Python client) │
                    └─────────────────┘
```

## Requirements

- Snowflake account (AWS or Azure region)
- ACCOUNTADMIN role or equivalent privileges
- Python 3.9+ (for CLI and model deployment)
- ffmpeg (optional, for audio format conversion)

## Quick Start

### 1. Deploy Snowflake Infrastructure

```sql
-- Run in Snowflake worksheet or via SnowSQL
-- Enable cross-region Cortex if needed
ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';

-- Run the setup script
@deploy/01_snowflake_setup.sql
```

### 2. Deploy Speaker Embedding Model

```bash
# Install Python dependencies
pip install snowflake-ml-python snowflake-snowpark-python

# Set connection (choose one method)
export SNOWFLAKE_CONNECTION_NAME=my_connection
# OR
export SNOWFLAKE_ACCOUNT=myaccount.us-east-1
export SNOWFLAKE_USER=myuser
export SNOWFLAKE_PASSWORD=mypassword

# Register and deploy model
python deploy/register_model.py
```

### 3. Create Service Functions

```sql
-- Run after model is deployed
@deploy/02_service_functions.sql
```

### 4. Install CLI Tool

```bash
cd cli
pip install -r requirements.txt

# Configure connection
export SNOWFLAKE_CONNECTION_NAME=my_connection
# OR set up keypair auth at ~/.snowflake/rsa_key.p8
```

## CLI Usage

### Upload and Transcribe a Call

```bash
python -m cli.call_cli upload recording.mp3 --title "Sales Meeting"
```

### Check Status

```bash
python -m cli.call_cli status CALL_20260211_SALES_MEETING_ABC123
```

### Export Transcript

```bash
python -m cli.call_cli export CALL_20260211_SALES_MEETING_ABC123
```

### List Recent Calls

```bash
python -m cli.call_cli list --limit 10
```

## Speaker Enrollment

To enable automatic speaker identification, enroll known speakers:

```sql
-- From Snowflake worksheet
CALL ENROLL_SPEAKER_FROM_CONTRIBUTION(
    'SPEAKER_JOHN_DOE',      -- Speaker ID
    'John Doe',              -- Display name
    'CALL_20260211_...',     -- Call ID with their voice
    'SPEAKER_01'             -- Diarization label to use
);
```

Once enrolled, future calls will automatically match voices against the voiceprint database.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SNOWFLAKE_CONNECTION_NAME` | Connection name from connections.toml | (none) |
| `SNOWFLAKE_ACCOUNT` | Snowflake account identifier | (none) |
| `SNOWFLAKE_USER` | Username | (none) |
| `SNOWFLAKE_PASSWORD` | Password (if not using keypair) | (none) |
| `SNOWFLAKE_DATABASE` | Database name | CALL_TRANSCRIPTS_DB |
| `SNOWFLAKE_SCHEMA` | Schema name | TRANSCRIPTS |
| `SNOWFLAKE_WAREHOUSE` | Warehouse name | CALL_TRANSCRIPTS_WH |

### Keypair Authentication (Recommended)

```bash
# Generate RSA key pair
openssl genrsa 2048 | openssl pkcs8 -topk8 -inform PEM -out ~/.snowflake/rsa_key.p8 -nocrypt
openssl rsa -in ~/.snowflake/rsa_key.p8 -pubout -out ~/.snowflake/rsa_key.pub

# Register public key in Snowflake
ALTER USER myuser SET RSA_PUBLIC_KEY='MIIBIjANBgkq...';
```

## Supported Audio Formats

**Native (no conversion):** MP3, WAV, FLAC, OGG, WEBM

**Requires ffmpeg:** M4A, MP4, MOV, AAC, WMA, OPUS

## Project Structure

```
├── deploy/
│   ├── 01_snowflake_setup.sql    # Database, tables, procedures
│   ├── 02_service_functions.sql  # Model service functions
│   └── register_model.py         # Model registration script
├── cli/
│   ├── call_cli.py               # Main CLI entry point
│   ├── snowflake_client.py       # Snowflake operations
│   ├── config.py                 # Configuration
│   └── requirements.txt          # Python dependencies
├── spcs_service/
│   └── speaker_model_registry.py # Speaker embedding model
└── README.md
```

## Troubleshooting

### "AI_TRANSCRIBE model not available"
Enable cross-region inference:
```sql
ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';
```

### "Invalid file format"
Cortex AI_TRANSCRIBE only supports: FLAC, MP3, OGG, WAV, WEBM. Convert other formats using ffmpeg.

### "No authentication method available"
Either:
1. Set up keypair authentication with RSA key at `~/.snowflake/rsa_key.p8`
2. Set `SNOWFLAKE_PASSWORD` environment variable

### Speaker identification returns 0 matches
- Ensure speakers are enrolled with `ENROLL_SPEAKER_FROM_CONTRIBUTION`
- Check minimum segment duration (>= 5 seconds recommended)
- Lower the threshold if needed (default 0.6)

## License

MIT License - See LICENSE file
