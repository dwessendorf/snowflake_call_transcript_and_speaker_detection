# Snowflake Speaker Detection

A complete solution for meeting transcription with automatic speaker diarization and identification, built on Snowflake.

## Features

- **Audio Upload & Transcription**: Upload meeting recordings (MP3, M4A, WAV, etc.) and get automatic transcription using Snowflake's AI_TRANSCRIBE
- **Speaker Diarization**: Automatic detection of different speakers in the meeting
- **Speaker Identification**: Match speakers to known profiles using voice embeddings (ECAPA-TDNN model via Snowflake Container Services)
- **Streamlit UI**: Web-based interface for speaker classification and management
- **CLI Tool**: Command-line interface for uploading meetings and monitoring progress

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Snowflake Account                           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │   Stages     │   │    Tables    │   │   SPCS Service       │    │
│  │              │   │              │   │                      │    │
│  │ - RECORDINGS │   │ - MEETINGS   │   │ Speaker ID Service   │    │
│  │ - SNIPPETS   │   │ - SPEAKERS   │   │ (ECAPA-TDNN Model)   │    │
│  │ - APPS       │   │ - CONTRIB.   │   │                      │    │
│  └──────────────┘   └──────────────┘   └──────────────────────┘    │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Streamlit App                              │  │
│  │  - Speaker Classification UI                                  │  │
│  │  - Audio Preview with Timestamps                              │  │
│  │  - Speaker Management                                         │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
         ▲
         │ Upload via CLI
         │
┌─────────────────┐
│   Local CLI     │
│                 │
│ meeting-cli     │
│   upload        │
│   status        │
│   export        │
└─────────────────┘
```

## Prerequisites

- Snowflake account with:
  - ACCOUNTADMIN role (for initial setup)
  - Access to Snowflake Container Services (SPCS)
  - AI_TRANSCRIBE function available
- Python 3.10+ (for CLI)
- Docker (for building SPCS image)
- ffmpeg (for audio conversion)

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd snowflake_speaker_detection
```

### Step 2: Configure Snowflake Connection

Create a Snowflake connection configuration. You can use either:

**Option A: SnowSQL config (~/.snowsql/config)**
```ini
[connections.my_connection]
accountname = <account_identifier>
username = <username>
password = <password>
warehousename = <warehouse>
dbname = MEETING_AGENT_DB
schemaname = MEETING_AGENT
```

**Option B: Environment variables**
```bash
export SNOWFLAKE_ACCOUNT=<account_identifier>
export SNOWFLAKE_USER=<username>
export SNOWFLAKE_PASSWORD=<password>
export SNOWFLAKE_WAREHOUSE=<warehouse>
export SNOWFLAKE_DATABASE=MEETING_AGENT_DB
export SNOWFLAKE_SCHEMA=MEETING_AGENT
```

### Step 3: Run SQL Setup Scripts

Execute the SQL scripts in order:

```bash
# Connect to Snowflake and run:
snowsql -c my_connection -f sql/01_setup_database.sql
snowsql -c my_connection -f sql/02_create_tables.sql
snowsql -c my_connection -f sql/03_create_stages.sql
snowsql -c my_connection -f sql/04_setup_spcs.sql
snowsql -c my_connection -f sql/05_create_functions.sql
snowsql -c my_connection -f sql/06_create_streamlit_app.sql
```

### Step 4: Build and Deploy SPCS Service

```bash
cd spcs_service

# Build Docker image
docker build -t speaker-identification:latest .

# Tag for Snowflake registry
docker tag speaker-identification:latest \
  <account>.registry.snowflakecomputing.com/meeting_agent_db/meeting_agent/speaker_identification_repo/speaker-identification:v1

# Login to Snowflake registry
docker login <account>.registry.snowflakecomputing.com

# Push image
docker push <account>.registry.snowflakecomputing.com/meeting_agent_db/meeting_agent/speaker_identification_repo/speaker-identification:v1
```

### Step 5: Install CLI

```bash
cd cli
pip install -r requirements.txt
pip install -e .
```

### Step 6: Update Configuration

Edit `cli/config.py` to match your Snowflake settings:

```python
SNOWFLAKE_CONNECTION = "my_connection"
DATABASE = "MEETING_AGENT_DB"
SCHEMA = "MEETING_AGENT"
WAREHOUSE = "YOUR_WAREHOUSE"
```

## Usage

### Upload a Meeting

```bash
# Upload and wait for transcription
meeting-cli upload "path/to/meeting.m4a" --watch

# Upload with custom title
meeting-cli upload "meeting.mp3" --title "Q4 Planning Session"
```

### Check Status

```bash
meeting-cli status
meeting-cli status <meeting_id>
```

### Export Transcript

```bash
meeting-cli export <meeting_id> --format markdown
meeting-cli export <meeting_id> --format json
```

### Classify Speakers (Web UI)

Open the Streamlit app in Snowsight:
1. Navigate to Streamlit Apps
2. Open "Speaker Classification App"
3. Select a meeting and assign speakers to voice segments

## Directory Structure

```
snowflake_speaker_detection/
├── README.md                 # This file
├── streamlit_app/
│   ├── app.py               # Main Streamlit application
│   └── environment.yml      # Conda environment for Snowflake
├── spcs_service/
│   ├── Dockerfile           # Container image definition
│   ├── speaker_service.py   # Flask service for embeddings
│   └── config.py            # Service configuration
├── cli/
│   ├── __init__.py
│   ├── meeting_cli.py       # CLI entry point
│   ├── audio.py             # Audio processing utilities
│   ├── snowflake_client.py  # Snowflake connection handler
│   ├── transcript.py        # Transcript export utilities
│   ├── config.py            # CLI configuration
│   ├── requirements.txt     # Python dependencies
│   └── setup.py             # Package installation
├── sql/
│   ├── 01_setup_database.sql
│   ├── 02_create_tables.sql
│   ├── 03_create_stages.sql
│   ├── 04_setup_spcs.sql
│   ├── 05_create_functions.sql
│   └── 06_create_streamlit_app.sql
└── docs/
    └── troubleshooting.md
```

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SNOWFLAKE_CONNECTION` | Connection name from snowsql config | - |
| `SNOWFLAKE_DATABASE` | Database name | `MEETING_AGENT_DB` |
| `SNOWFLAKE_SCHEMA` | Schema name | `MEETING_AGENT` |
| `SNOWFLAKE_WAREHOUSE` | Warehouse name | - |

### Embedding Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `MIN_DURATION_FOR_EMBEDDING` | Minimum segment duration (seconds) for embedding extraction | `5.0` |
| `SIMILARITY_THRESHOLD` | Threshold for auto-matching speakers | `0.75` |

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md) for common issues and solutions.

### Common Issues

1. **SPCS Service not responding**: Check compute pool status and service logs
2. **Audio conversion fails**: Ensure ffmpeg is installed
3. **Embedding extraction fails**: Verify network rules allow S3 access for presigned URLs

## License

MIT License - see LICENSE file for details.
