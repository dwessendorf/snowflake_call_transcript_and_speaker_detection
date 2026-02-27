# Call Transcript & Speaker Detection for Snowflake

A complete solution for transcribing call recordings and automatically identifying speakers using Snowflake's native AI capabilities.

## Features

- **Automatic Transcription**: Uses Cortex AI_TRANSCRIBE for speech-to-text with speaker diarization
- **Speaker Identification**: ML-based voice embedding matching using SpeechBrain ECAPA-TDNN on GPU
- **Voice Enrollment**: Register known speakers from call segments for future identification
- **Streamlit App**: Web UI for speaker assignment and call management
- **Background Processing**: Automated tasks for embedding extraction and data maintenance
- **CLI Tool**: Upload, transcribe, and export calls from the command line

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Snowflake                                  │
│  ┌─────────────────┐  ┌────────────────┐  ┌─────────────────────┐   │
│  │ CALL_RECORDINGS │  │ Cortex         │  │ GPU Service (SPCS)  │   │
│  │ Stage (audio)   │→ │ AI_TRANSCRIBE  │→ │ Speaker Embedding   │   │
│  └─────────────────┘  └────────────────┘  └─────────────────────┘   │
│                              ↓                      ↓               │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ CALLS, CALL_CONTRIBUTIONS, SPEAKERS, SPEAKER_VOICEPRINTS      │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              ↑                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Background Tasks: Embedding extraction, Housekeeping          │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              ↑                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ Streamlit App: Speaker Assignment UI                          │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              ↑
                    ┌─────────────────┐
                    │   CLI Tool      │
                    │ (Python client) │
                    └─────────────────┘
```

## Requirements

- Snowflake account (AWS or Azure region)
- ACCOUNTADMIN role or equivalent privileges
- Python 3.11+ (for CLI and model deployment)
- Snowflake CLI (`snow`) for Streamlit deployment
- ffmpeg (optional, for audio format conversion)

## Quick Start

### One-Command Deployment

```bash
# Deploy everything with a single command
./deploy/deploy.sh YOUR_CONNECTION_NAME
```

### Manual Deployment Steps

#### 1. Deploy Snowflake Infrastructure

```sql
-- Enable cross-region Cortex if needed
ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';

-- Run the setup script (creates DB, tables, procedures, tasks)
-- Use Snowflake worksheet or: snow sql -f deploy/01_snowflake_setup.sql
```

#### 2. Deploy Speaker Embedding Model (GPU Service)

```bash
# Install Python dependencies
pip install snowflake-ml-python snowflake-snowpark-python

# Register and deploy model to GPU compute pool
python deploy/register_model.py --connection YOUR_CONNECTION_NAME
```

#### 3. Create Service Functions

```sql
-- Run after model is deployed
-- deploy/02_service_functions.sql
```

#### 4. Deploy Streamlit App

```bash
cd streamlit_app
snow streamlit deploy --replace --connection YOUR_CONNECTION_NAME
```

## Project Structure

```
├── deploy/
│   ├── deploy.sh                 # Master deployment script
│   ├── 01_snowflake_setup.sql    # Database, tables, procedures, tasks
│   ├── 02_service_functions.sql  # GPU service functions
│   └── register_model.py         # Model registration script
├── streamlit_app/
│   ├── app.py                    # Speaker assignment web UI
│   ├── speaker_helper.py         # Speaker utilities
│   └── snowflake.yml             # Streamlit deployment config
├── spcs_service/
│   ├── speaker_model_registry.py # Speaker embedding model
│   ├── speaker_service.py        # HTTP service wrapper
│   └── config.py                 # Service configuration
├── cli/
│   ├── call_cli.py               # Main CLI entry point
│   ├── snowflake_client.py       # Snowflake operations
│   └── config.py                 # CLI configuration
├── sql/_deprecated/              # Old SQL files (superseded by deploy/)
└── test_data/                    # Sample audio files
```

## Database Objects

### Tables
| Table | Description |
|-------|-------------|
| CALLS | Call metadata and status |
| CALL_CONTRIBUTIONS | Speech segments with embeddings |
| SPEAKERS | Known speaker registry |
| SPEAKER_VOICEPRINTS | Voice embeddings for identification |
| VOICEPRINT_QUEUE | Async voiceprint creation queue |
| CLASSIFICATION_QUEUE | Manual review workflow |

### Procedures
| Procedure | Description |
|-----------|-------------|
| TRANSCRIBE_CALL | Transcribe audio with speaker diarization |
| CREATE_SPEAKER_VOICEPRINT_FROM_CONTRIBUTION | Create voiceprint from segment |
| EXTRACT_NEW_EMBEDDINGS | Batch extract embeddings for contributions |

### Tasks (Background)
| Task | Schedule | Description |
|------|----------|-------------|
| EXTRACT_NEW_EMBEDDINGS_TASK | Hourly | Extract embeddings for new contributions |
| SPEAKER_ASSIGNMENT_HOUSEKEEPING | Every 5 min | Update call status & speaker counts |

### Service Functions
| Function | Description |
|----------|-------------|
| SPEAKER_EMBEDDING_URL | Extract embedding from audio URL |
| SPEAKER_EMBEDDING_HEALTH | Service health check |

## Usage

### CLI: Upload and Transcribe

```bash
python -m cli.call_cli upload recording.mp3 --title "Sales Meeting"
python -m cli.call_cli status CALL_ID
python -m cli.call_cli export CALL_ID
```

### SQL: Manual Transcription

```sql
-- Transcribe a call
CALL TRANSCRIBE_CALL('CALL_ID');

-- Extract embeddings for new contributions
CALL EXTRACT_NEW_EMBEDDINGS(100);

-- Create voiceprint for a speaker
CALL CREATE_SPEAKER_VOICEPRINT_FROM_CONTRIBUTION('speaker_id', 'call_id', 'SPEAKER_01');
```

### Streamlit App

Access the speaker assignment app at:
`https://app.snowflake.com/<account>/CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_ASSIGNMENTS_APP`

## Configuration

### GPU Service
- **Compute Pool**: `SPEAKER_IDENTIFICATION_POOL` (GPU_NV_S)
- **Auto-suspend**: 300 seconds (5 minutes)
- **Instances**: 1 (configurable)

### Warehouses
- **CALL_TRANSCRIPTS_WH**: Main operations (SMALL)
- **STREAMLIT_APP_WH**: Streamlit app (X-SMALL)

## Troubleshooting

### "AI_TRANSCRIBE model not available"
```sql
ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';
```

### GPU service not responding
```sql
-- Check service status
SHOW SERVICES LIKE 'SPEAKER_EMBEDDING%';

-- Resume if suspended
ALTER SERVICE SPEAKER_EMBEDDING_SVC RESUME;
```

### Embeddings not being extracted
```sql
-- Check task status
SHOW TASKS LIKE '%EMBEDDING%';

-- Manually run extraction
CALL EXTRACT_NEW_EMBEDDINGS(100);
```

## Supported Audio Formats

**Native (no conversion):** MP3, WAV, FLAC, OGG, WEBM

**Requires ffmpeg:** M4A, MP4, MOV, AAC, WMA, OPUS

## License

MIT License - See LICENSE file
