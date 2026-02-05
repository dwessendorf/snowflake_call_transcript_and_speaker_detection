# Snowflake Call Transcript and Speaker Detection

A complete solution for call/meeting transcription with automatic speaker diarization and identification, built on Snowflake.

## Features

- **Audio Upload & Transcription**: Upload call recordings (MP3, M4A, WAV, etc.) and get automatic transcription using Snowflake's AI_TRANSCRIBE
- **Speaker Diarization**: Automatic detection of different speakers in the call
- **Speaker Identification**: Match speakers to known profiles using voice embeddings (ECAPA-TDNN model via Snowflake Container Services)
- **Streamlit UI**: Web-based interface for speaker classification and management
- **CLI Tool**: Command-line interface for uploading calls and monitoring progress

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Snowflake Account                           │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │   Stages     │   │    Tables    │   │   SPCS Service       │    │
│  │              │   │              │   │                      │    │
│  │ - RECORDINGS │   │ - CALLS      │   │ Speaker ID Service   │    │
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
│ call-cli        │
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
cd snowflake_call_transcript_and_speaker_detection
```

### Step 2: Configure Snowflake Connection

Create a Snowflake connection configuration. You can use either:

**Option A: Snowflake CLI config (~/.snowflake/connections.toml)**
```toml
[my_connection]
account = "<account_identifier>"
user = "<username>"
password = "<password>"
warehouse = "CALL_TRANSCRIPTS_WH"
database = "CALL_TRANSCRIPTS_DB"
schema = "TRANSCRIPTS"
role = "ACCOUNTADMIN"
```

**Option B: SnowSQL config (~/.snowsql/config)**
```ini
[connections.my_connection]
accountname = <account_identifier>
username = <username>
password = <password>
warehousename = CALL_TRANSCRIPTS_WH
dbname = CALL_TRANSCRIPTS_DB
schemaname = TRANSCRIPTS
```

**Option C: Environment variables**
```bash
export SNOWFLAKE_ACCOUNT=<account_identifier>
export SNOWFLAKE_USER=<username>
export SNOWFLAKE_PASSWORD=<password>
export SNOWFLAKE_WAREHOUSE=CALL_TRANSCRIPTS_WH
export SNOWFLAKE_DATABASE=CALL_TRANSCRIPTS_DB
export SNOWFLAKE_SCHEMA=TRANSCRIPTS
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

#### 4.1 Get Your Registry URL

First, get your Snowflake image registry URL:

```sql
SHOW IMAGE REPOSITORIES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
-- Note the 'repository_url' value, e.g.:
-- <account>.registry.snowflakecomputing.com/call_transcripts_db/transcripts/speaker_identification_repo
```

#### 4.2 Build the Docker Image

```bash
cd spcs_service

# Build for linux/amd64 (required for SPCS)
docker build --platform linux/amd64 -t speaker-identification:v1 .
```

#### 4.3 Login to Snowflake Registry

**Option A: Using Snowflake CLI (if using JWT auth)**
```bash
snow spcs image-registry login --connection my_connection
```

**Option B: Using username/password or PAT token**
```bash
# Get your registry URL from Step 4.1
REGISTRY_URL="<account>.registry.snowflakecomputing.com"

# Login with your Snowflake credentials
docker login $REGISTRY_URL -u <username>
# Enter your password or PAT token when prompted
```

**Option C: Using PAT token directly**
```bash
echo '<your_pat_token>' | docker login $REGISTRY_URL -u <username> --password-stdin
```

#### 4.4 Tag and Push the Image

```bash
# Replace <account> with your account identifier
REGISTRY_URL="<account>.registry.snowflakecomputing.com"

# Tag the image
docker tag speaker-identification:v1 \
  $REGISTRY_URL/call_transcripts_db/transcripts/speaker_identification_repo/speaker-identification:v1

# Push the image
docker push $REGISTRY_URL/call_transcripts_db/transcripts/speaker_identification_repo/speaker-identification:v1
```

#### 4.5 Verify the Image Upload

```sql
-- List images in the repository
CALL SYSTEM$REGISTRY_LIST_IMAGES('/call_transcripts_db/transcripts/speaker_identification_repo');
```

### Step 5: Create the SPCS Service

After the image is pushed, the service can be created. If you ran `04_setup_spcs.sql` before pushing the image, you'll need to create the service manually:

```sql
CREATE SERVICE IF NOT EXISTS CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_IDENTIFICATION_SERVICE
    IN COMPUTE POOL SPEAKER_IDENTIFICATION_POOL
    MIN_INSTANCES = 1
    MAX_INSTANCES = 3
    MIN_READY_INSTANCES = 1
    AUTO_RESUME = TRUE
    EXTERNAL_ACCESS_INTEGRATIONS = (CALL_TRANSCRIPTS_DB_HF_ACCESS, CALL_TRANSCRIPTS_DB_S3_ACCESS)
    COMMENT = 'Speaker identification service using ECAPA-TDNN model'
    FROM SPECIFICATION $$
    spec:
      containers:
      - name: speaker-service
        image: /call_transcripts_db/transcripts/speaker_identification_repo/speaker-identification:v1
        env:
          SNOWFLAKE_DATABASE: CALL_TRANSCRIPTS_DB
          SNOWFLAKE_SCHEMA: TRANSCRIPTS
        resources:
          limits:
            memory: 8G
            cpu: 3
          requests:
            memory: 4G
            cpu: 2
      endpoints:
      - name: speaker-api
        port: 8080
        public: true
    $$;
```

### Step 6: Install CLI

```bash
cd cli
pip install -r requirements.txt
pip install -e .
```

### Step 7: Update Configuration

Edit `cli/config.py` to match your Snowflake settings:

```python
SNOWFLAKE_CONNECTION = "my_connection"
DATABASE = "CALL_TRANSCRIPTS_DB"
SCHEMA = "TRANSCRIPTS"
WAREHOUSE = "CALL_TRANSCRIPTS_WH"
```

## Usage

### Upload a Call

```bash
# Upload and wait for transcription
call-cli upload "path/to/call.m4a" --watch

# Upload with custom title
call-cli upload "call.mp3" --title "Q4 Planning Session"
```

### Check Status

```bash
call-cli status
call-cli status <call_id>
```

### Export Transcript

```bash
call-cli export <call_id> --format markdown
call-cli export <call_id> --format json
```

### Classify Speakers (Web UI)

Open the Streamlit app in Snowsight:
1. Navigate to Streamlit Apps
2. Open "Speaker Classification App"
3. Select a call and assign speakers to voice segments

## Directory Structure

```
snowflake_call_transcript_and_speaker_detection/
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
│   ├── call_cli.py          # CLI entry point
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
| `SNOWFLAKE_DATABASE` | Database name | `CALL_TRANSCRIPTS_DB` |
| `SNOWFLAKE_SCHEMA` | Schema name | `TRANSCRIPTS` |
| `SNOWFLAKE_WAREHOUSE` | Warehouse name | `CALL_TRANSCRIPTS_WH` |

### Embedding Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `MIN_DURATION_FOR_EMBEDDING` | Minimum segment duration (seconds) for embedding extraction | `5.0` |
| `SIMILARITY_THRESHOLD` | Threshold for auto-matching speakers | `0.75` |

## Deployment Checklist

Before the solution is fully operational, verify the following:

- [ ] Database `CALL_TRANSCRIPTS_DB` exists
- [ ] Schema `TRANSCRIPTS` exists
- [ ] Warehouse `CALL_TRANSCRIPTS_WH` exists and is accessible
- [ ] All 6 tables created (SPEAKERS, CALLS, CALL_CONTRIBUTIONS, etc.)
- [ ] All 6 stages created (CALL_RECORDINGS, AUDIO_SNIPPETS, etc.)
- [ ] Image repository `SPEAKER_IDENTIFICATION_REPO` exists
- [ ] Docker image pushed to repository
- [ ] Compute pool `SPEAKER_IDENTIFICATION_POOL` is ACTIVE or IDLE
- [ ] External access integrations created (HF_ACCESS, S3_ACCESS)
- [ ] Network rules allow HuggingFace and S3 access
- [ ] SPCS service `SPEAKER_IDENTIFICATION_SERVICE` is RUNNING
- [ ] Service functions created (SPEAKER_EMBEDDING_URL, EXTRACT_CONTRIBUTION_EMBEDDING)
- [ ] Streamlit app uploaded and created

### Verification Commands

```sql
-- Check database objects
SHOW TABLES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
SHOW STAGES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
SHOW IMAGE REPOSITORIES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- Check SPCS components
SHOW COMPUTE POOLS LIKE 'SPEAKER%';
DESCRIBE SERVICE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_IDENTIFICATION_SERVICE;
SHOW ENDPOINTS IN SERVICE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_IDENTIFICATION_SERVICE;

-- Check functions
SHOW FUNCTIONS LIKE '%SPEAKER%' IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
SHOW PROCEDURES LIKE '%EMBEDDING%' IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- Check Streamlit app
SHOW STREAMLITS IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
```

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md) for common issues and solutions.

### Common Issues

1. **Docker registry login fails**: Use your Snowflake username with password or PAT token
2. **"Image not found" when creating service**: Push the Docker image before creating the service
3. **SPCS Service not responding**: Check compute pool status and service logs
4. **Audio conversion fails**: Ensure ffmpeg is installed
5. **Embedding extraction fails**: Verify network rules allow S3 access for presigned URLs
6. **External access integration errors**: Ensure network rules are created in the same schema

## Re-deployment / Migration

If you need to migrate to a new database or re-deploy:

1. **Stop existing services first**:
   ```sql
   ALTER SERVICE <old_service> SUSPEND;
   DROP SERVICE IF EXISTS <old_service>;
   ```

2. **Docker images are tied to repositories**: When you drop a database, all image repositories and their images are deleted. You must rebuild and push images to any new repository.

3. **External access integrations reference network rules by full path**: If you drop a database containing network rules, any integrations referencing them will fail. Recreate both the network rules and integrations.

4. **Update local connection configs**: After migration, update `~/.snowflake/connections.toml` or environment variables to point to the new database/schema.

## License

MIT License - see LICENSE file for details.
