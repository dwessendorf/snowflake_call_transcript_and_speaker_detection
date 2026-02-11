# Migration Guide: SPCS to Model Registry

This guide covers migrating the speaker embedding service from manual SPCS deployment to Snowflake Model Registry with Real-time Inference.

## Overview

| Aspect | Old (Manual SPCS) | New (Model Registry) |
|--------|-------------------|----------------------|
| Container | Custom Dockerfile | Auto-generated |
| Deployment | Manual `CREATE SERVICE` | `model.create_service()` |
| Scaling | Manual config | Auto-scaling |
| Updates | Rebuild & redeploy | `log_model()` new version |
| Monitoring | Custom logging | Built-in observability |
| GPU Support | Manual spec | `gpu_requests` parameter |

## Prerequisites

1. **Python packages**:
   ```bash
   pip install snowflake-ml-python>=1.25.0 snowflake-snowpark-python
   ```

2. **Snowflake privileges**:
   - `CREATE MODEL` on schema
   - `USAGE` on compute pool (or use system pools)
   - `BIND SERVICE ENDPOINT` on account

## Migration Steps

### Step 1: Run Infrastructure Setup

```sql
-- Execute the new setup script
-- This creates network rules and external access integration
@sql/04_setup_model_registry.sql
```

### Step 2: Register and Deploy Model

```bash
# Set your connection
export SNOWFLAKE_CONNECTION_NAME=your_connection

# Run the deployment script
cd spcs_service
python speaker_model_registry.py
```

Or manually in Python:

```python
from snowflake.snowpark import Session
from speaker_model_registry import register_model, deploy_service

session = Session.builder.config("connection_name", "your_conn").create()

# Register model
mv = register_model(
    session=session,
    model_name="SPEAKER_EMBEDDING",
    version_name="v1",
    database="CALL_TRANSCRIPTS_DB",
    schema="TRANSCRIPTS"
)

# Deploy service
deploy_service(
    model_version=mv,
    service_name="SPEAKER_EMBEDDING_SERVICE",
    compute_pool="SYSTEM_COMPUTE_POOL_GPU",
    gpu_requests="1",
    max_instances=3
)

session.close()
```

### Step 3: Create Service Functions

```sql
-- Execute the new functions script
@sql/05_create_functions_model_registry.sql
```

### Step 4: Verify Deployment

```sql
-- Check model exists
SHOW MODELS IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- Check service endpoints
SHOW ENDPOINTS IN SERVICE SPEAKER_EMBEDDING_SERVICE;

-- Test health
SELECT SPEAKER_EMBEDDING_HEALTH();

-- Test embedding extraction
SELECT SPEAKER_EMBEDDING_URL(
    GET_PRESIGNED_URL('@CALL_RECORDINGS', 'test.mp3', 3600),
    0.0,
    30.0
);
```

### Step 5: Update Streamlit App (Optional)

The Streamlit app uses stored procedures that have the same names, so it should work automatically. Optionally, add the new helper:

```python
# In streamlit_app/app.py, add import:
from speaker_helper import (
    check_service_health,
    auto_identify_call_speakers
)

# Use for health checks:
is_healthy, status = check_service_health(session)
```

### Step 6: Clean Up Old Service

After verifying the new service works:

```sql
-- Drop old SPCS service
DROP SERVICE IF EXISTS SPEAKER_IDENTIFICATION_SERVICE;

-- Drop old compute pool (if dedicated)
DROP COMPUTE POOL IF EXISTS SPEAKER_IDENTIFICATION_POOL;

-- Drop old image repository
DROP IMAGE REPOSITORY IF EXISTS SPEAKER_IDENTIFICATION_REPO;

-- Drop old network rules (if not shared)
DROP NETWORK RULE IF EXISTS HF_NETWORK_RULE;
DROP NETWORK RULE IF EXISTS S3_NETWORK_RULE;

-- Drop old external access integrations
DROP EXTERNAL ACCESS INTEGRATION IF EXISTS CALL_TRANSCRIPTS_DB_HF_ACCESS;
DROP EXTERNAL ACCESS INTEGRATION IF EXISTS CALL_TRANSCRIPTS_DB_S3_ACCESS;
```

## API Endpoint Mapping

| Old Endpoint | New Endpoint | Notes |
|--------------|--------------|-------|
| `/extract-embedding` | `/extract-embedding` | Same - base64 audio |
| `/extract-embedding-url` | `/extract-embedding-url` | Same - presigned URL |
| `/extract-embedding-b64` | `/extract-embedding` | Consolidated |
| `/match` | `/batch-match` | Enhanced - batch support |
| `/health` | `/health` | Same |
| N/A | `/compute-similarity` | New - direct comparison |

## SQL Function Mapping

| Old Function | New Function | Notes |
|--------------|--------------|-------|
| `SPEAKER_EMBEDDING_URL()` | `SPEAKER_EMBEDDING_URL()` | Same signature |
| N/A | `SPEAKER_EMBEDDING_B64()` | New |
| N/A | `SPEAKER_SIMILARITY()` | New |
| N/A | `SPEAKER_BATCH_MATCH()` | New |
| N/A | `SPEAKER_EMBEDDING_HEALTH()` | New |

## Stored Procedure Mapping

| Old Procedure | New Procedure | Notes |
|---------------|---------------|-------|
| `EXTRACT_CONTRIBUTION_EMBEDDING()` | `EXTRACT_CONTRIBUTION_EMBEDDING()` | Same |
| `CREATE_SPEAKER_VOICEPRINT_FROM_CONTRIBUTION()` | `CREATE_SPEAKER_VOICEPRINT_FROM_CONTRIBUTION()` | Same |
| N/A | `AUTO_IDENTIFY_CALL_SPEAKERS()` | New |
| N/A | `STORE_CONTRIBUTION_EMBEDDING()` | New |

## Using the Python Client

### Within Snowflake (Streamlit/Notebooks)

```python
from speaker_client import SpeakerClient

session = get_active_session()
client = SpeakerClient(session)

# Extract embedding
result = client.extract_embedding_url(presigned_url, start_time, end_time)
if result.success:
    embedding = result.embedding

# Identify speaker
match = client.identify_speaker(embedding, threshold=0.75)
if match.matched:
    print(f"Identified: {match.speaker_name} ({match.confidence:.2f})")
```

### External REST API

```python
from speaker_client import SpeakerRestClient

client = SpeakerRestClient(
    endpoint="https://xxx-yyy.snowflakecomputing.app",
    token="your_pat_token"
)

# Extract embedding
result = client.extract_embedding_b64(audio_base64)

# Compare embeddings
similarity = client.compute_similarity(emb1, emb2)
```

## Troubleshooting

### Model Registration Fails

```
Error: Cannot create model - insufficient privileges
```

**Solution**: Grant `CREATE MODEL` privilege:
```sql
GRANT CREATE MODEL ON SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS TO ROLE your_role;
```

### Service Deployment Fails

```
Error: Compute pool not found
```

**Solution**: Use system compute pools or create a custom one:
```sql
-- Check available system pools
SHOW COMPUTE POOLS LIKE 'SYSTEM%';

-- Or create custom pool
CREATE COMPUTE POOL SPEAKER_MODEL_POOL
    MIN_NODES = 1
    MAX_NODES = 3
    INSTANCE_FAMILY = GPU_NV_S;
```

### Network Access Issues

```
Error: Failed to download audio from URL
```

**Solution**: Verify network rules include your S3/Azure endpoints:
```sql
-- Find your S3 endpoint
SELECT GET_PRESIGNED_URL('@CALL_RECORDINGS', 'test.mp3', 60);
-- Extract hostname and add to MODEL_REGISTRY_S3_RULE
```

### Model Loading Slow

First request may take 1-2 minutes as the model downloads from HuggingFace. Subsequent requests use cached model.

## File Structure After Migration

```
snowflake_call_transcript_and_speaker_detection/
├── sql/
│   ├── 01_setup_database.sql
│   ├── 02_create_tables.sql
│   ├── 03_create_stages.sql
│   ├── 04_setup_model_registry.sql      # NEW - Model Registry setup
│   ├── 04_setup_spcs.sql                # OLD - Keep for reference
│   ├── 05_create_functions_model_registry.sql  # NEW - Model Registry functions
│   ├── 05_create_functions.sql          # OLD - Keep for reference
│   └── 06_create_streamlit_app.sql
├── spcs_service/
│   ├── speaker_model_registry.py        # NEW - Model registration
│   ├── speaker_client.py                # NEW - Python client
│   ├── speaker_service.py               # OLD - Flask service (deprecated)
│   ├── Dockerfile                       # OLD - No longer needed
│   └── test_speaker_model_registry.py   # NEW - Tests
├── streamlit_app/
│   ├── app.py
│   └── speaker_helper.py                # NEW - Streamlit helpers
└── docs/
    └── MIGRATION_MODEL_REGISTRY.md      # This file
```

## Benefits of Migration

1. **No Docker Management**: Snowflake builds containers automatically
2. **Auto-scaling**: Service scales based on load
3. **Simplified Updates**: Just `log_model()` a new version
4. **Traffic Splitting**: A/B test model versions
5. **Built-in Observability**: Metrics and logging included
6. **GPU Optimization**: Easy GPU allocation with `gpu_requests`
7. **Version Control**: Native model versioning in registry
