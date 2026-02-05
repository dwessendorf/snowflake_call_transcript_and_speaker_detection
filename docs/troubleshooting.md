# Troubleshooting Guide

## Deployment Issues

### 1. Docker Registry Authentication Fails

**Symptoms:**
- `unauthorized` error when pushing images
- `AUTHENTICATION_FAIL` errors

**Solutions:**

1. **Get your registry URL first**:
   ```sql
   SHOW IMAGE REPOSITORIES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
   ```

2. **Login with username and password/PAT**:
   ```bash
   # Using password interactively
   docker login <account>.registry.snowflakecomputing.com -u <username>
   
   # Using PAT token (non-interactive)
   echo '<pat_token>' | docker login <account>.registry.snowflakecomputing.com -u <username> --password-stdin
   ```

3. **If using Snowflake CLI with key-pair auth**, you need JWT authenticator:
   ```toml
   # In ~/.snowflake/connections.toml
   [my_connection]
   authenticator = "SNOWFLAKE_JWT"
   private_key_path = "/path/to/rsa_key.p8"
   ```

### 2. "Image Not Found" When Creating SPCS Service

**Symptoms:**
- Error: `Image /database/schema/repo/image:tag not found`
- Service creation fails

**Cause:** The Docker image hasn't been pushed to the Snowflake registry yet.

**Solutions:**

1. **Build with correct platform**:
   ```bash
   docker build --platform linux/amd64 -t speaker-identification:v1 .
   ```

2. **Tag with full registry path**:
   ```bash
   docker tag speaker-identification:v1 \
     <account>.registry.snowflakecomputing.com/call_transcripts_db/transcripts/speaker_identification_repo/speaker-identification:v1
   ```

3. **Push the image**:
   ```bash
   docker push <account>.registry.snowflakecomputing.com/call_transcripts_db/transcripts/speaker_identification_repo/speaker-identification:v1
   ```

4. **Verify upload**:
   ```sql
   CALL SYSTEM$REGISTRY_LIST_IMAGES('/call_transcripts_db/transcripts/speaker_identification_repo');
   ```

### 3. External Access Integration Errors

**Symptoms:**
- `Network rule 'DATABASE.SCHEMA.RULE' does not exist`
- `Database 'X' does not exist or not authorized`

**Cause:** Network rules must exist before creating external access integrations. If the database containing the network rules was dropped, integrations referencing them will fail.

**Solutions:**

1. **Create network rules first**:
   ```sql
   -- HuggingFace access
   CREATE OR REPLACE NETWORK RULE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.HF_NETWORK_RULE
       TYPE = HOST_PORT
       MODE = EGRESS
       VALUE_LIST = ('huggingface.co:443', 'cdn-lfs.huggingface.co:443', 'cdn-lfs-us-1.huggingface.co:443');
   
   -- S3 access for presigned URLs (update region as needed)
   CREATE OR REPLACE NETWORK RULE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.S3_NETWORK_RULE
       TYPE = HOST_PORT
       MODE = EGRESS
       VALUE_LIST = (
           'sfc-eu-ds1-38-customer-stage.s3.eu-central-1.amazonaws.com:443',
           'sfc-eu-ds1-38-customer-stage.s3.amazonaws.com:443'
       );
   ```

2. **Then create integrations**:
   ```sql
   CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION CALL_TRANSCRIPTS_DB_HF_ACCESS
       ALLOWED_NETWORK_RULES = (CALL_TRANSCRIPTS_DB.TRANSCRIPTS.HF_NETWORK_RULE)
       ENABLED = TRUE;
   
   CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION CALL_TRANSCRIPTS_DB_S3_ACCESS
       ALLOWED_NETWORK_RULES = (CALL_TRANSCRIPTS_DB.TRANSCRIPTS.S3_NETWORK_RULE)
       ENABLED = TRUE;
   ```

### 4. Compute Pool Not Starting

**Symptoms:**
- Compute pool stuck in `STARTING` or `SUSPENDED`
- Service won't start

**Solutions:**

```sql
-- Check compute pool status
DESCRIBE COMPUTE POOL SPEAKER_IDENTIFICATION_POOL;

-- Resume if suspended
ALTER COMPUTE POOL SPEAKER_IDENTIFICATION_POOL RESUME;

-- If stuck, try recreating
DROP COMPUTE POOL IF EXISTS SPEAKER_IDENTIFICATION_POOL;
CREATE COMPUTE POOL SPEAKER_IDENTIFICATION_POOL
    MIN_NODES = 1
    MAX_NODES = 3
    INSTANCE_FAMILY = CPU_X64_S
    AUTO_RESUME = TRUE
    AUTO_SUSPEND_SECS = 600;
```

### 5. Service Fails to Start After Image Push

**Symptoms:**
- Service status shows `FAILED` or `PENDING`
- Container keeps restarting

**Solutions:**

1. **Check service logs**:
   ```sql
   CALL SYSTEM$GET_SERVICE_LOGS('CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_IDENTIFICATION_SERVICE', '0', 'speaker-service', 100);
   ```

2. **Verify external access integrations are attached**:
   ```sql
   DESCRIBE SERVICE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_IDENTIFICATION_SERVICE;
   -- Check 'external_access_integrations' column
   ```

3. **Recreate service with correct integrations**:
   ```sql
   DROP SERVICE IF EXISTS CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_IDENTIFICATION_SERVICE;
   -- Then recreate with CREATE SERVICE command from README
   ```

## Runtime Issues

### 6. SPCS Service Not Responding

**Symptoms:**
- Embedding extraction fails with connection errors
- Service status shows "SUSPENDED" or "FAILED"

**Solutions:**

```sql
-- Check service status
DESCRIBE SERVICE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_IDENTIFICATION_SERVICE;

-- Check compute pool status
DESCRIBE COMPUTE POOL SPEAKER_IDENTIFICATION_POOL;

-- Resume service if suspended
ALTER SERVICE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_IDENTIFICATION_SERVICE RESUME;

-- Check service logs
CALL SYSTEM$GET_SERVICE_LOGS('CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_IDENTIFICATION_SERVICE', '0', 'speaker-service', 100);
```

### 7. Audio Conversion Fails

**Symptoms:**
- Error: "ffmpeg not found" or similar
- M4A/MP4 files fail to upload

**Solutions:**

1. Install ffmpeg:
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # Windows (with Chocolatey)
   choco install ffmpeg
   ```

2. Verify installation:
   ```bash
   ffmpeg -version
   ```

### 8. Embedding Extraction Fails

**Symptoms:**
- "No embedding in response" error
- HTTP errors from service function

**Solutions:**

1. **Check network rules** - Ensure S3 access is allowed:
   ```sql
   DESCRIBE NETWORK RULE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.S3_NETWORK_RULE;
   ```

2. **Find your S3 endpoint** by generating a presigned URL:
   ```sql
   SELECT GET_PRESIGNED_URL('@CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALL_RECORDINGS', 'test.mp3', 60);
   -- Extract the hostname (e.g., sfc-xxx.s3.region.amazonaws.com)
   ```

3. **Update network rule** with correct S3 endpoint:
   ```sql
   CREATE OR REPLACE NETWORK RULE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.S3_NETWORK_RULE
       TYPE = HOST_PORT
       MODE = EGRESS
       VALUE_LIST = ('<your-s3-endpoint>:443');
   ```

4. **Test the service function directly**:
   ```sql
   -- Get a presigned URL
   SELECT GET_PRESIGNED_URL('@CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALL_RECORDINGS', 'your_file.mp3', 3600) as url;
   
   -- Test embedding extraction (use the URL from above)
   SELECT CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_EMBEDDING_URL('<presigned_url>', 0.0, 30.0);
   ```

### 9. Streamlit App Shows Old Version

**Symptoms:**
- Changes to app.py not reflected
- Old UI elements still visible

**Solutions:**

1. **Re-upload the app file**:
   ```sql
   REMOVE @CALL_TRANSCRIPTS_DB.TRANSCRIPTS.STREAMLIT_APPS/speaker_classification_app/app.py;
   PUT file:///path/to/app.py @CALL_TRANSCRIPTS_DB.TRANSCRIPTS.STREAMLIT_APPS/speaker_classification_app/ OVERWRITE=TRUE AUTO_COMPRESS=FALSE;
   ```

2. **Hard refresh in browser**: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)

3. **Clear Snowflake cache**: Close and reopen the Streamlit app in Snowsight

### 10. Transcription Stuck in "Processing"

**Symptoms:**
- Call stays in "processing" status
- No contributions appear

**Solutions:**

1. **Check if AI_TRANSCRIBE completed**:
   ```sql
   SELECT transcription_status, transcription_path 
   FROM CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS 
   WHERE call_id = '<your_call_id>';
   ```

2. **Manually trigger reprocessing** (if needed):
   ```sql
   UPDATE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS 
   SET transcription_status = 'pending' 
   WHERE call_id = '<your_call_id>';
   ```

### 11. Permission Errors

**Symptoms:**
- "Insufficient privileges" errors
- Unable to create objects

**Solutions:**

```sql
-- Grant to your role
GRANT USAGE ON DATABASE CALL_TRANSCRIPTS_DB TO ROLE <your_role>;
GRANT USAGE ON SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS TO ROLE <your_role>;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS TO ROLE <your_role>;
GRANT ALL PRIVILEGES ON ALL STAGES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS TO ROLE <your_role>;
GRANT USAGE ON WAREHOUSE CALL_TRANSCRIPTS_WH TO ROLE <your_role>;
```

### 12. CLI Connection Issues

**Symptoms:**
- "Connection refused" or authentication errors
- Unable to connect to Snowflake

**Solutions:**

1. **Verify connection settings** in `cli/config.py`

2. **Test connection**:
   ```bash
   # Using snowsql
   snowsql -c <connection_name>
   
   # Or test with Python
   python -c "from cli.snowflake_client import get_connection; print(get_connection())"
   ```

3. **Check for network/firewall issues**

## Getting Help

If you encounter issues not covered here:

1. Check service logs: `CALL SYSTEM$GET_SERVICE_LOGS(...)`
2. Check Snowflake documentation for SPCS
3. Review the error message carefully for hints

## Useful Diagnostic Queries

```sql
-- Check all services
SHOW SERVICES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- Check compute pools
SHOW COMPUTE POOLS;

-- Check external access integrations
SHOW EXTERNAL ACCESS INTEGRATIONS;

-- Check network rules
SHOW NETWORK RULES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- Check recent calls
SELECT call_id, title, transcription_status, classification_status 
FROM CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS 
ORDER BY created_at DESC 
LIMIT 10;

-- Check contribution counts
SELECT c.title, COUNT(cc.contribution_id) as contributions
FROM CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS c
LEFT JOIN CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALL_CONTRIBUTIONS cc ON c.call_id = cc.call_id
GROUP BY c.call_id, c.title;

-- Check speaker assignments
SELECT 
    c.title,
    cc.diarization_label,
    s.display_name,
    cc.classification_status
FROM CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALL_CONTRIBUTIONS cc
JOIN CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS c ON cc.call_id = c.call_id
LEFT JOIN CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKERS s ON cc.identified_speaker_id = s.speaker_id
ORDER BY c.created_at DESC, cc.segment_number;

-- Test service health
SHOW ENDPOINTS IN SERVICE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_IDENTIFICATION_SERVICE;
```

## Complete Re-deployment Steps

If you need to completely redeploy from scratch:

```sql
-- 1. Stop and drop existing service
ALTER SERVICE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_IDENTIFICATION_SERVICE SUSPEND;
DROP SERVICE IF EXISTS CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_IDENTIFICATION_SERVICE;

-- 2. Drop the database (WARNING: deletes all data!)
DROP DATABASE IF EXISTS CALL_TRANSCRIPTS_DB;

-- 3. Run SQL scripts 01-06 in order to recreate everything

-- 4. Rebuild and push Docker image (required after dropping database)
-- See README.md Step 4

-- 5. Create the service
-- See README.md Step 5
```
