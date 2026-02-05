# Troubleshooting Guide

## Common Issues and Solutions

### 1. SPCS Service Not Responding

**Symptoms:**
- Embedding extraction fails with connection errors
- Service status shows "SUSPENDED" or "FAILED"

**Solutions:**

```sql
-- Check service status
DESCRIBE SERVICE SPEAKER_IDENTIFICATION_SERVICE;

-- Check compute pool status
DESCRIBE COMPUTE POOL SPEAKER_IDENTIFICATION_POOL;

-- Resume service if suspended
ALTER SERVICE SPEAKER_IDENTIFICATION_SERVICE RESUME;

-- Check service logs
CALL SYSTEM$GET_SERVICE_LOGS('SPEAKER_IDENTIFICATION_SERVICE', '0', 'speaker-service', 100);
```

### 2. Audio Conversion Fails

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

### 3. Embedding Extraction Fails

**Symptoms:**
- "No embedding in response" error
- OAuth redirection errors

**Solutions:**

1. **Check network rules** - Ensure S3 access is allowed:
   ```sql
   DESCRIBE NETWORK RULE SPEAKER_SERVICE_NETWORK_RULE;
   ```

2. **Update network rule** with correct S3 endpoint:
   ```sql
   -- Get your S3 endpoint by generating a presigned URL
   SELECT GET_PRESIGNED_URL('@MEETING_RECORDINGS', 'test.mp3', 60);
   -- Extract the hostname and add to network rule
   ```

3. **Test the service function directly**:
   ```sql
   -- Get a presigned URL
   SELECT GET_PRESIGNED_URL('@MEETING_RECORDINGS', 'your_file.mp3', 3600) as url;
   
   -- Test embedding extraction (use the URL from above)
   SELECT SPEAKER_EMBEDDING_URL('<presigned_url>', 0.0, 30.0);
   ```

### 4. Streamlit App Shows Old Version

**Symptoms:**
- Changes to app.py not reflected
- Old UI elements still visible

**Solutions:**

1. **Re-upload the app file**:
   ```sql
   REMOVE @STREAMLIT_APPS/speaker_classification_app/app.py;
   -- Then upload new version
   ```

2. **Hard refresh in browser**: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)

3. **Clear Snowflake cache**: Close and reopen the Streamlit app in Snowsight

### 5. Transcription Stuck in "Processing"

**Symptoms:**
- Meeting stays in "processing" status
- No contributions appear

**Solutions:**

1. **Check if AI_TRANSCRIBE completed**:
   ```sql
   SELECT transcription_status, transcription_path 
   FROM MEETINGS 
   WHERE meeting_id = '<your_meeting_id>';
   ```

2. **Manually trigger reprocessing** (if needed):
   ```sql
   UPDATE MEETINGS 
   SET transcription_status = 'pending' 
   WHERE meeting_id = '<your_meeting_id>';
   ```

### 6. Permission Errors

**Symptoms:**
- "Insufficient privileges" errors
- Unable to create objects

**Solutions:**

1. **Grant necessary privileges**:
   ```sql
   -- Grant to your role
   GRANT USAGE ON DATABASE MEETING_AGENT_DB TO ROLE <your_role>;
   GRANT USAGE ON SCHEMA MEETING_AGENT_DB.MEETING_AGENT TO ROLE <your_role>;
   GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA MEETING_AGENT_DB.MEETING_AGENT TO ROLE <your_role>;
   GRANT READ, WRITE ON ALL STAGES IN SCHEMA MEETING_AGENT_DB.MEETING_AGENT TO ROLE <your_role>;
   ```

### 7. CLI Connection Issues

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
SHOW SERVICES IN SCHEMA MEETING_AGENT_DB.MEETING_AGENT;

-- Check compute pools
SHOW COMPUTE POOLS;

-- Check recent meetings
SELECT meeting_id, title, transcription_status, classification_status 
FROM MEETINGS 
ORDER BY created_at DESC 
LIMIT 10;

-- Check contribution counts
SELECT m.title, COUNT(c.contribution_id) as contributions
FROM MEETINGS m
LEFT JOIN MEETING_CONTRIBUTIONS c ON m.meeting_id = c.meeting_id
GROUP BY m.meeting_id, m.title;

-- Check speaker assignments
SELECT 
    m.title,
    c.diarization_label,
    s.display_name,
    c.classification_status
FROM MEETING_CONTRIBUTIONS c
JOIN MEETINGS m ON c.meeting_id = m.meeting_id
LEFT JOIN SPEAKERS s ON c.identified_speaker_id = s.speaker_id
ORDER BY m.created_at DESC, c.segment_number;
```
