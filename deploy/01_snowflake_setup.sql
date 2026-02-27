-- ============================================================================
-- Call Transcript & Speaker Detection - Snowflake Setup
-- ============================================================================
-- 
-- This script sets up all Snowflake infrastructure for the solution.
-- Run this FIRST before deploying the speaker embedding model.
--
-- Requirements:
--   - ACCOUNTADMIN role (or equivalent privileges)
--   - Cortex AI functions enabled in your region
--   - For cross-region Cortex: ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';
--
-- ============================================================================

-- ============================================================================
-- PART 1: Database, Schema, and Warehouses
-- ============================================================================

CREATE DATABASE IF NOT EXISTS CALL_TRANSCRIPTS_DB
    COMMENT = 'Call transcription and speaker detection solution';

CREATE SCHEMA IF NOT EXISTS CALL_TRANSCRIPTS_DB.TRANSCRIPTS
    COMMENT = 'Main schema for call transcription objects';

USE SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- Main warehouse for transcription operations
CREATE WAREHOUSE IF NOT EXISTS CALL_TRANSCRIPTS_WH
    WAREHOUSE_SIZE = 'SMALL'
    AUTO_SUSPEND = 300
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for call transcription operations';

-- Warehouse for Streamlit app
CREATE WAREHOUSE IF NOT EXISTS STREAMLIT_APP_WH
    WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for Streamlit speaker assignment app';

-- Grant usage
GRANT USAGE ON DATABASE CALL_TRANSCRIPTS_DB TO ROLE ACCOUNTADMIN;
GRANT USAGE ON SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS TO ROLE ACCOUNTADMIN;
GRANT USAGE ON WAREHOUSE CALL_TRANSCRIPTS_WH TO ROLE ACCOUNTADMIN;
GRANT USAGE ON WAREHOUSE STREAMLIT_APP_WH TO ROLE ACCOUNTADMIN;

USE WAREHOUSE CALL_TRANSCRIPTS_WH;

-- ============================================================================
-- PART 2: Tables
-- ============================================================================

-- Speakers registry
CREATE TABLE IF NOT EXISTS SPEAKERS (
    SPEAKER_ID VARCHAR(50) NOT NULL PRIMARY KEY,
    DISPLAY_NAME VARCHAR(200) NOT NULL,
    EMAIL VARCHAR(200),
    DEPARTMENT VARCHAR(100),
    COMPANY VARCHAR(200),
    NOTES VARCHAR(2000),
    IS_INTERNAL BOOLEAN DEFAULT FALSE,
    MEETING_COUNT NUMBER DEFAULT 0,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    CREATED_BY VARCHAR(100),
    METADATA VARIANT
);

-- Calls metadata
CREATE TABLE IF NOT EXISTS CALLS (
    CALL_ID VARCHAR(50) NOT NULL PRIMARY KEY,
    TITLE VARCHAR(500),
    CALL_DATE DATE,
    CALL_TIME TIME,
    DURATION_MINUTES NUMBER,
    CALL_TYPE VARCHAR(100),
    RECORDING_PATH VARCHAR(500),
    TRANSCRIPTION_PATH VARCHAR(500),
    TRANSCRIPTION_STATUS VARCHAR(50) DEFAULT 'pending',
    CLASSIFICATION_STATUS VARCHAR(50) DEFAULT 'pending',
    TOTAL_SPEAKERS NUMBER,
    IDENTIFIED_SPEAKERS NUMBER DEFAULT 0,
    UNIDENTIFIED_SPEAKERS NUMBER DEFAULT 0,
    LANGUAGE VARCHAR(10),
    SUMMARY VARCHAR(5000),
    ACTION_ITEMS VARIANT,
    PARTICIPANTS VARIANT,
    TAGS VARIANT,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    CREATED_BY VARCHAR(100),
    METADATA VARIANT
);

-- Call contributions (speech segments)
CREATE TABLE IF NOT EXISTS CALL_CONTRIBUTIONS (
    CONTRIBUTION_ID VARCHAR(50) NOT NULL PRIMARY KEY,
    CALL_ID VARCHAR(50) NOT NULL REFERENCES CALLS(CALL_ID),
    SEGMENT_NUMBER NUMBER NOT NULL,
    DIARIZATION_LABEL VARCHAR(20),
    IDENTIFIED_SPEAKER_ID VARCHAR(50) REFERENCES SPEAKERS(SPEAKER_ID),
    IDENTIFICATION_METHOD VARCHAR(50),
    IDENTIFICATION_CONFIDENCE FLOAT,
    MATCHED_PROFILE_ID VARCHAR(50),
    AUDIO_SNIPPET_PATH VARCHAR(500),
    EMBEDDING VECTOR(FLOAT, 192),
    EMBEDDING_STATUS VARCHAR(20),  -- SUCCESS, FAILED, TOO_SHORT, NULL=pending
    TEXT_CONTENT VARCHAR(100000),
    START_TIME_SECONDS FLOAT,
    END_TIME_SECONDS FLOAT,
    DURATION_SECONDS FLOAT,
    WORD_COUNT NUMBER,
    CLASSIFICATION_STATUS VARCHAR(50) DEFAULT 'pending',
    CLASSIFICATION_CONFIDENCE FLOAT,
    REVIEWED_AT TIMESTAMP_NTZ,
    REVIEWED_BY VARCHAR(100),
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    METADATA VARIANT
);

-- Speaker voiceprints (voice embeddings for identification)
CREATE TABLE IF NOT EXISTS SPEAKER_VOICEPRINTS (
    SPEAKER_ID VARCHAR(50) NOT NULL PRIMARY KEY,
    SPEAKER_NAME VARCHAR(200) NOT NULL,
    EMAIL VARCHAR(200),
    DEPARTMENT VARCHAR(100),
    EMBEDDING VECTOR(FLOAT, 192),
    SAMPLE_AUDIO_PATH VARCHAR(500) NOT NULL,
    SAMPLE_DURATION_SECONDS FLOAT,
    ENROLLMENT_DATE TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    LAST_UPDATED TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    QUALITY_SCORE FLOAT,
    METADATA VARIANT
);

-- Speaker profiles (for future multi-profile support)
CREATE TABLE IF NOT EXISTS SPEAKER_PROFILES (
    PROFILE_ID VARCHAR(50) NOT NULL PRIMARY KEY,
    SPEAKER_ID VARCHAR(50) NOT NULL REFERENCES SPEAKERS(SPEAKER_ID),
    PROFILE_TYPE VARCHAR(50) DEFAULT 'voice',
    EMBEDDING VECTOR(FLOAT, 192),
    SAMPLE_AUDIO_PATH VARCHAR(500),
    SAMPLE_DURATION_SECONDS FLOAT,
    QUALITY_SCORE FLOAT,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    METADATA VARIANT
);

-- Classification queue (for manual review workflow)
CREATE TABLE IF NOT EXISTS CLASSIFICATION_QUEUE (
    QUEUE_ID VARCHAR(50) NOT NULL PRIMARY KEY,
    CALL_ID VARCHAR(50) NOT NULL REFERENCES CALLS(CALL_ID),
    CONTRIBUTION_ID VARCHAR(50) NOT NULL REFERENCES CALL_CONTRIBUTIONS(CONTRIBUTION_ID),
    DIARIZATION_LABEL VARCHAR(20),
    SUGGESTED_SPEAKER_ID VARCHAR(50),
    SUGGESTED_SPEAKER_NAME VARCHAR(200),
    SUGGESTION_CONFIDENCE FLOAT,
    ALTERNATIVE_SUGGESTIONS VARIANT,
    AUDIO_SNIPPET_PATH VARCHAR(500),
    SNIPPET_DURATION_SECONDS FLOAT,
    TEXT_PREVIEW VARCHAR(1000),
    STATUS VARCHAR(50) DEFAULT 'pending',
    PRIORITY NUMBER DEFAULT 5,
    ASSIGNED_TO VARCHAR(100),
    ASSIGNED_AT TIMESTAMP_NTZ,
    SELECTED_SPEAKER_ID VARCHAR(50),
    CREATED_NEW_SPEAKER BOOLEAN DEFAULT FALSE,
    NEW_PROFILE_CREATED BOOLEAN DEFAULT FALSE,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    COMPLETED_AT TIMESTAMP_NTZ,
    COMPLETED_BY VARCHAR(100),
    METADATA VARIANT
);

-- Voiceprint creation queue (async processing)
CREATE TABLE IF NOT EXISTS VOICEPRINT_QUEUE (
    QUEUE_ID VARCHAR NOT NULL DEFAULT UUID_STRING() PRIMARY KEY,
    SPEAKER_ID VARCHAR NOT NULL,
    CALL_ID VARCHAR NOT NULL,
    DIARIZATION_LABEL VARCHAR NOT NULL,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    STATUS VARCHAR DEFAULT 'pending'
);

-- Contribution embeddings (separate storage, for future use)
CREATE TABLE IF NOT EXISTS CONTRIBUTION_EMBEDDINGS (
    CONTRIBUTION_ID VARCHAR(50) NOT NULL PRIMARY KEY,
    CALL_ID VARCHAR(50) NOT NULL,
    EMBEDDING VECTOR(FLOAT, 192),
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_contributions_call ON CALL_CONTRIBUTIONS(CALL_ID);
CREATE INDEX IF NOT EXISTS idx_contributions_speaker ON CALL_CONTRIBUTIONS(IDENTIFIED_SPEAKER_ID);
CREATE INDEX IF NOT EXISTS idx_queue_call ON CLASSIFICATION_QUEUE(CALL_ID);
CREATE INDEX IF NOT EXISTS idx_queue_status ON CLASSIFICATION_QUEUE(STATUS);

-- ============================================================================
-- PART 3: Stages
-- ============================================================================

-- Call recordings stage (audio/video files)
CREATE STAGE IF NOT EXISTS CALL_RECORDINGS
    DIRECTORY = (ENABLE = TRUE)
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    COMMENT = 'Storage for call audio/video recordings';

-- Audio snippets stage (extracted segments)
CREATE STAGE IF NOT EXISTS AUDIO_SNIPPETS
    DIRECTORY = (ENABLE = TRUE)
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    COMMENT = 'Audio snippets of individual speech segments';

-- ============================================================================
-- PART 4: Network Rules for External Access (GPU Service)
-- ============================================================================
-- Required for Model Registry service to download audio from presigned URLs
-- and to download HuggingFace models

-- HuggingFace access for model downloads
CREATE OR REPLACE NETWORK RULE MODEL_REGISTRY_HF_RULE
    TYPE = HOST_PORT
    MODE = EGRESS
    VALUE_LIST = (
        'huggingface.co:443',
        'cdn-lfs.huggingface.co:443',
        'cdn-lfs-us-1.huggingface.co:443',
        'cdn-lfs.hf.co:443'
    );

-- AWS S3 access for presigned URLs
CREATE OR REPLACE NETWORK RULE MODEL_REGISTRY_S3_RULE
    TYPE = HOST_PORT
    MODE = EGRESS
    VALUE_LIST = (
        '*.s3.amazonaws.com:443',
        '*.s3.us-east-1.amazonaws.com:443',
        '*.s3.us-west-2.amazonaws.com:443',
        '*.s3.eu-central-1.amazonaws.com:443',
        '*.s3.eu-west-1.amazonaws.com:443',
        '*.s3.ap-northeast-1.amazonaws.com:443',
        '*.s3.ap-southeast-1.amazonaws.com:443'
    );

-- Azure Blob access for presigned URLs
CREATE OR REPLACE NETWORK RULE MODEL_REGISTRY_AZURE_RULE
    TYPE = HOST_PORT
    MODE = EGRESS
    VALUE_LIST = (
        '*.blob.core.windows.net:443',
        '*.blob.storage.azure.net:443'
    );

-- Combined external access integration
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION MODEL_REGISTRY_EXTERNAL_ACCESS
    ALLOWED_NETWORK_RULES = (
        MODEL_REGISTRY_HF_RULE,
        MODEL_REGISTRY_S3_RULE,
        MODEL_REGISTRY_AZURE_RULE
    )
    ENABLED = TRUE
    COMMENT = 'External access for Model Registry speaker embedding service';

-- ============================================================================
-- PART 5: Compute Pool for GPU Service
-- ============================================================================

CREATE COMPUTE POOL IF NOT EXISTS SPEAKER_IDENTIFICATION_POOL
    MIN_NODES = 1
    MAX_NODES = 1
    INSTANCE_FAMILY = GPU_NV_S
    AUTO_SUSPEND_SECS = 300
    AUTO_RESUME = TRUE
    COMMENT = 'GPU compute pool for speaker embedding extraction';

-- ============================================================================
-- PART 6: Transcription Procedure (Cortex AI_TRANSCRIBE)
-- ============================================================================

CREATE OR REPLACE PROCEDURE TRANSCRIBE_CALL(P_CALL_ID VARCHAR)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'transcribe_call'
EXECUTE AS OWNER
AS $$
import json
import uuid
from datetime import datetime

def transcribe_call(session, p_call_id):
    """Transcribe a call using Cortex AI_TRANSCRIBE with speaker diarization"""
    try:
        # Get call info
        call_info = session.sql(f"""
            SELECT RECORDING_PATH, TRANSCRIPTION_STATUS, DURATION_MINUTES
            FROM CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS
            WHERE CALL_ID = '{p_call_id}'
        """).collect()
        
        if not call_info:
            return {"status": "error", "message": f"Call not found: {p_call_id}"}
        
        recording_path = call_info[0]['RECORDING_PATH']
        current_status = call_info[0]['TRANSCRIPTION_STATUS']
        
        if current_status == 'completed':
            return {"status": "skipped", "message": "Call already transcribed"}
        
        if recording_path is None:
            return {"status": "error", "message": f"Recording path is null for call: {p_call_id}"}
            
        # Extract stage and file from recording path
        path = recording_path.lstrip('@')
        parts = path.split('/')
        if len(parts) >= 2:
            stage_name = '@' + '/'.join(parts[:-1])
            file_name = parts[-1]
        else:
            return {"status": "error", "message": f"Invalid recording path format: {recording_path}"}
        
        # Update status to processing
        session.sql(f"""
            UPDATE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS
            SET TRANSCRIPTION_STATUS = 'processing',
                UPDATED_AT = CURRENT_TIMESTAMP()
            WHERE CALL_ID = '{p_call_id}'
        """).collect()
        
        # Call AI_TRANSCRIBE with speaker diarization
        transcription_result = session.sql(f"""
            SELECT SNOWFLAKE.CORTEX.AI_TRANSCRIBE(
                TO_FILE('{stage_name}', '{file_name}'),
                {{'timestamp_granularity': 'speaker'}}
            ) as result
        """).collect()
        
        if not transcription_result:
            session.sql(f"""
                UPDATE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS
                SET TRANSCRIPTION_STATUS = 'failed'
                WHERE CALL_ID = '{p_call_id}'
            """).collect()
            return {"status": "error", "message": "No result from AI_TRANSCRIBE"}
        
        result = transcription_result[0]['RESULT']
        if isinstance(result, str):
            result = json.loads(result)
        
        segments = result.get('segments', [])
        audio_duration = result.get('audio_duration', 0)
        
        if not segments:
            return {"status": "error", "message": "No segments in transcription"}
        
        # Delete existing contributions for this call
        session.sql(f"""
            DELETE FROM CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALL_CONTRIBUTIONS
            WHERE CALL_ID = '{p_call_id}'
        """).collect()
        
        # Insert contributions from segments
        contributions_added = 0
        unique_speakers = set()
        
        for idx, segment in enumerate(segments):
            contrib_id = str(uuid.uuid4())
            speaker_label = segment.get('speaker_label', 'UNKNOWN')
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            text = segment.get('text', '').replace("'", "''")
            duration = end_time - start_time
            word_count = len(text.split())
            
            unique_speakers.add(speaker_label)
            
            session.sql(f"""
                INSERT INTO CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALL_CONTRIBUTIONS (
                    CONTRIBUTION_ID, CALL_ID, SEGMENT_NUMBER, DIARIZATION_LABEL,
                    START_TIME_SECONDS, END_TIME_SECONDS, DURATION_SECONDS,
                    TEXT_CONTENT, WORD_COUNT, CLASSIFICATION_STATUS, CREATED_AT
                ) VALUES (
                    '{contrib_id}', '{p_call_id}', {idx + 1}, '{speaker_label}',
                    {start_time}, {end_time}, {duration}, '{text}', {word_count},
                    'pending', CURRENT_TIMESTAMP()
                )
            """).collect()
            contributions_added += 1
        
        # Update call status
        duration_minutes = round(audio_duration / 60.0, 1)
        session.sql(f"""
            UPDATE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS
            SET TRANSCRIPTION_STATUS = 'completed',
                CLASSIFICATION_STATUS = 'pending',
                DURATION_MINUTES = {duration_minutes},
                TOTAL_SPEAKERS = {len(unique_speakers)},
                UPDATED_AT = CURRENT_TIMESTAMP()
            WHERE CALL_ID = '{p_call_id}'
        """).collect()
        
        return {
            "status": "success",
            "call_id": p_call_id,
            "audio_duration_seconds": audio_duration,
            "duration_minutes": duration_minutes,
            "num_segments": len(segments),
            "num_speakers": len(unique_speakers),
            "speakers": list(unique_speakers),
            "contributions_added": contributions_added
        }
        
    except Exception as e:
        try:
            session.sql(f"""
                UPDATE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS
                SET TRANSCRIPTION_STATUS = 'failed'
                WHERE CALL_ID = '{p_call_id}'
            """).collect()
        except:
            pass
        return {"status": "error", "message": str(e)}
$$;

-- ============================================================================
-- PART 7: Speaker Voiceprint Creation Procedure
-- ============================================================================

CREATE OR REPLACE PROCEDURE CREATE_SPEAKER_VOICEPRINT_FROM_CONTRIBUTION(
    P_SPEAKER_ID VARCHAR, 
    P_CALL_ID VARCHAR, 
    P_DIARIZATION_LABEL VARCHAR
)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'create_voiceprint'
EXECUTE AS OWNER
AS '
import json

def create_voiceprint(session, p_speaker_id, p_call_id, p_diarization_label):
    try:
        # Find a suitable contribution (at least 10 seconds) for this speaker
        contrib = session.sql(f"""
            SELECT cc.contribution_id, cc.start_time_seconds, cc.end_time_seconds,
                   cc.duration_seconds, c.recording_path
            FROM CALL_CONTRIBUTIONS cc
            JOIN CALLS c ON cc.call_id = c.call_id
            WHERE cc.call_id = ''{p_call_id}''
            AND cc.diarization_label = ''{p_diarization_label}''
            AND cc.duration_seconds >= 10
            ORDER BY cc.duration_seconds DESC
            LIMIT 1
        """).collect()
        
        if not contrib:
            return {"status": "error", "message": "No suitable contribution found (need >= 10s)"}
        
        contribution_id = contrib[0][''CONTRIBUTION_ID'']
        start_time = float(contrib[0][''START_TIME_SECONDS''])
        end_time = float(contrib[0][''END_TIME_SECONDS''])
        duration = float(contrib[0][''DURATION_SECONDS''])
        recording_path = contrib[0][''RECORDING_PATH'']
        
        # Parse the stage path to get presigned URL
        stage_path = recording_path.lstrip(''@'')
        if ''/'' in stage_path:
            parts = stage_path.split(''/'', 1)
            stage_name = ''@'' + parts[0]
            file_name = parts[1]
        else:
            return {"status": "error", "message": "Invalid recording path"}
        
        url_result = session.sql(f"""
            SELECT GET_PRESIGNED_URL(''{stage_name}'', ''{file_name}'', 3600) as url
        """).collect()
        
        if not url_result or not url_result[0][''URL'']:
            return {"status": "error", "message": "Could not get presigned URL"}
        
        presigned_url = url_result[0][''URL'']
        
        # Extract embedding from the audio segment
        result = session.sql(f"""
            SELECT SPEAKER_EMBEDDING_URL(''{presigned_url}'', {start_time}, {end_time}) as result
        """).collect()
        
        if not result or not result[0][''RESULT'']:
            return {"status": "error", "message": "No result from embedding service"}
        
        svc_result = result[0][''RESULT'']
        if isinstance(svc_result, str):
            svc_result = json.loads(svc_result)
        
        if svc_result.get(''status'') == ''error'':
            return {"status": "error", "message": svc_result.get(''error'', ''Unknown error'')}
        
        embedding = svc_result.get(''embedding'')
        if not embedding:
            return {"status": "error", "message": "No embedding returned"}
        
        if isinstance(embedding, str):
            embedding = json.loads(embedding)
        
        embedding_json = json.dumps(embedding)
        
        # Get speaker info
        speaker = session.sql(f"""
            SELECT display_name, email, department 
            FROM SPEAKERS 
            WHERE speaker_id = ''{p_speaker_id}''
        """).collect()
        
        if not speaker:
            return {"status": "error", "message": "Speaker not found"}
        
        speaker_name = speaker[0][''DISPLAY_NAME''].replace("''", "''''")
        email = (speaker[0][''EMAIL''] or '''').replace("''", "''''")
        department = (speaker[0][''DEPARTMENT''] or '''').replace("''", "''''")
        
        # Upsert the voiceprint
        session.sql(f"""
            MERGE INTO SPEAKER_VOICEPRINTS t
            USING (SELECT ''{p_speaker_id}'' as speaker_id) s
            ON t.speaker_id = s.speaker_id
            WHEN MATCHED THEN UPDATE SET
                speaker_name = ''{speaker_name}'',
                email = ''{email}'',
                department = ''{department}'',
                embedding = PARSE_JSON(''{embedding_json}'')::VECTOR(FLOAT, 192),
                sample_audio_path = ''{recording_path}'',
                sample_duration_seconds = {duration},
                last_updated = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT (
                speaker_id, speaker_name, email, department, embedding,
                sample_audio_path, sample_duration_seconds, enrollment_date, last_updated
            ) VALUES (
                ''{p_speaker_id}'', ''{speaker_name}'', ''{email}'', ''{department}'',
                PARSE_JSON(''{embedding_json}'')::VECTOR(FLOAT, 192),
                ''{recording_path}'', {duration}, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()
            )
        """).collect()
        
        return {
            "status": "success",
            "speaker_id": p_speaker_id,
            "speaker_name": speaker_name,
            "duration_seconds": duration,
            "contribution_id": contribution_id
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
';

-- ============================================================================
-- PART 8: Embedding Extraction Procedure (Bulk/Batch)
-- ============================================================================

CREATE OR REPLACE PROCEDURE EXTRACT_NEW_EMBEDDINGS(P_BATCH_SIZE NUMBER DEFAULT 100)
RETURNS VARIANT
LANGUAGE SQL
EXECUTE AS OWNER
AS '
DECLARE
    rows_updated INT := 0;
    total_updated INT := 0;
    batch_num INT := 0;
    pending_count INT;
    skipped_short INT := 0;
BEGIN
    -- First mark any new short contributions as skipped
    UPDATE CALL_CONTRIBUTIONS
    SET EMBEDDING_STATUS = ''TOO_SHORT''
    WHERE embedding IS NULL
    AND EMBEDDING_STATUS IS NULL
    AND (end_time_seconds - start_time_seconds) < 0.5;
    
    skipped_short := SQLROWCOUNT;

    LOOP
        batch_num := batch_num + 1;
        
        -- Only get contributions without embedding AND without skip status
        CREATE OR REPLACE TEMPORARY TABLE PENDING_BATCH AS
        SELECT 
            c.contribution_id,
            c.start_time_seconds,
            c.end_time_seconds,
            GET_PRESIGNED_URL(
                ''@CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALL_RECORDINGS'',
                SPLIT_PART(cl.recording_path, ''/'', -1),
                3600
            ) as presigned_url
        FROM CALL_CONTRIBUTIONS c
        JOIN CALLS cl ON c.call_id = cl.call_id
        WHERE c.embedding IS NULL
        AND c.EMBEDDING_STATUS IS NULL
        LIMIT :P_BATCH_SIZE;
        
        SELECT COUNT(*) INTO :pending_count FROM PENDING_BATCH;
        
        IF (pending_count = 0) THEN
            RETURN OBJECT_CONSTRUCT(
                ''status'', ''complete'', 
                ''total_updated'', total_updated, 
                ''batches'', batch_num - 1,
                ''skipped_short'', skipped_short
            );
        END IF;
        
        -- Extract embeddings in parallel (SQL handles parallelism)
        CREATE OR REPLACE TEMPORARY TABLE BATCH_RESULTS AS
        SELECT 
            contribution_id,
            SPEAKER_EMBEDDING_URL(presigned_url, start_time_seconds, end_time_seconds) as result
        FROM PENDING_BATCH;
        
        -- Update successful extractions
        UPDATE CALL_CONTRIBUTIONS c
        SET EMBEDDING = PARSE_JSON(r.result:embedding)::VECTOR(FLOAT, 192),
            EMBEDDING_STATUS = ''SUCCESS''
        FROM BATCH_RESULTS r
        WHERE c.contribution_id = r.contribution_id
        AND r.result:status = ''success''
        AND r.result:embedding IS NOT NULL;
        
        rows_updated := SQLROWCOUNT;
        total_updated := total_updated + rows_updated;
        
        -- Mark failed extractions
        UPDATE CALL_CONTRIBUTIONS c
        SET EMBEDDING_STATUS = ''FAILED''
        FROM BATCH_RESULTS r
        WHERE c.contribution_id = r.contribution_id
        AND (r.result:status != ''success'' OR r.result:embedding IS NULL)
        AND c.EMBEDDING_STATUS IS NULL;
        
    END LOOP;
END;
';

-- ============================================================================
-- PART 9: Background Tasks
-- ============================================================================

-- Task: Hourly embedding extraction for new contributions
CREATE OR REPLACE TASK EXTRACT_NEW_EMBEDDINGS_TASK
    WAREHOUSE = CALL_TRANSCRIPTS_WH
    SCHEDULE = 'USING CRON 0 * * * * UTC'
AS
    CALL EXTRACT_NEW_EMBEDDINGS(100);

-- Task: Housekeeping - update call status and speaker meeting counts (every 5 min)
CREATE OR REPLACE TASK SPEAKER_ASSIGNMENT_HOUSEKEEPING
    WAREHOUSE = CALL_TRANSCRIPTS_WH
    SCHEDULE = 'USING CRON */5 * * * * UTC'
AS
BEGIN
    -- 1. Update call status for all calls based on actual data
    UPDATE CALLS c
    SET classification_status = 'completed'
    WHERE classification_status != 'completed'
    AND NOT EXISTS (
        SELECT 1 FROM CALL_CONTRIBUTIONS cc
        WHERE cc.call_id = c.call_id
        AND cc.identified_speaker_id IS NULL
    )
    AND EXISTS (
        SELECT 1 FROM CALL_CONTRIBUTIONS cc
        WHERE cc.call_id = c.call_id
    );
    
    -- 2. Update speaker meeting counts based on actual assignments
    UPDATE SPEAKERS s
    SET meeting_count = (
        SELECT COUNT(DISTINCT call_id) 
        FROM CALL_CONTRIBUTIONS 
        WHERE identified_speaker_id = s.speaker_id
        AND classification_status = 'classified'
    ),
    updated_at = CURRENT_TIMESTAMP;
    
END;

-- Start the tasks
ALTER TASK EXTRACT_NEW_EMBEDDINGS_TASK RESUME;
ALTER TASK SPEAKER_ASSIGNMENT_HOUSEKEEPING RESUME;

-- ============================================================================
-- VERIFICATION
-- ============================================================================

SELECT 'Setup complete!' as STATUS;
SHOW TABLES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
SHOW STAGES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
SHOW PROCEDURES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
SHOW TASKS IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
