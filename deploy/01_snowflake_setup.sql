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
-- Supported Snowflake accounts: AWS and Azure regions
--
-- ============================================================================

-- ============================================================================
-- PART 1: Database, Schema, and Warehouse
-- ============================================================================

CREATE DATABASE IF NOT EXISTS CALL_TRANSCRIPTS_DB
    COMMENT = 'Call transcription and speaker detection solution';

CREATE SCHEMA IF NOT EXISTS CALL_TRANSCRIPTS_DB.TRANSCRIPTS
    COMMENT = 'Main schema for call transcription objects';

USE SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

CREATE WAREHOUSE IF NOT EXISTS CALL_TRANSCRIPTS_WH
    WAREHOUSE_SIZE = 'SMALL'
    AUTO_SUSPEND = 300
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for call transcription operations';

-- Grant usage
GRANT USAGE ON DATABASE CALL_TRANSCRIPTS_DB TO ROLE ACCOUNTADMIN;
GRANT USAGE ON SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS TO ROLE ACCOUNTADMIN;
GRANT USAGE ON WAREHOUSE CALL_TRANSCRIPTS_WH TO ROLE ACCOUNTADMIN;

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

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_contributions_call ON CALL_CONTRIBUTIONS(CALL_ID);
CREATE INDEX IF NOT EXISTS idx_contributions_speaker ON CALL_CONTRIBUTIONS(IDENTIFIED_SPEAKER_ID);

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
-- PART 4: Network Rules for External Access
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
-- PART 5: Transcription Procedure (Cortex AI_TRANSCRIBE)
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
-- PART 6: Speaker Enrollment Procedure
-- ============================================================================

CREATE OR REPLACE PROCEDURE ENROLL_SPEAKER_FROM_CONTRIBUTION(
    P_SPEAKER_ID VARCHAR,
    P_SPEAKER_NAME VARCHAR,
    P_CALL_ID VARCHAR,
    P_DIARIZATION_LABEL VARCHAR
)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'enroll_speaker'
EXECUTE AS OWNER
AS $$
import json

def enroll_speaker(session, p_speaker_id, p_speaker_name, p_call_id, p_diarization_label):
    """Enroll a speaker by creating voiceprint from a contribution segment"""
    try:
        # Find best contribution (longest duration >= 5s)
        contrib = session.sql(f"""
            SELECT cc.contribution_id, cc.start_time_seconds, cc.end_time_seconds,
                   cc.duration_seconds, c.recording_path
            FROM CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALL_CONTRIBUTIONS cc
            JOIN CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS c ON cc.call_id = c.call_id
            WHERE cc.call_id = '{p_call_id}'
            AND cc.diarization_label = '{p_diarization_label}'
            AND cc.duration_seconds >= 5
            ORDER BY cc.duration_seconds DESC
            LIMIT 1
        """).collect()
        
        if not contrib:
            return {"status": "error", "message": "No suitable contribution found (need >= 5 seconds)"}
        
        start_time = float(contrib[0]['START_TIME_SECONDS'])
        end_time = float(contrib[0]['END_TIME_SECONDS'])
        duration = float(contrib[0]['DURATION_SECONDS'])
        recording_path = contrib[0]['RECORDING_PATH']
        
        # Parse stage path
        stage_path = recording_path.lstrip('@')
        parts = stage_path.split('/')
        if len(parts) < 2:
            return {"status": "error", "message": "Invalid recording path"}
        
        stage_name = '@' + '/'.join(parts[:-1])
        file_name = parts[-1]
        
        # Get presigned URL
        url_result = session.sql(f"""
            SELECT GET_PRESIGNED_URL('{stage_name}', '{file_name}', 3600) as url
        """).collect()
        
        if not url_result or not url_result[0]['URL']:
            return {"status": "error", "message": "Could not get presigned URL"}
        
        presigned_url = url_result[0]['URL']
        
        # Extract embedding via speaker embedding service
        result = session.sql(f"""
            SELECT SPEAKER_EMBEDDING_V9('{presigned_url}', {start_time}, {end_time}) as result
        """).collect()
        
        if not result or not result[0]['RESULT']:
            return {"status": "error", "message": "No result from embedding service"}
        
        svc_result = result[0]['RESULT']
        if isinstance(svc_result, str):
            svc_result = json.loads(svc_result)
        
        if svc_result.get('status') == 'error':
            return {"status": "error", "message": svc_result.get('error', 'Unknown error')}
        
        embedding = svc_result.get('embedding')
        if not embedding:
            return {"status": "error", "message": "No embedding returned"}
        
        if isinstance(embedding, str):
            embedding = json.loads(embedding)
        
        embedding_json = json.dumps(embedding)
        
        # Create/update speaker record
        session.sql(f"""
            MERGE INTO CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKERS t
            USING (SELECT '{p_speaker_id}' as speaker_id) s
            ON t.speaker_id = s.speaker_id
            WHEN MATCHED THEN UPDATE SET
                display_name = '{p_speaker_name.replace("'", "''")}',
                updated_at = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT (speaker_id, display_name, created_at, updated_at)
            VALUES ('{p_speaker_id}', '{p_speaker_name.replace("'", "''")}', CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())
        """).collect()
        
        # Create/update voiceprint
        session.sql(f"""
            MERGE INTO CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_VOICEPRINTS t
            USING (SELECT '{p_speaker_id}' as speaker_id) s
            ON t.speaker_id = s.speaker_id
            WHEN MATCHED THEN UPDATE SET
                speaker_name = '{p_speaker_name.replace("'", "''")}',
                embedding = PARSE_JSON('{embedding_json}')::VECTOR(FLOAT, 192),
                sample_audio_path = '{recording_path}',
                sample_duration_seconds = {duration},
                last_updated = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT (
                speaker_id, speaker_name, embedding, sample_audio_path, 
                sample_duration_seconds, enrollment_date, last_updated
            ) VALUES (
                '{p_speaker_id}', '{p_speaker_name.replace("'", "''")}',
                PARSE_JSON('{embedding_json}')::VECTOR(FLOAT, 192),
                '{recording_path}', {duration}, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()
            )
        """).collect()
        
        return {
            "status": "success",
            "speaker_id": p_speaker_id,
            "speaker_name": p_speaker_name,
            "duration_seconds": duration
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
$$;

-- ============================================================================
-- PART 7: Auto Speaker Detection Procedure
-- ============================================================================

CREATE OR REPLACE PROCEDURE AUTO_DETECT_SPEAKERS(
    P_CALL_ID VARCHAR,
    P_THRESHOLD FLOAT DEFAULT 0.6
)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'auto_detect'
EXECUTE AS OWNER
AS $$
import json

def auto_detect(session, p_call_id, p_threshold=0.6):
    """Auto-detect speakers by matching contribution embeddings against voiceprints"""
    try:
        # Get all voiceprints
        voiceprints = session.sql("""
            SELECT speaker_id, speaker_name, embedding::VARCHAR as embedding
            FROM CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_VOICEPRINTS
            WHERE embedding IS NOT NULL
        """).collect()
        
        if not voiceprints:
            return {"status": "warning", "message": "No voiceprints enrolled"}
        
        # Build profiles list
        profiles = []
        for vp in voiceprints:
            emb_str = vp['EMBEDDING']
            if emb_str:
                try:
                    emb = json.loads(emb_str) if isinstance(emb_str, str) else emb_str
                    profiles.append({
                        "speaker_id": vp['SPEAKER_ID'],
                        "speaker_name": vp['SPEAKER_NAME'],
                        "embedding": emb
                    })
                except:
                    pass
        
        if not profiles:
            return {"status": "warning", "message": "No valid voiceprints"}
        
        # Get call recording info
        call_info = session.sql(f"""
            SELECT recording_path FROM CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS
            WHERE call_id = '{p_call_id}'
        """).collect()
        
        if not call_info or not call_info[0]['RECORDING_PATH']:
            return {"status": "error", "message": "Call not found or no recording"}
        
        recording_path = call_info[0]['RECORDING_PATH']
        
        # Parse stage path
        stage_path = recording_path.lstrip('@')
        parts = stage_path.split('/')
        if len(parts) < 2:
            return {"status": "error", "message": "Invalid recording path"}
        
        stage_name = '@' + '/'.join(parts[:-1])
        file_name = parts[-1]
        
        # Get presigned URL
        url_result = session.sql(f"""
            SELECT GET_PRESIGNED_URL('{stage_name}', '{file_name}', 3600) as url
        """).collect()
        
        if not url_result or not url_result[0]['URL']:
            return {"status": "error", "message": "Could not get presigned URL"}
        
        presigned_url = url_result[0]['URL']
        
        # Get unique diarization labels with their best segment
        labels = session.sql(f"""
            SELECT diarization_label, 
                   MAX(duration_seconds) as max_dur,
                   MIN(start_time_seconds) as min_start,
                   MAX(end_time_seconds) as max_end
            FROM CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALL_CONTRIBUTIONS
            WHERE call_id = '{p_call_id}'
            AND duration_seconds >= 3
            GROUP BY diarization_label
        """).collect()
        
        matched_count = 0
        total_labels = len(labels)
        
        for label_row in labels:
            diar_label = label_row['DIARIZATION_LABEL']
            
            # Get best segment for this label
            best_seg = session.sql(f"""
                SELECT start_time_seconds, end_time_seconds
                FROM CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALL_CONTRIBUTIONS
                WHERE call_id = '{p_call_id}'
                AND diarization_label = '{diar_label}'
                AND duration_seconds >= 3
                ORDER BY duration_seconds DESC
                LIMIT 1
            """).collect()
            
            if not best_seg:
                continue
            
            start_time = float(best_seg[0]['START_TIME_SECONDS'])
            end_time = float(best_seg[0]['END_TIME_SECONDS'])
            
            # Extract embedding
            emb_result = session.sql(f"""
                SELECT SPEAKER_EMBEDDING_V9('{presigned_url}', {start_time}, {end_time}) as result
            """).collect()
            
            if not emb_result or not emb_result[0]['RESULT']:
                continue
            
            svc_result = emb_result[0]['RESULT']
            if isinstance(svc_result, str):
                svc_result = json.loads(svc_result)
            
            if svc_result.get('status') == 'error':
                continue
            
            query_emb = svc_result.get('embedding')
            if not query_emb:
                continue
            
            if isinstance(query_emb, str):
                query_emb = json.loads(query_emb)
            
            # Find best match
            import numpy as np
            query_vec = np.array(query_emb)
            
            best_score = 0.0
            best_match = None
            
            for profile in profiles:
                prof_vec = np.array(profile['embedding'])
                score = float(np.dot(query_vec, prof_vec))
                if score > best_score:
                    best_score = score
                    best_match = profile
            
            if best_match and best_score >= p_threshold:
                # Update all contributions with this label
                session.sql(f"""
                    UPDATE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALL_CONTRIBUTIONS
                    SET identified_speaker_id = '{best_match["speaker_id"]}',
                        identification_method = 'voice_embedding_auto',
                        identification_confidence = {best_score},
                        classification_status = 'completed'
                    WHERE call_id = '{p_call_id}'
                    AND diarization_label = '{diar_label}'
                """).collect()
                matched_count += 1
        
        # Update call status
        session.sql(f"""
            UPDATE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS
            SET classification_status = 'completed',
                updated_at = CURRENT_TIMESTAMP()
            WHERE call_id = '{p_call_id}'
        """).collect()
        
        return {
            "status": "success",
            "call_id": p_call_id,
            "total": total_labels,
            "matched": matched_count,
            "threshold": p_threshold
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
$$;

-- ============================================================================
-- VERIFICATION
-- ============================================================================

SELECT 'Setup complete!' as STATUS;
SHOW TABLES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
SHOW STAGES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
SHOW PROCEDURES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
