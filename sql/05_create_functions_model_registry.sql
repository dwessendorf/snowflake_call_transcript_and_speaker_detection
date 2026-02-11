-- ============================================================================
-- Snowflake Call Transcript and Speaker Detection - Model Registry Functions
-- ============================================================================
-- Creates service functions and stored procedures for Model Registry deployment
-- Run AFTER deploying the model via speaker_model_registry.py
--
-- This replaces 05_create_functions.sql for Model Registry-based deployments
-- ============================================================================

USE SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- ============================================================================
-- Service Function: Extract embedding from presigned URL
-- ============================================================================
-- Primary function for extracting speaker embeddings from audio files
-- Uses the Model Registry service endpoint

CREATE OR REPLACE FUNCTION SPEAKER_EMBEDDING_URL(
    AUDIO_URL VARCHAR,
    START_TIME FLOAT,
    END_TIME FLOAT
)
RETURNS VARIANT
SERVICE = SPEAKER_EMBEDDING_SERVICE
ENDPOINT = 'inference'
AS '/extract-embedding-url';

-- ============================================================================
-- Service Function: Extract embedding from base64 audio
-- ============================================================================
CREATE OR REPLACE FUNCTION SPEAKER_EMBEDDING_B64(
    AUDIO_BASE64 VARCHAR,
    START_TIME FLOAT,
    END_TIME FLOAT
)
RETURNS VARIANT
SERVICE = SPEAKER_EMBEDDING_SERVICE
ENDPOINT = 'inference'
AS '/extract-embedding';

-- ============================================================================
-- Service Function: Compute similarity between embeddings
-- ============================================================================
CREATE OR REPLACE FUNCTION SPEAKER_SIMILARITY(
    EMBEDDING1 VARCHAR,
    EMBEDDING2 VARCHAR,
    THRESHOLD FLOAT
)
RETURNS VARIANT
SERVICE = SPEAKER_EMBEDDING_SERVICE
ENDPOINT = 'inference'
AS '/compute-similarity';

-- ============================================================================
-- Service Function: Batch match against speaker profiles
-- ============================================================================
CREATE OR REPLACE FUNCTION SPEAKER_BATCH_MATCH(
    QUERY_EMBEDDING VARCHAR,
    PROFILES VARCHAR,
    THRESHOLD FLOAT
)
RETURNS VARIANT
SERVICE = SPEAKER_EMBEDDING_SERVICE
ENDPOINT = 'inference'
AS '/batch-match';

-- ============================================================================
-- Service Function: Health check
-- ============================================================================
CREATE OR REPLACE FUNCTION SPEAKER_EMBEDDING_HEALTH()
RETURNS VARIANT
SERVICE = SPEAKER_EMBEDDING_SERVICE
ENDPOINT = 'inference'
AS '/health';

-- ============================================================================
-- Procedure: Extract embedding for a specific contribution
-- ============================================================================
CREATE OR REPLACE PROCEDURE EXTRACT_CONTRIBUTION_EMBEDDING(
    P_CALL_ID VARCHAR,
    P_CONTRIBUTION_ID VARCHAR
)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'extract_embedding'
EXECUTE AS OWNER
AS $$
import json

def extract_embedding(session, p_call_id, p_contribution_id):
    try:
        # Get contribution details
        contrib = session.sql(f"""
            SELECT c.start_time_seconds, c.end_time_seconds, cl.recording_path
            FROM CALL_CONTRIBUTIONS c
            JOIN CALLS cl ON c.call_id = cl.call_id
            WHERE c.contribution_id = '{p_contribution_id}'
        """).collect()
        
        if not contrib:
            return {"status": "error", "message": "Contribution not found"}
        
        start_time = float(contrib[0]['START_TIME_SECONDS'])
        end_time = float(contrib[0]['END_TIME_SECONDS'])
        recording_path = contrib[0]['RECORDING_PATH']
        
        # Parse stage path
        stage_path = recording_path.lstrip('@')
        if '/' in stage_path:
            parts = stage_path.split('/', 1)
            stage_name = '@' + parts[0]
            file_name = parts[1]
        else:
            return {"status": "error", "message": "Invalid recording path"}
        
        # Get presigned URL
        url_result = session.sql(f"""
            SELECT GET_PRESIGNED_URL('{stage_name}', '{file_name}', 3600) as url
        """).collect()
        
        if not url_result or not url_result[0]['URL']:
            return {"status": "error", "message": "Could not get presigned URL"}
        
        presigned_url = url_result[0]['URL']
        
        # Call Model Registry service function
        result = session.sql(f"""
            SELECT SPEAKER_EMBEDDING_URL('{presigned_url}', {start_time}, {end_time}) as result
        """).collect()
        
        if not result or not result[0]['RESULT']:
            return {"status": "error", "message": "No result from service"}
        
        svc_result = result[0]['RESULT']
        if isinstance(svc_result, str):
            svc_result = json.loads(svc_result)
        
        if svc_result.get('status') == 'error':
            return {"status": "error", "message": svc_result.get('error', 'Unknown error')}
        
        embedding = svc_result.get('embedding')
        if embedding:
            # Parse embedding if it's a JSON string
            if isinstance(embedding, str):
                embedding = json.loads(embedding)
            return {
                "status": "success",
                "embedding": embedding,
                "start_time": start_time,
                "end_time": end_time
            }
        
        return {"status": "error", "message": "No embedding in response"}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
$$;

-- ============================================================================
-- Procedure: Create speaker voiceprint from contribution
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
AS $$
import json
import uuid

def create_voiceprint(session, p_speaker_id, p_call_id, p_diarization_label):
    try:
        # Find best contribution for this speaker/label (longest duration >= 10s)
        contrib = session.sql(f"""
            SELECT cc.contribution_id, cc.start_time_seconds, cc.end_time_seconds,
                   cc.duration_seconds, c.recording_path
            FROM CALL_CONTRIBUTIONS cc
            JOIN CALLS c ON cc.call_id = c.call_id
            WHERE cc.call_id = '{p_call_id}'
            AND cc.diarization_label = '{p_diarization_label}'
            AND cc.duration_seconds >= 10
            ORDER BY cc.duration_seconds DESC
            LIMIT 1
        """).collect()
        
        if not contrib:
            return {"status": "error", "message": "No suitable contribution found (need >= 10s)"}
        
        contribution_id = contrib[0]['CONTRIBUTION_ID']
        start_time = float(contrib[0]['START_TIME_SECONDS'])
        end_time = float(contrib[0]['END_TIME_SECONDS'])
        duration = float(contrib[0]['DURATION_SECONDS'])
        recording_path = contrib[0]['RECORDING_PATH']
        
        # Parse stage path
        stage_path = recording_path.lstrip('@')
        if '/' in stage_path:
            parts = stage_path.split('/', 1)
            stage_name = '@' + parts[0]
            file_name = parts[1]
        else:
            return {"status": "error", "message": "Invalid recording path"}
        
        # Get presigned URL
        url_result = session.sql(f"""
            SELECT GET_PRESIGNED_URL('{stage_name}', '{file_name}', 3600) as url
        """).collect()
        
        if not url_result or not url_result[0]['URL']:
            return {"status": "error", "message": "Could not get presigned URL"}
        
        presigned_url = url_result[0]['URL']
        
        # Extract embedding via Model Registry service
        result = session.sql(f"""
            SELECT SPEAKER_EMBEDDING_URL('{presigned_url}', {start_time}, {end_time}) as result
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
        
        # Parse embedding if string
        if isinstance(embedding, str):
            embedding = json.loads(embedding)
        
        embedding_json = json.dumps(embedding)
        
        # Get speaker details
        speaker = session.sql(f"""
            SELECT display_name, email, department 
            FROM SPEAKERS 
            WHERE speaker_id = '{p_speaker_id}'
        """).collect()
        
        if not speaker:
            return {"status": "error", "message": "Speaker not found"}
        
        speaker_name = speaker[0]['DISPLAY_NAME']
        email = speaker[0]['EMAIL'] or ''
        department = speaker[0]['DEPARTMENT'] or ''
        
        # Upsert voiceprint
        session.sql(f"""
            MERGE INTO SPEAKER_VOICEPRINTS t
            USING (SELECT '{p_speaker_id}' as speaker_id) s
            ON t.speaker_id = s.speaker_id
            WHEN MATCHED THEN UPDATE SET
                speaker_name = '{speaker_name}',
                email = '{email}',
                department = '{department}',
                embedding = PARSE_JSON('{embedding_json}')::VECTOR(FLOAT, 192),
                sample_audio_path = '{recording_path}',
                sample_duration_seconds = {duration},
                last_updated = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT (
                speaker_id, speaker_name, email, department, embedding,
                sample_audio_path, sample_duration_seconds, enrollment_date, last_updated
            ) VALUES (
                '{p_speaker_id}', '{speaker_name}', '{email}', '{department}',
                PARSE_JSON('{embedding_json}')::VECTOR(FLOAT, 192),
                '{recording_path}', {duration}, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()
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
$$;

-- ============================================================================
-- Procedure: Auto-identify speakers in a call
-- ============================================================================
CREATE OR REPLACE PROCEDURE AUTO_IDENTIFY_CALL_SPEAKERS(
    P_CALL_ID VARCHAR,
    P_THRESHOLD FLOAT DEFAULT 0.75
)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'auto_identify'
EXECUTE AS OWNER
AS $$
import json

def auto_identify(session, p_call_id, p_threshold=0.75):
    try:
        # Get all voiceprints as profiles
        voiceprints = session.sql("""
            SELECT speaker_id, speaker_name, embedding::VARCHAR as embedding
            FROM SPEAKER_VOICEPRINTS
            WHERE embedding IS NOT NULL
        """).collect()
        
        if not voiceprints:
            return {"status": "warning", "message": "No voiceprints available for matching"}
        
        # Build profiles JSON
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
            return {"status": "warning", "message": "No valid voiceprint embeddings"}
        
        profiles_json = json.dumps(profiles).replace("'", "''")
        
        # Get unidentified contributions with embeddings
        contributions = session.sql(f"""
            SELECT ce.contribution_id, ce.diarization_label, 
                   ce.embedding::VARCHAR as embedding
            FROM CONTRIBUTION_EMBEDDINGS ce
            JOIN CALL_CONTRIBUTIONS cc ON ce.contribution_id = cc.contribution_id
            WHERE ce.call_id = '{p_call_id}'
            AND cc.identified_speaker_id IS NULL
            AND ce.embedding IS NOT NULL
        """).collect()
        
        identified = 0
        unidentified = 0
        
        for contrib in contributions:
            contrib_id = contrib['CONTRIBUTION_ID']
            diar_label = contrib['DIARIZATION_LABEL']
            emb_str = contrib['EMBEDDING']
            
            if not emb_str:
                unidentified += 1
                continue
            
            try:
                query_emb = json.loads(emb_str) if isinstance(emb_str, str) else emb_str
                query_json = json.dumps(query_emb).replace("'", "''")
                
                # Call batch_match via service function
                result = session.sql(f"""
                    SELECT SPEAKER_BATCH_MATCH('{query_json}', '{profiles_json}', {p_threshold}) as result
                """).collect()
                
                if result and result[0]['RESULT']:
                    match_result = result[0]['RESULT']
                    if isinstance(match_result, str):
                        match_result = json.loads(match_result)
                    
                    if match_result.get('matched'):
                        speaker_id = match_result['speaker_id']
                        confidence = match_result['confidence']
                        
                        # Update all contributions with this diarization label
                        session.sql(f"""
                            UPDATE CALL_CONTRIBUTIONS
                            SET identified_speaker_id = '{speaker_id}',
                                identification_method = 'voice_embedding_auto',
                                identification_confidence = {confidence},
                                classification_status = 'auto_identified'
                            WHERE call_id = '{p_call_id}'
                            AND diarization_label = '{diar_label}'
                            AND identified_speaker_id IS NULL
                        """).collect()
                        
                        identified += 1
                    else:
                        unidentified += 1
                else:
                    unidentified += 1
                    
            except Exception as e:
                unidentified += 1
        
        return {
            "status": "success",
            "call_id": p_call_id,
            "identified": identified,
            "unidentified": unidentified,
            "profiles_checked": len(profiles)
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
$$;

-- ============================================================================
-- Procedure: Store contribution embedding
-- ============================================================================
CREATE OR REPLACE PROCEDURE STORE_CONTRIBUTION_EMBEDDING(
    P_CALL_ID VARCHAR,
    P_CONTRIBUTION_ID VARCHAR
)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'store_embedding'
EXECUTE AS OWNER
AS $$
import json

def store_embedding(session, p_call_id, p_contribution_id):
    try:
        # Extract embedding
        result = session.sql(f"""
            CALL EXTRACT_CONTRIBUTION_EMBEDDING('{p_call_id}', '{p_contribution_id}')
        """).collect()
        
        if not result:
            return {"status": "error", "message": "No result from extraction"}
        
        extract_result = result[0][0]
        if isinstance(extract_result, str):
            extract_result = json.loads(extract_result)
        
        if extract_result.get('status') != 'success':
            return extract_result
        
        embedding = extract_result.get('embedding')
        if not embedding:
            return {"status": "error", "message": "No embedding returned"}
        
        embedding_json = json.dumps(embedding)
        
        # Get contribution details
        contrib = session.sql(f"""
            SELECT diarization_label, duration_seconds
            FROM CALL_CONTRIBUTIONS
            WHERE contribution_id = '{p_contribution_id}'
        """).collect()
        
        if not contrib:
            return {"status": "error", "message": "Contribution not found"}
        
        diar_label = contrib[0]['DIARIZATION_LABEL']
        duration = float(contrib[0]['DURATION_SECONDS'] or 0)
        
        # Store embedding
        session.sql(f"""
            MERGE INTO CONTRIBUTION_EMBEDDINGS t
            USING (SELECT '{p_contribution_id}' as contribution_id) s
            ON t.contribution_id = s.contribution_id
            WHEN MATCHED THEN UPDATE SET
                embedding = PARSE_JSON('{embedding_json}')::VECTOR(FLOAT, 192),
                updated_at = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT (
                embedding_id, call_id, contribution_id, diarization_label,
                embedding, duration_seconds, created_at, updated_at
            ) VALUES (
                UUID_STRING(), '{p_call_id}', '{p_contribution_id}', '{diar_label}',
                PARSE_JSON('{embedding_json}')::VECTOR(FLOAT, 192),
                {duration}, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()
            )
        """).collect()
        
        return {
            "status": "success",
            "contribution_id": p_contribution_id,
            "embedding_dim": len(embedding)
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
$$;

-- ============================================================================
-- Helper Table: Contribution Embeddings (if not exists)
-- ============================================================================
CREATE TABLE IF NOT EXISTS CONTRIBUTION_EMBEDDINGS (
    EMBEDDING_ID VARCHAR(50) NOT NULL PRIMARY KEY,
    CALL_ID VARCHAR(50) NOT NULL,
    CONTRIBUTION_ID VARCHAR(50) NOT NULL,
    DIARIZATION_LABEL VARCHAR(20),
    EMBEDDING VECTOR(FLOAT, 192),
    DURATION_SECONDS FLOAT,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    UPDATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    UNIQUE (CONTRIBUTION_ID)
);

CREATE INDEX IF NOT EXISTS idx_contrib_emb_call ON CONTRIBUTION_EMBEDDINGS(CALL_ID);
CREATE INDEX IF NOT EXISTS idx_contrib_emb_label ON CONTRIBUTION_EMBEDDINGS(DIARIZATION_LABEL);

-- ============================================================================
-- Verify Functions Created
-- ============================================================================
SHOW FUNCTIONS LIKE '%SPEAKER%' IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
SHOW PROCEDURES LIKE '%EMBEDDING%' IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
SHOW PROCEDURES LIKE '%SPEAKER%' IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- ============================================================================
-- Test Queries (uncomment to test)
-- ============================================================================

-- Test health check:
-- SELECT SPEAKER_EMBEDDING_HEALTH();

-- Test embedding extraction:
-- SELECT SPEAKER_EMBEDDING_URL(
--     GET_PRESIGNED_URL('@CALL_RECORDINGS', 'sample.mp3', 3600),
--     0.0,
--     30.0
-- );

-- Test similarity:
-- SELECT SPEAKER_SIMILARITY(
--     '[0.1, 0.2, ...]',  -- embedding1
--     '[0.15, 0.22, ...]', -- embedding2
--     0.75  -- threshold
-- );
