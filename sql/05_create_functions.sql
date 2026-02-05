-- ============================================================================
-- Snowflake Speaker Detection - Functions and Procedures
-- ============================================================================
-- Creates service functions and stored procedures
-- Run AFTER the SPCS service is deployed and running
-- ============================================================================

USE SCHEMA MEETING_AGENT_DB.MEETING_AGENT;

-- ============================================================================
-- Service Function: Extract speaker embedding from audio URL
-- ============================================================================
-- This function calls the SPCS service to extract voice embeddings
-- Parameters:
--   AUDIO_URL: Presigned URL to the audio file
--   START_TIME: Start time in seconds for the segment
--   END_TIME: End time in seconds for the segment

CREATE OR REPLACE FUNCTION SPEAKER_EMBEDDING_URL(
    AUDIO_URL VARCHAR,
    START_TIME FLOAT,
    END_TIME FLOAT
)
RETURNS VARIANT
SERVICE = SPEAKER_IDENTIFICATION_SERVICE
ENDPOINT = 'speaker-api'
AS '/extract-embedding-url';

-- ============================================================================
-- Procedure: Extract embedding for a specific contribution
-- ============================================================================
-- This procedure wraps the embedding extraction with business logic
-- Parameters:
--   P_MEETING_ID: Meeting ID
--   P_CONTRIBUTION_ID: Contribution ID to extract embedding for

CREATE OR REPLACE PROCEDURE EXTRACT_CONTRIBUTION_EMBEDDING(
    P_MEETING_ID VARCHAR,
    P_CONTRIBUTION_ID VARCHAR
)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'extract_embedding'
EXECUTE AS OWNER
AS '
import json

def extract_embedding(session, p_meeting_id, p_contribution_id):
    try:
        # Get contribution details
        contrib = session.sql(f"""
            SELECT c.start_time_seconds, c.end_time_seconds, m.recording_path
            FROM MEETING_CONTRIBUTIONS c
            JOIN MEETINGS m ON c.meeting_id = m.meeting_id
            WHERE c.contribution_id = ''{p_contribution_id}''
        """).collect()
        
        if not contrib:
            return {"status": "error", "message": "Contribution not found"}
        
        start_time = float(contrib[0][''START_TIME_SECONDS''])
        end_time = float(contrib[0][''END_TIME_SECONDS''])
        recording_path = contrib[0][''RECORDING_PATH'']
        
        # Parse stage path to get presigned URL
        stage_path = recording_path.lstrip(''@'')
        if ''/'' in stage_path:
            parts = stage_path.split(''/'', 1)
            stage_name = ''@'' + parts[0]
            file_name = parts[1]
        else:
            return {"status": "error", "message": "Invalid recording path"}
        
        # Get presigned URL
        url_result = session.sql(f"""
            SELECT GET_PRESIGNED_URL(''{stage_name}'', ''{file_name}'', 3600) as url
        """).collect()
        
        if not url_result or not url_result[0][''URL'']:
            return {"status": "error", "message": "Could not get presigned URL"}
        
        presigned_url = url_result[0][''URL'']
        
        # Call service function (uses internal SPCS endpoint)
        result = session.sql(f"""
            SELECT SPEAKER_EMBEDDING_URL(''{presigned_url}'', {start_time}, {end_time}) as result
        """).collect()
        
        if not result or not result[0][''RESULT'']:
            return {"status": "error", "message": "No result from service function"}
        
        svc_result = result[0][''RESULT'']
        if isinstance(svc_result, str):
            svc_result = json.loads(svc_result)
        
        if svc_result.get(''error''):
            return {"status": "error", "message": svc_result[''error'']}
        
        embedding = svc_result.get(''embedding'')
        if embedding:
            return {
                "status": "success",
                "embedding": embedding,
                "start_time": start_time,
                "end_time": end_time
            }
        
        return {"status": "error", "message": "No embedding in response"}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
';

-- ============================================================================
-- Test the functions (run after service is up)
-- ============================================================================
-- First, get a presigned URL for testing:
-- SELECT GET_PRESIGNED_URL('@MEETING_RECORDINGS', 'your_audio_file.mp3', 3600) as test_url;

-- Then test the embedding extraction:
-- SELECT SPEAKER_EMBEDDING_URL('<presigned_url>', 0.0, 30.0);

-- Test the procedure:
-- CALL EXTRACT_CONTRIBUTION_EMBEDDING('<meeting_id>', '<contribution_id>');

-- ============================================================================
-- Verify functions created
-- ============================================================================
SHOW FUNCTIONS LIKE '%SPEAKER%' IN SCHEMA MEETING_AGENT_DB.MEETING_AGENT;
SHOW PROCEDURES LIKE '%EMBEDDING%' IN SCHEMA MEETING_AGENT_DB.MEETING_AGENT;
