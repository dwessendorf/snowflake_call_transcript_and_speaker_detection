-- ============================================================================
-- Call Transcript & Speaker Detection - Service Functions
-- ============================================================================
-- 
-- Run this AFTER deploying the speaker embedding model via:
--   python deploy/register_model.py
--
-- This creates the service functions that call the GPU inference service.
-- The service is deployed via Snowflake Model Registry.
--
-- ============================================================================

USE SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
USE WAREHOUSE CALL_TRANSCRIPTS_WH;

-- ============================================================================
-- Service Function: Extract embedding from presigned URL
-- ============================================================================
-- This is the primary function for extracting speaker embeddings.
-- It accepts presigned URLs from GET_PRESIGNED_URL() for stage files.
--
-- Parameters:
--   AUDIO_URL: Presigned URL to the audio file  
--   START_TIME: Start time in seconds for the segment
--   END_TIME: End time in seconds for the segment
--
-- Returns: OBJECT with fields:
--   - status: 'success' or 'error'
--   - embedding: array of 192 floats (on success)
--   - error: error message (on failure)
--
-- Note: The SERVICE name is set by the Model Registry deployment.
-- Default service name is SPEAKER_EMBEDDING_SVC (from V20_GPU model version).
-- If you deployed with a different version, update the SERVICE name below.

CREATE OR REPLACE FUNCTION SPEAKER_EMBEDDING_URL(
    AUDIO_URL VARCHAR,
    START_TIME FLOAT,
    END_TIME FLOAT
)
RETURNS OBJECT
MAX_BATCH_ROWS = 1
SERVICE = SPEAKER_EMBEDDING_SVC
SERVICE ENDPOINT = 'inference'
AS '/extract-embedding-url';

-- ============================================================================
-- Service Function: Health check
-- ============================================================================
-- Simple health check to verify the service is running.

CREATE OR REPLACE FUNCTION SPEAKER_EMBEDDING_HEALTH()
RETURNS VARIANT
SERVICE = SPEAKER_EMBEDDING_SVC
ENDPOINT = 'inference'
AS '/health';

-- ============================================================================
-- VERIFICATION
-- ============================================================================

SELECT 'Service functions created!' as STATUS;
SHOW USER FUNCTIONS LIKE 'SPEAKER_EMBEDDING%' IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- Test health check (uncomment to verify service is running)
-- SELECT SPEAKER_EMBEDDING_HEALTH();

-- Test embedding extraction (uncomment with real URL)
-- SELECT SPEAKER_EMBEDDING_URL(
--     GET_PRESIGNED_URL('@CALL_RECORDINGS', 'sample.mp3', 3600),
--     0.0, 
--     30.0
-- );
