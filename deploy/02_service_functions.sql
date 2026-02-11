-- ============================================================================
-- Call Transcript & Speaker Detection - Service Functions
-- ============================================================================
-- 
-- Run this AFTER deploying the speaker embedding model via:
--   python deploy/register_model.py
--
-- This creates the service functions that call the Model Registry service.
--
-- ============================================================================

USE SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
USE WAREHOUSE CALL_TRANSCRIPTS_WH;

-- ============================================================================
-- Service Function: Extract embedding from presigned URL
-- ============================================================================
-- This is the primary function for extracting speaker embeddings.
-- It accepts presigned URLs from GET_PRESIGNED_URL() for stage files.

CREATE OR REPLACE FUNCTION SPEAKER_EMBEDDING_V9(
    AUDIO_URL VARCHAR,
    START_TIME FLOAT,
    END_TIME FLOAT
)
RETURNS VARIANT
SERVICE = SPEAKER_EMBEDDING_SVC_V9
ENDPOINT = 'extract_embedding_url'
AS '/extract_embedding_url';

-- ============================================================================
-- Service Function: Health check
-- ============================================================================

CREATE OR REPLACE FUNCTION SPEAKER_EMBEDDING_HEALTH()
RETURNS VARIANT
SERVICE = SPEAKER_EMBEDDING_SVC_V9
ENDPOINT = 'health'
AS '/health';

-- ============================================================================
-- VERIFICATION
-- ============================================================================

SELECT 'Service functions created!' as STATUS;
SHOW USER FUNCTIONS LIKE 'SPEAKER_EMBEDDING%' IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- Test health check (uncomment to verify service is running)
-- SELECT SPEAKER_EMBEDDING_HEALTH();
