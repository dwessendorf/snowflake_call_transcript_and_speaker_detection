-- ============================================================================
-- Snowflake Call Transcript and Speaker Detection - Stage Creation
-- ============================================================================
-- Creates internal stages for storing audio files and application assets
-- ============================================================================

USE SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- ============================================================================
-- Stage for call recordings (audio/video files)
-- Using NO CSE for AI_TRANSCRIBE compatibility
-- ============================================================================
CREATE STAGE IF NOT EXISTS CALL_RECORDINGS
    DIRECTORY = (ENABLE = TRUE)
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    COMMENT = 'Storage for call audio/video recordings';

-- ============================================================================
-- Stage for audio snippets (extracted segments)
-- ============================================================================
CREATE STAGE IF NOT EXISTS AUDIO_SNIPPETS
    DIRECTORY = (ENABLE = TRUE)
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    COMMENT = 'Audio snippets of individual speech segments';

-- ============================================================================
-- Stage for speaker enrollment samples
-- ============================================================================
CREATE STAGE IF NOT EXISTS ENROLLMENT_SAMPLES
    DIRECTORY = (ENABLE = TRUE)
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    COMMENT = 'Storage for speaker enrollment audio samples';

-- ============================================================================
-- Stage for speaker profile samples
-- ============================================================================
CREATE STAGE IF NOT EXISTS SPEAKER_PROFILE_SAMPLES
    DIRECTORY = (ENABLE = TRUE)
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    COMMENT = 'Audio samples for speaker profiles';

-- ============================================================================
-- Stage for Streamlit applications
-- ============================================================================
CREATE STAGE IF NOT EXISTS STREAMLIT_APPS
    DIRECTORY = (ENABLE = FALSE)
    ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
    COMMENT = 'Stage for Streamlit application files';

-- ============================================================================
-- Stage for transcription outputs (optional)
-- ============================================================================
CREATE STAGE IF NOT EXISTS CALL_TRANSCRIPTIONS
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Storage for call transcription outputs';

-- Verify stages created
SHOW STAGES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
