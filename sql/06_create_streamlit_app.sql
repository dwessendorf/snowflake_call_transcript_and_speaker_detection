-- ============================================================================
-- Snowflake Call Transcript and Speaker Detection - Streamlit App Deployment
-- ============================================================================
-- Deploys the Speaker Classification Streamlit application
-- ============================================================================

USE SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- ============================================================================
-- Step 1: Upload app.py to stage
-- ============================================================================
-- Run this from your local machine using snowsql:
-- PUT file://streamlit_app/app.py @STREAMLIT_APPS/speaker_classification_app/ OVERWRITE=TRUE AUTO_COMPRESS=FALSE;

-- Or using Python/snowflake-connector:
-- cursor.execute("PUT file:///path/to/app.py @STREAMLIT_APPS/speaker_classification_app/ OVERWRITE=TRUE AUTO_COMPRESS=FALSE")

-- ============================================================================
-- Step 2: Create Streamlit App
-- ============================================================================
CREATE STREAMLIT IF NOT EXISTS SPEAKER_CLASSIFICATION_APP
    ROOT_LOCATION = '@CALL_TRANSCRIPTS_DB.TRANSCRIPTS.STREAMLIT_APPS/speaker_classification_app'
    MAIN_FILE = 'app.py'
    QUERY_WAREHOUSE = CALL_TRANSCRIPTS_WH  -- Update to your warehouse name
    TITLE = 'Speaker Classification'
    COMMENT = 'UI for classifying and managing speakers in call transcripts';

-- ============================================================================
-- Step 3: Grant access (optional - for other roles)
-- ============================================================================
-- GRANT USAGE ON STREAMLIT SPEAKER_CLASSIFICATION_APP TO ROLE <your_role>;

-- ============================================================================
-- Verify deployment
-- ============================================================================
SHOW STREAMLITS IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- ============================================================================
-- To update the app after changes:
-- ============================================================================
-- 1. Remove old file:
--    REMOVE @STREAMLIT_APPS/speaker_classification_app/app.py;
--
-- 2. Upload new file:
--    PUT file://streamlit_app/app.py @STREAMLIT_APPS/speaker_classification_app/ OVERWRITE=TRUE AUTO_COMPRESS=FALSE;
--
-- 3. Refresh the Streamlit app in Snowsight (the app picks up changes automatically)
