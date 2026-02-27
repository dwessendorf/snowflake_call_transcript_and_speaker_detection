-- ============================================================================
-- Snowflake Call Transcript and Speaker Detection - Model Registry Setup
-- ============================================================================
-- Sets up Snowflake Model Registry for speaker embedding service
-- This replaces the manual SPCS Docker deployment with managed model serving
--
-- Prerequisites:
--   - snowflake-ml-python >= 1.25.0 installed locally
--   - ACCOUNTADMIN or appropriate privileges
--
-- Deployment Steps:
--   1. Run this script to create infrastructure
--   2. Run Python: python spcs_service/speaker_model_registry.py
--   3. Run 05_create_functions_model_registry.sql to create service functions
-- ============================================================================

USE DATABASE CALL_TRANSCRIPTS_DB;
USE SCHEMA TRANSCRIPTS;
USE WAREHOUSE CALL_TRANSCRIPTS_WH;

-- ============================================================================
-- Step 1: Grant Model Registry Privileges
-- ============================================================================
-- The role needs CREATE MODEL privilege on the schema

-- If using a custom role, grant these privileges:
-- GRANT CREATE MODEL ON SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS TO ROLE <your_role>;
-- GRANT USAGE ON WAREHOUSE CALL_TRANSCRIPTS_WH TO ROLE <your_role>;

-- ============================================================================
-- Step 2: Create Network Rules for External Access
-- ============================================================================
-- Model Registry services need network access for:
-- 1. HuggingFace (model downloads)
-- 2. S3/Azure Blob (presigned URLs for audio files)

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

-- S3 access for presigned URLs (AWS regions)
-- Update these endpoints based on your Snowflake account region
CREATE OR REPLACE NETWORK RULE MODEL_REGISTRY_S3_RULE
    TYPE = HOST_PORT
    MODE = EGRESS
    VALUE_LIST = (
        -- Add your region's S3 endpoints
        -- Find yours: SELECT GET_PRESIGNED_URL('@CALL_RECORDINGS', 'test.mp3', 60);
        'sfc-eu-ds1-38-customer-stage.s3.eu-central-1.amazonaws.com:443',
        'sfc-eu-ds1-38-customer-stage.s3.amazonaws.com:443',
        '*.s3.amazonaws.com:443',
        '*.s3.us-east-1.amazonaws.com:443',
        '*.s3.us-west-2.amazonaws.com:443',
        '*.s3.eu-central-1.amazonaws.com:443',
        '*.s3.eu-west-1.amazonaws.com:443'
    );

-- Azure Blob access for presigned URLs (Azure regions)
CREATE OR REPLACE NETWORK RULE MODEL_REGISTRY_AZURE_RULE
    TYPE = HOST_PORT
    MODE = EGRESS
    VALUE_LIST = (
        '*.blob.core.windows.net:443',
        '*.blob.storage.azure.net:443'
    );

-- ============================================================================
-- Step 3: Create External Access Integration
-- ============================================================================
-- Combines network rules into a single integration for the model service

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION MODEL_REGISTRY_EXTERNAL_ACCESS
    ALLOWED_NETWORK_RULES = (
        MODEL_REGISTRY_HF_RULE,
        MODEL_REGISTRY_S3_RULE,
        MODEL_REGISTRY_AZURE_RULE
    )
    ENABLED = TRUE
    COMMENT = 'External access for Model Registry speaker embedding service';

-- ============================================================================
-- Step 4: Verify System Compute Pools (Optional)
-- ============================================================================
-- Model Registry can use system compute pools for deployment
-- These are pre-created by Snowflake and available in most accounts

-- Check available system pools:
SHOW COMPUTE POOLS LIKE 'SYSTEM%';

-- Expected pools:
-- - SYSTEM_COMPUTE_POOL_CPU: For CPU-only inference
-- - SYSTEM_COMPUTE_POOL_GPU: For GPU-accelerated inference (recommended)

-- ============================================================================
-- Step 5: Create Custom Compute Pool (Optional)
-- ============================================================================
-- Use this if you want dedicated resources instead of system pools

-- CREATE COMPUTE POOL IF NOT EXISTS SPEAKER_MODEL_POOL
--     MIN_NODES = 1
--     MAX_NODES = 3
--     INSTANCE_FAMILY = GPU_NV_S  -- Small GPU instance
--     AUTO_RESUME = TRUE
--     AUTO_SUSPEND_SECS = 300
--     COMMENT = 'Dedicated pool for speaker embedding model';

-- ============================================================================
-- Step 6: Register and Deploy Model (Run from Python)
-- ============================================================================
-- 
-- Option A: Using the provided script
--   export SNOWFLAKE_CONNECTION_NAME=your_connection
--   cd spcs_service
--   python speaker_model_registry.py
--
-- Option B: Manual registration in Python
--   from snowflake.snowpark import Session
--   from speaker_model_registry import register_model, deploy_service
--   
--   session = Session.builder.config("connection_name", "your_conn").create()
--   mv = register_model(session, "SPEAKER_EMBEDDING", "v1")
--   deploy_service(mv, "SPEAKER_EMBEDDING_SERVICE")
--
-- The model will be registered at: CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_EMBEDDING

-- ============================================================================
-- Step 7: Verify Model Registration
-- ============================================================================
-- After running the Python script, verify the model was created:

SHOW MODELS IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- Check model versions:
-- SHOW VERSIONS IN MODEL CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_EMBEDDING;

-- ============================================================================
-- Step 8: Check Service Status
-- ============================================================================
-- After deployment, check service status:

-- List all services from models:
-- SELECT * FROM INFORMATION_SCHEMA.MODEL_VERSIONS 
-- WHERE MODEL_NAME = 'SPEAKER_EMBEDDING';

-- Show service endpoints:
-- SHOW ENDPOINTS IN SERVICE SPEAKER_EMBEDDING_SERVICE;

-- Get service logs if needed:
-- CALL SYSTEM$GET_SERVICE_LOGS('SPEAKER_EMBEDDING_SERVICE', '0', 'inference', 100);

-- ============================================================================
-- Step 9: Test the Service
-- ============================================================================
-- Quick test using the service functions (created in 05_create_functions_model_registry.sql)

-- Test health endpoint:
-- SELECT SPEAKER_EMBEDDING_HEALTH();

-- Test embedding extraction with presigned URL:
-- SELECT SPEAKER_EMBEDDING_URL(
--     GET_PRESIGNED_URL('@CALL_RECORDINGS', 'sample.mp3', 3600),
--     0.0,
--     30.0
-- );

-- ============================================================================
-- Cleanup (if needed)
-- ============================================================================
-- To remove the model and service:

-- DROP SERVICE IF EXISTS SPEAKER_EMBEDDING_SERVICE;
-- DROP MODEL IF EXISTS CALL_TRANSCRIPTS_DB.TRANSCRIPTS.SPEAKER_EMBEDDING;
-- DROP EXTERNAL ACCESS INTEGRATION IF EXISTS MODEL_REGISTRY_EXTERNAL_ACCESS;
-- DROP NETWORK RULE IF EXISTS MODEL_REGISTRY_HF_RULE;
-- DROP NETWORK RULE IF EXISTS MODEL_REGISTRY_S3_RULE;
-- DROP NETWORK RULE IF EXISTS MODEL_REGISTRY_AZURE_RULE;

-- ============================================================================
-- Migration from Old SPCS Service
-- ============================================================================
-- If you have the old manual SPCS service running:
--
-- 1. Keep old service running during migration
-- 2. Deploy new Model Registry service
-- 3. Update service functions to point to new service
-- 4. Test thoroughly
-- 5. Drop old service:
--    DROP SERVICE IF EXISTS SPEAKER_IDENTIFICATION_SERVICE;
--    DROP COMPUTE POOL IF EXISTS SPEAKER_IDENTIFICATION_POOL;
--    DROP IMAGE REPOSITORY IF EXISTS SPEAKER_IDENTIFICATION_REPO;

SHOW NETWORK RULES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
SHOW EXTERNAL ACCESS INTEGRATIONS;
