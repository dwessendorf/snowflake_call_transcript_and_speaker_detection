-- ============================================================================
-- Snowflake Call Transcript and Speaker Detection - SPCS Setup
-- ============================================================================
-- Sets up Snowflake Container Services for the speaker identification service
-- Requires ACCOUNTADMIN role
--
-- IMPORTANT: This script creates the infrastructure but the Docker image must
-- be built and pushed BEFORE the service can start successfully.
--
-- Deployment Order:
-- 1. Run this script to create repository, compute pool, network rules
-- 2. Build and push Docker image (see Step 6 below)
-- 3. Create the service (Step 7) AFTER the image is pushed
-- ============================================================================

USE SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- ============================================================================
-- Step 1: Create Image Repository
-- ============================================================================
CREATE IMAGE REPOSITORY IF NOT EXISTS SPEAKER_IDENTIFICATION_REPO;

-- Get repository URL (you'll need this for docker push)
SHOW IMAGE REPOSITORIES IN SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;
-- Note the repository_url value, e.g.:
-- <account>.registry.snowflakecomputing.com/call_transcripts_db/transcripts/speaker_identification_repo

-- ============================================================================
-- Step 2: Create Compute Pool
-- ============================================================================
CREATE COMPUTE POOL IF NOT EXISTS SPEAKER_IDENTIFICATION_POOL
    MIN_NODES = 1
    MAX_NODES = 3
    INSTANCE_FAMILY = CPU_X64_S
    AUTO_RESUME = TRUE
    AUTO_SUSPEND_SECS = 600
    COMMENT = 'Compute pool for speaker identification service';

-- Wait for compute pool to be active (may take 1-2 minutes)
-- Run this until state = 'ACTIVE' or 'IDLE':
-- DESCRIBE COMPUTE POOL SPEAKER_IDENTIFICATION_POOL;

-- ============================================================================
-- Step 3: Create Network Rule for HuggingFace (model download)
-- ============================================================================
CREATE OR REPLACE NETWORK RULE HF_NETWORK_RULE
    TYPE = HOST_PORT
    MODE = EGRESS
    VALUE_LIST = (
        'huggingface.co:443',
        'cdn-lfs.huggingface.co:443',
        'cdn-lfs-us-1.huggingface.co:443'
    );

-- ============================================================================
-- Step 4: Create Network Rule for S3 Access (presigned URLs)
-- ============================================================================
-- IMPORTANT: You may need to update these S3 endpoints for your region.
-- To find your S3 endpoint, run:
--   SELECT GET_PRESIGNED_URL('@CALL_RECORDINGS', 'test.mp3', 60);
-- And extract the hostname from the URL.

CREATE OR REPLACE NETWORK RULE S3_NETWORK_RULE
    TYPE = HOST_PORT
    MODE = EGRESS
    VALUE_LIST = (
        -- Common S3 endpoints - add your region's endpoint if different
        'sfc-eu-ds1-38-customer-stage.s3.eu-central-1.amazonaws.com:443',
        'sfc-eu-ds1-38-customer-stage.s3.amazonaws.com:443'
    );

-- ============================================================================
-- Step 5: Create External Access Integrations
-- ============================================================================
-- These integrations allow the SPCS service to access external resources.
-- IMPORTANT: Network rules must exist BEFORE creating integrations.

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION CALL_TRANSCRIPTS_DB_HF_ACCESS
    ALLOWED_NETWORK_RULES = (CALL_TRANSCRIPTS_DB.TRANSCRIPTS.HF_NETWORK_RULE)
    ENABLED = TRUE
    COMMENT = 'Access to HuggingFace for model downloads';

CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION CALL_TRANSCRIPTS_DB_S3_ACCESS
    ALLOWED_NETWORK_RULES = (CALL_TRANSCRIPTS_DB.TRANSCRIPTS.S3_NETWORK_RULE)
    ENABLED = TRUE
    COMMENT = 'Access to S3 for presigned URLs';

-- ============================================================================
-- Step 6: Build and Push Docker Image
-- ============================================================================
-- STOP HERE and run these commands in your terminal before proceeding:
--
-- # Get your registry URL from Step 1 above
-- REGISTRY_URL="<account>.registry.snowflakecomputing.com"
--
-- # Build the image (from spcs_service directory)
-- cd spcs_service
-- docker build --platform linux/amd64 -t speaker-identification:v1 .
--
-- # Login to Snowflake registry
-- # Option A: With password
-- docker login $REGISTRY_URL -u <username>
--
-- # Option B: With PAT token
-- echo '<pat_token>' | docker login $REGISTRY_URL -u <username> --password-stdin
--
-- # Tag and push
-- docker tag speaker-identification:v1 $REGISTRY_URL/call_transcripts_db/transcripts/speaker_identification_repo/speaker-identification:v1
-- docker push $REGISTRY_URL/call_transcripts_db/transcripts/speaker_identification_repo/speaker-identification:v1
--
-- Verify the image was uploaded:
CALL SYSTEM$REGISTRY_LIST_IMAGES('/call_transcripts_db/transcripts/speaker_identification_repo');
-- Should show: {"images":["speaker-identification"]}

-- ============================================================================
-- Step 7: Create the Service
-- ============================================================================
-- IMPORTANT: Only run this AFTER the Docker image has been pushed!
-- If you run this before pushing the image, you'll get:
--   "Image /call_transcripts_db/transcripts/speaker_identification_repo/speaker-identification:v1 not found"

CREATE SERVICE IF NOT EXISTS SPEAKER_IDENTIFICATION_SERVICE
    IN COMPUTE POOL SPEAKER_IDENTIFICATION_POOL
    MIN_INSTANCES = 1
    MAX_INSTANCES = 3
    MIN_READY_INSTANCES = 1
    AUTO_RESUME = TRUE
    EXTERNAL_ACCESS_INTEGRATIONS = (CALL_TRANSCRIPTS_DB_HF_ACCESS, CALL_TRANSCRIPTS_DB_S3_ACCESS)
    COMMENT = 'Speaker identification service using ECAPA-TDNN model'
    FROM SPECIFICATION $$
    spec:
      containers:
      - name: speaker-service
        image: /call_transcripts_db/transcripts/speaker_identification_repo/speaker-identification:v1
        env:
          SNOWFLAKE_DATABASE: CALL_TRANSCRIPTS_DB
          SNOWFLAKE_SCHEMA: TRANSCRIPTS
        resources:
          limits:
            memory: 8G
            cpu: 3
          requests:
            memory: 4G
            cpu: 2
      endpoints:
      - name: speaker-api
        port: 8080
        public: true
    $$;

-- ============================================================================
-- Step 8: Verify Service is Running
-- ============================================================================
-- Wait for service to start (may take 2-5 minutes for first startup)
-- Run this until status = 'RUNNING':
DESCRIBE SERVICE SPEAKER_IDENTIFICATION_SERVICE;

-- If service fails, check logs:
-- CALL SYSTEM$GET_SERVICE_LOGS('SPEAKER_IDENTIFICATION_SERVICE', '0', 'speaker-service', 100);

-- Get service endpoint URL:
SHOW ENDPOINTS IN SERVICE SPEAKER_IDENTIFICATION_SERVICE;

-- ============================================================================
-- Troubleshooting
-- ============================================================================
-- 
-- Problem: "Image not found" error
-- Solution: Push the Docker image first (Step 6), then create service
--
-- Problem: "Network rule does not exist" error  
-- Solution: Create network rules (Steps 3-4) before integrations (Step 5)
--
-- Problem: Service stuck in PENDING
-- Solution: Check compute pool status, ensure it's ACTIVE or IDLE
--
-- Problem: Service fails to start
-- Solution: Check logs with SYSTEM$GET_SERVICE_LOGS
--
-- Problem: Embedding extraction fails with S3 errors
-- Solution: Update S3_NETWORK_RULE with your region's S3 endpoint
--
-- See docs/troubleshooting.md for more detailed solutions.
-- ============================================================================
