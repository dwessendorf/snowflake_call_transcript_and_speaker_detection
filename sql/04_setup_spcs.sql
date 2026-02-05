-- ============================================================================
-- Snowflake Speaker Detection - SPCS Setup
-- ============================================================================
-- Sets up Snowflake Container Services for the speaker identification service
-- Requires ACCOUNTADMIN role
-- ============================================================================

USE SCHEMA MEETING_AGENT_DB.MEETING_AGENT;

-- ============================================================================
-- Step 1: Create Image Repository
-- ============================================================================
CREATE IMAGE REPOSITORY IF NOT EXISTS SPEAKER_IDENTIFICATION_REPO;

-- Get repository URL (you'll need this for docker push)
SHOW IMAGE REPOSITORIES IN SCHEMA MEETING_AGENT_DB.MEETING_AGENT;
-- Note the repository_url value for docker push commands

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

-- Wait for compute pool to be active
-- DESCRIBE COMPUTE POOL SPEAKER_IDENTIFICATION_POOL;

-- ============================================================================
-- Step 3: Create External Access Integration (for HuggingFace model download)
-- ============================================================================
-- Network rule to allow access to HuggingFace
CREATE OR REPLACE NETWORK RULE HF_NETWORK_RULE
    TYPE = HOST_PORT
    MODE = EGRESS
    VALUE_LIST = ('huggingface.co:443', 'cdn-lfs.huggingface.co:443', 'cdn-lfs-us-1.huggingface.co:443');

-- External access integration
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION HF_ACCESS_INTEGRATION
    ALLOWED_NETWORK_RULES = (HF_NETWORK_RULE)
    ENABLED = TRUE
    COMMENT = 'Access to HuggingFace for model downloads';

-- ============================================================================
-- Step 4: Create Network Rule for S3 Access (presigned URLs)
-- ============================================================================
-- NOTE: Update the S3 bucket URL to match your Snowflake region
-- You can find this by running: SELECT GET_PRESIGNED_URL('@MEETING_RECORDINGS', 'test.mp3', 60);
-- and extracting the S3 hostname

CREATE OR REPLACE NETWORK RULE SPEAKER_SERVICE_NETWORK_RULE
    TYPE = HOST_PORT
    MODE = EGRESS
    VALUE_LIST = (
        -- Add your account's SPCS endpoint (replace <account> with your account identifier)
        '<account>.snowflakecomputing.app:443',
        -- Add your region's S3 endpoint for presigned URLs (example for eu-central-1)
        'sfc-eu-ds1-38-customer-stage.s3.eu-central-1.amazonaws.com:443'
    );

-- ============================================================================
-- Step 5: Create External Access Integration for Service
-- ============================================================================
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION SPEAKER_SERVICE_ACCESS
    ALLOWED_NETWORK_RULES = (SPEAKER_SERVICE_NETWORK_RULE)
    ENABLED = TRUE
    COMMENT = 'Access to Speaker Identification SPCS Service';

-- ============================================================================
-- Step 6: Create the Service
-- ============================================================================
-- NOTE: Update the image path to match your repository URL
-- Replace <account> with your account identifier

CREATE SERVICE IF NOT EXISTS SPEAKER_IDENTIFICATION_SERVICE
    IN COMPUTE POOL SPEAKER_IDENTIFICATION_POOL
    MIN_INSTANCES = 1
    MAX_INSTANCES = 3
    MIN_READY_INSTANCES = 1
    AUTO_RESUME = TRUE
    EXTERNAL_ACCESS_INTEGRATIONS = (HF_ACCESS_INTEGRATION)
    COMMENT = 'Speaker identification service using ECAPA-TDNN model'
    FROM SPECIFICATION $$
    spec:
      containers:
      - name: speaker-service
        image: /<database>/<schema>/speaker_identification_repo/speaker-identification:v1
        env:
          SNOWFLAKE_DATABASE: MEETING_AGENT_DB
          SNOWFLAKE_SCHEMA: MEETING_AGENT
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
-- Step 7: Wait for Service to Start
-- ============================================================================
-- Check service status (wait until status = 'RUNNING')
-- DESCRIBE SERVICE SPEAKER_IDENTIFICATION_SERVICE;
-- SHOW SERVICES IN SCHEMA MEETING_AGENT_DB.MEETING_AGENT;

-- Check service logs if needed
-- CALL SYSTEM$GET_SERVICE_LOGS('SPEAKER_IDENTIFICATION_SERVICE', '0', 'speaker-service', 100);

-- ============================================================================
-- Step 8: Get Service Endpoint URL
-- ============================================================================
SHOW ENDPOINTS IN SERVICE SPEAKER_IDENTIFICATION_SERVICE;
-- Note the ingress_url for testing

-- ============================================================================
-- Verification Commands (run after service is up)
-- ============================================================================
-- SELECT SYSTEM$GET_SERVICE_STATUS('SPEAKER_IDENTIFICATION_SERVICE');
-- SHOW SERVICES IN SCHEMA MEETING_AGENT_DB.MEETING_AGENT;
