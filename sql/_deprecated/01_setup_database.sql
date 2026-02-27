-- ============================================================================
-- Snowflake Call Transcript and Speaker Detection - Database Setup
-- ============================================================================
-- Run this script first to create the database, schema, and warehouse
-- Requires ACCOUNTADMIN or equivalent privileges
-- ============================================================================

-- Create database
CREATE DATABASE IF NOT EXISTS CALL_TRANSCRIPTS_DB;

-- Create schema
CREATE SCHEMA IF NOT EXISTS CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- Use the schema
USE SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- Create warehouse (adjust size as needed)
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

-- Verify setup
SELECT CURRENT_DATABASE(), CURRENT_SCHEMA();
