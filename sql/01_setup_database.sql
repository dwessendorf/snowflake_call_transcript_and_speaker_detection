-- ============================================================================
-- Snowflake Speaker Detection - Database Setup
-- ============================================================================
-- Run this script first to create the database, schema, and warehouse
-- Requires ACCOUNTADMIN or equivalent privileges
-- ============================================================================

-- Create database
CREATE DATABASE IF NOT EXISTS MEETING_AGENT_DB;

-- Create schema
CREATE SCHEMA IF NOT EXISTS MEETING_AGENT_DB.MEETING_AGENT;

-- Use the schema
USE SCHEMA MEETING_AGENT_DB.MEETING_AGENT;

-- Create warehouse (adjust size as needed)
CREATE WAREHOUSE IF NOT EXISTS MEETING_AGENT_WH
    WAREHOUSE_SIZE = 'SMALL'
    AUTO_SUSPEND = 300
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'Warehouse for Meeting Agent operations';

-- Grant usage
GRANT USAGE ON DATABASE MEETING_AGENT_DB TO ROLE ACCOUNTADMIN;
GRANT USAGE ON SCHEMA MEETING_AGENT_DB.MEETING_AGENT TO ROLE ACCOUNTADMIN;
GRANT USAGE ON WAREHOUSE MEETING_AGENT_WH TO ROLE ACCOUNTADMIN;

-- Verify setup
SELECT CURRENT_DATABASE(), CURRENT_SCHEMA();
