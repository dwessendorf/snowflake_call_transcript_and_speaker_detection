#!/usr/bin/env python3
"""
Register and deploy the Speaker Embedding model to Snowflake Model Registry.

This script:
1. Registers the SpeechBrain ECAPA-TDNN model in Snowflake Model Registry
2. Deploys it as a real-time inference service

Prerequisites:
- snowflake-ml-python >= 1.5.0
- snowflake-snowpark-python
- Connection configured in ~/.snowflake/connections.toml or environment variables

Usage:
    python register_model.py [--connection CONNECTION_NAME]

Example:
    python register_model.py --connection my_snowflake_connection
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'spcs_service'))


def get_session(connection_name: str = None):
    """Create Snowpark session from connection config."""
    from snowflake.snowpark import Session
    
    if connection_name:
        logger.info(f"Connecting using connection: {connection_name}")
        session = Session.builder.config("connection_name", connection_name).create()
    else:
        # Try default connection
        logger.info("Connecting using default connection")
        session = Session.builder.create()
    
    account = session.sql("SELECT CURRENT_ACCOUNT()").collect()[0][0]
    logger.info(f"Connected to account: {account}")
    return session


def register_and_deploy(session, model_name: str = "SPEAKER_EMBEDDING", version: str = "v9"):
    """Register model and deploy as service."""
    from speaker_model_registry import register_model, deploy_service
    
    # Set context
    session.sql("USE DATABASE CALL_TRANSCRIPTS_DB").collect()
    session.sql("USE SCHEMA TRANSCRIPTS").collect()
    session.sql("USE WAREHOUSE CALL_TRANSCRIPTS_WH").collect()
    
    logger.info(f"Registering model {model_name}/{version}...")
    
    # Register the model
    mv = register_model(
        session=session,
        model_name=model_name,
        version_name=version,
        database="CALL_TRANSCRIPTS_DB",
        schema="TRANSCRIPTS"
    )
    
    logger.info(f"Model registered successfully")
    
    # Deploy as service
    service_name = f"{model_name}_SVC_{version.upper()}"
    logger.info(f"Deploying service {service_name}...")
    
    deploy_service(
        model_version=mv,
        service_name=service_name,
        compute_pool="SYSTEM_COMPUTE_POOL_GPU",  # Use system GPU pool
        gpu_requests="1",
        max_instances=1,
        min_instances=1
    )
    
    logger.info(f"Service {service_name} deployed successfully!")
    logger.info(f"\nNext steps:")
    logger.info(f"1. Run deploy/02_service_functions.sql to create service functions")
    logger.info(f"2. Test with: SELECT SPEAKER_EMBEDDING_HEALTH();")
    
    return service_name


def main():
    parser = argparse.ArgumentParser(description="Register and deploy Speaker Embedding model")
    parser.add_argument("--connection", "-c", help="Snowflake connection name from connections.toml")
    parser.add_argument("--model-name", default="SPEAKER_EMBEDDING", help="Model name in registry")
    parser.add_argument("--version", default="v9", help="Model version")
    args = parser.parse_args()
    
    try:
        session = get_session(args.connection)
        service_name = register_and_deploy(session, args.model_name, args.version)
        logger.info(f"\n✓ Deployment complete: {service_name}")
        return 0
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
