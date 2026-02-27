#!/bin/bash
# ============================================================================
# Call Transcript & Speaker Detection - Master Deployment Script
# ============================================================================
#
# This script deploys the complete solution to Snowflake.
#
# Prerequisites:
#   - Snowflake CLI (snow) installed and configured
#   - Python 3.11+ with snowflake-ml-python, snowflake-snowpark-python
#   - ACCOUNTADMIN role or equivalent privileges
#   - Cortex AI functions enabled in your region
#
# Usage:
#   ./deploy.sh [CONNECTION_NAME]
#
# Example:
#   ./deploy.sh DWESSENDORF_AWS1
#
# ============================================================================

set -e

# Configuration
CONNECTION=${1:-"default"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "Call Transcript & Speaker Detection Deploy"
echo "============================================"
echo "Connection: $CONNECTION"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Step 1: Run Snowflake setup SQL
echo "[1/4] Setting up Snowflake infrastructure..."
snow sql -f "$SCRIPT_DIR/01_snowflake_setup.sql" --connection "$CONNECTION"
echo "✓ Infrastructure setup complete"
echo ""

# Step 2: Register and deploy the GPU model
echo "[2/4] Registering and deploying speaker embedding model..."
cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/register_model.py" --connection "$CONNECTION"
echo "✓ Model deployed"
echo ""

# Step 3: Create service functions
echo "[3/4] Creating service functions..."
snow sql -f "$SCRIPT_DIR/02_service_functions.sql" --connection "$CONNECTION"
echo "✓ Service functions created"
echo ""

# Step 4: Deploy Streamlit app
echo "[4/4] Deploying Streamlit app..."
cd "$PROJECT_ROOT/streamlit_app"
snow streamlit deploy --replace --connection "$CONNECTION"
echo "✓ Streamlit app deployed"
echo ""

echo "============================================"
echo "Deployment Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Upload audio files to @CALL_RECORDINGS stage"
echo "  2. Register calls in the CALLS table"
echo "  3. Run CALL TRANSCRIBE_CALL('<call_id>') to transcribe"
echo "  4. Use the Streamlit app for speaker assignments"
echo ""
echo "Test the service:"
echo "  SELECT SPEAKER_EMBEDDING_HEALTH();"
echo ""
