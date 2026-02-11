"""
Speaker Service Helper for Streamlit App

This module provides helper functions for the Streamlit app to interact
with the Model Registry-based speaker embedding service.

The functions wrap the SQL service functions and stored procedures,
providing a cleaner interface for the Streamlit app.

Usage in app.py:
    from speaker_helper import (
        extract_embedding_for_contribution,
        identify_speaker_by_embedding,
        create_voiceprint_for_speaker
    )
"""

import json
import logging
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def get_speaker_profiles(session) -> List[Dict[str, Any]]:
    """
    Get all speaker voiceprints as profiles for matching.
    
    Args:
        session: Snowpark session
        
    Returns:
        List of {speaker_id, speaker_name, embedding} dicts
    """
    try:
        results = session.sql("""
            SELECT speaker_id, speaker_name, embedding::VARCHAR as embedding
            FROM SPEAKER_VOICEPRINTS
            WHERE embedding IS NOT NULL
        """).collect()
        
        profiles = []
        for row in results:
            try:
                emb_str = row['EMBEDDING']
                emb = json.loads(emb_str) if isinstance(emb_str, str) else emb_str
                profiles.append({
                    "speaker_id": row['SPEAKER_ID'],
                    "speaker_name": row['SPEAKER_NAME'],
                    "embedding": emb
                })
            except Exception as e:
                logger.warning(f"Failed to parse profile: {e}")
        
        return profiles
    except Exception as e:
        logger.error(f"Error getting profiles: {e}")
        return []


def extract_embedding_for_contribution(
    session,
    call_id: str,
    contribution_id: str
) -> Tuple[bool, Optional[List[float]], Optional[str]]:
    """
    Extract embedding for a specific contribution.
    
    Args:
        session: Snowpark session
        call_id: Call ID
        contribution_id: Contribution ID
        
    Returns:
        Tuple of (success, embedding, error_message)
    """
    try:
        result = session.sql(f"""
            CALL EXTRACT_CONTRIBUTION_EMBEDDING('{call_id}', '{contribution_id}')
        """).collect()
        
        if not result:
            return False, None, "No result from extraction"
        
        data = result[0][0]
        if isinstance(data, str):
            data = json.loads(data)
        
        if data.get('status') == 'success':
            embedding = data.get('embedding')
            if isinstance(embedding, str):
                embedding = json.loads(embedding)
            return True, embedding, None
        else:
            return False, None, data.get('message', 'Unknown error')
            
    except Exception as e:
        return False, None, str(e)


def create_voiceprint_for_speaker(
    session,
    speaker_id: str,
    call_id: str,
    diarization_label: str
) -> Tuple[bool, Optional[str]]:
    """
    Create or update voiceprint for a speaker from a contribution.
    
    Args:
        session: Snowpark session
        speaker_id: Speaker ID
        call_id: Call ID
        diarization_label: Diarization label
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        result = session.sql(f"""
            CALL CREATE_SPEAKER_VOICEPRINT_FROM_CONTRIBUTION(
                '{speaker_id}', '{call_id}', '{diarization_label}'
            )
        """).collect()
        
        if not result:
            return False, "No result from voiceprint creation"
        
        data = result[0][0]
        if isinstance(data, str):
            data = json.loads(data)
        
        if data.get('status') == 'success':
            return True, None
        else:
            return False, data.get('message', 'Unknown error')
            
    except Exception as e:
        return False, str(e)


def identify_speaker_by_embedding(
    session,
    embedding: List[float],
    threshold: float = 0.75
) -> Tuple[bool, Optional[str], Optional[str], Optional[float]]:
    """
    Identify a speaker by matching embedding against voiceprints.
    
    Args:
        session: Snowpark session
        embedding: 192-dim embedding to match
        threshold: Match threshold
        
    Returns:
        Tuple of (matched, speaker_id, speaker_name, confidence)
    """
    try:
        profiles = get_speaker_profiles(session)
        if not profiles:
            return False, None, None, None
        
        query_json = json.dumps(embedding).replace("'", "''")
        profiles_json = json.dumps(profiles).replace("'", "''")
        
        result = session.sql(f"""
            SELECT SPEAKER_BATCH_MATCH('{query_json}', '{profiles_json}', {threshold})
        """).collect()
        
        if not result:
            return False, None, None, None
        
        data = result[0][0]
        if isinstance(data, str):
            data = json.loads(data)
        
        if data.get('matched'):
            return (
                True,
                data.get('speaker_id'),
                data.get('speaker_name'),
                data.get('confidence')
            )
        else:
            return False, None, None, data.get('confidence')
            
    except Exception as e:
        logger.error(f"Error identifying speaker: {e}")
        return False, None, None, None


def auto_identify_call_speakers(
    session,
    call_id: str,
    threshold: float = 0.75
) -> Tuple[int, int, Optional[str]]:
    """
    Automatically identify speakers in a call.
    
    Args:
        session: Snowpark session
        call_id: Call ID
        threshold: Match threshold
        
    Returns:
        Tuple of (identified_count, unidentified_count, error_message)
    """
    try:
        result = session.sql(f"""
            CALL AUTO_IDENTIFY_CALL_SPEAKERS('{call_id}', {threshold})
        """).collect()
        
        if not result:
            return 0, 0, "No result from auto-identification"
        
        data = result[0][0]
        if isinstance(data, str):
            data = json.loads(data)
        
        if data.get('status') == 'success':
            return (
                data.get('identified', 0),
                data.get('unidentified', 0),
                None
            )
        else:
            return 0, 0, data.get('message', 'Unknown error')
            
    except Exception as e:
        return 0, 0, str(e)


def compute_embedding_similarity(
    session,
    embedding1: List[float],
    embedding2: List[float],
    threshold: float = 0.75
) -> Tuple[Optional[float], bool, Optional[str]]:
    """
    Compute similarity between two embeddings.
    
    Args:
        session: Snowpark session
        embedding1: First embedding
        embedding2: Second embedding
        threshold: Match threshold
        
    Returns:
        Tuple of (similarity, is_match, error_message)
    """
    try:
        emb1_json = json.dumps(embedding1).replace("'", "''")
        emb2_json = json.dumps(embedding2).replace("'", "''")
        
        result = session.sql(f"""
            SELECT SPEAKER_SIMILARITY('{emb1_json}', '{emb2_json}', {threshold})
        """).collect()
        
        if not result:
            return None, False, "No result from similarity computation"
        
        data = result[0][0]
        if isinstance(data, str):
            data = json.loads(data)
        
        if data.get('error'):
            return None, False, data.get('error')
        
        return data.get('similarity'), data.get('match', False), None
        
    except Exception as e:
        return None, False, str(e)


def check_service_health(session) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if the speaker embedding service is healthy.
    
    Args:
        session: Snowpark session
        
    Returns:
        Tuple of (is_healthy, status_info)
    """
    try:
        result = session.sql("SELECT SPEAKER_EMBEDDING_HEALTH()").collect()
        
        if not result:
            return False, {"error": "No response from service"}
        
        data = result[0][0]
        if isinstance(data, str):
            data = json.loads(data)
        
        is_healthy = data.get('status') == 'healthy'
        return is_healthy, data
        
    except Exception as e:
        return False, {"error": str(e)}


def get_contribution_embedding(
    session,
    contribution_id: str
) -> Optional[List[float]]:
    """
    Get stored embedding for a contribution.
    
    Args:
        session: Snowpark session
        contribution_id: Contribution ID
        
    Returns:
        Embedding or None if not found
    """
    try:
        result = session.sql(f"""
            SELECT embedding::VARCHAR as embedding
            FROM CONTRIBUTION_EMBEDDINGS
            WHERE contribution_id = '{contribution_id}'
        """).collect()
        
        if result and result[0]['EMBEDDING']:
            emb_str = result[0]['EMBEDDING']
            return json.loads(emb_str) if isinstance(emb_str, str) else emb_str
        return None
        
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None


def store_contribution_embedding(
    session,
    call_id: str,
    contribution_id: str
) -> Tuple[bool, Optional[str]]:
    """
    Extract and store embedding for a contribution.
    
    Args:
        session: Snowpark session
        call_id: Call ID
        contribution_id: Contribution ID
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        result = session.sql(f"""
            CALL STORE_CONTRIBUTION_EMBEDDING('{call_id}', '{contribution_id}')
        """).collect()
        
        if not result:
            return False, "No result from storage"
        
        data = result[0][0]
        if isinstance(data, str):
            data = json.loads(data)
        
        if data.get('status') == 'success':
            return True, None
        else:
            return False, data.get('message', 'Unknown error')
            
    except Exception as e:
        return False, str(e)
