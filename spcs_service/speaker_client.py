"""
Speaker Embedding Client for Snowflake Model Registry Service

This module provides a Python client for interacting with the speaker embedding
model deployed via Snowflake Model Registry. It can be used both within
Snowflake (Streamlit, stored procedures) and externally via REST API.

Usage within Snowflake (Streamlit/Snowpark):
    from speaker_client import SpeakerClient
    
    session = get_active_session()
    client = SpeakerClient(session)
    
    # Extract embedding from presigned URL
    result = client.extract_embedding_url(presigned_url, start_time, end_time)
    
    # Match against known speakers
    match = client.identify_speaker(embedding, threshold=0.75)

Usage externally via REST API:
    from speaker_client import SpeakerRestClient
    
    client = SpeakerRestClient(
        endpoint="https://xxx.snowflakecomputing.app",
        token="your_pat_token"
    )
    result = client.extract_embedding_b64(audio_base64)
"""

import json
import logging
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embedding extraction"""
    success: bool
    embedding: Optional[List[float]] = None
    error: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'EmbeddingResult':
        if data.get('status') == 'success':
            emb = data.get('embedding')
            if isinstance(emb, str):
                emb = json.loads(emb)
            return cls(success=True, embedding=emb)
        return cls(success=False, error=data.get('error'))


@dataclass
class MatchResult:
    """Result from speaker matching"""
    matched: bool
    speaker_id: Optional[str] = None
    speaker_name: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MatchResult':
        return cls(
            matched=data.get('matched', False),
            speaker_id=data.get('speaker_id'),
            speaker_name=data.get('speaker_name'),
            confidence=data.get('confidence'),
            error=data.get('error')
        )


@dataclass
class SimilarityResult:
    """Result from similarity computation"""
    similarity: Optional[float] = None
    match: bool = False
    error: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SimilarityResult':
        return cls(
            similarity=data.get('similarity'),
            match=data.get('match', False),
            error=data.get('error')
        )


class SpeakerClient:
    """
    Client for speaker embedding service via Snowflake SQL functions.
    
    Use this client within Snowflake (Streamlit apps, stored procedures, notebooks).
    It calls the service functions created by 05_create_functions_model_registry.sql.
    """
    
    def __init__(self, session, schema: str = "CALL_TRANSCRIPTS_DB.TRANSCRIPTS"):
        """
        Initialize the client.
        
        Args:
            session: Snowpark session (from get_active_session() or Session.builder)
            schema: Schema where service functions are defined
        """
        self.session = session
        self.schema = schema
    
    def _run_query(self, sql: str) -> Any:
        """Execute SQL and return first result"""
        try:
            result = self.session.sql(sql).collect()
            if result and len(result) > 0:
                return result[0][0]
            return None
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def _parse_result(self, result: Any) -> dict:
        """Parse result from service function"""
        if result is None:
            return {"status": "error", "error": "No result"}
        if isinstance(result, str):
            return json.loads(result)
        if isinstance(result, dict):
            return result
        # Snowflake VARIANT type
        return dict(result)
    
    def health_check(self) -> dict:
        """Check service health"""
        result = self._run_query(f"SELECT {self.schema}.SPEAKER_EMBEDDING_HEALTH()")
        return self._parse_result(result)
    
    def extract_embedding_url(
        self,
        audio_url: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> EmbeddingResult:
        """
        Extract speaker embedding from audio at a presigned URL.
        
        Args:
            audio_url: Presigned URL from GET_PRESIGNED_URL()
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            
        Returns:
            EmbeddingResult with 192-dim embedding or error
        """
        start = start_time if start_time is not None else 'NULL'
        end = end_time if end_time is not None else 'NULL'
        
        # Escape single quotes in URL
        safe_url = audio_url.replace("'", "''")
        
        sql = f"""
            SELECT {self.schema}.SPEAKER_EMBEDDING_URL(
                '{safe_url}', {start}, {end}
            )
        """
        result = self._run_query(sql)
        return EmbeddingResult.from_dict(self._parse_result(result))
    
    def extract_embedding_b64(
        self,
        audio_base64: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> EmbeddingResult:
        """
        Extract speaker embedding from base64-encoded audio.
        
        Args:
            audio_base64: Base64-encoded audio bytes
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            
        Returns:
            EmbeddingResult with 192-dim embedding or error
        """
        start = start_time if start_time is not None else 'NULL'
        end = end_time if end_time is not None else 'NULL'
        
        sql = f"""
            SELECT {self.schema}.SPEAKER_EMBEDDING_B64(
                '{audio_base64}', {start}, {end}
            )
        """
        result = self._run_query(sql)
        return EmbeddingResult.from_dict(self._parse_result(result))
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
        threshold: float = 0.75
    ) -> SimilarityResult:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First 192-dim embedding
            embedding2: Second 192-dim embedding
            threshold: Match threshold (default 0.75)
            
        Returns:
            SimilarityResult with similarity score and match flag
        """
        emb1_json = json.dumps(embedding1).replace("'", "''")
        emb2_json = json.dumps(embedding2).replace("'", "''")
        
        sql = f"""
            SELECT {self.schema}.SPEAKER_SIMILARITY(
                '{emb1_json}', '{emb2_json}', {threshold}
            )
        """
        result = self._run_query(sql)
        return SimilarityResult.from_dict(self._parse_result(result))
    
    def batch_match(
        self,
        query_embedding: List[float],
        profiles: List[Dict[str, Any]],
        threshold: float = 0.75
    ) -> MatchResult:
        """
        Match an embedding against multiple speaker profiles.
        
        Args:
            query_embedding: 192-dim embedding to match
            profiles: List of {speaker_id, speaker_name, embedding} dicts
            threshold: Match threshold (default 0.75)
            
        Returns:
            MatchResult with best matching speaker or no match
        """
        query_json = json.dumps(query_embedding).replace("'", "''")
        profiles_json = json.dumps(profiles).replace("'", "''")
        
        sql = f"""
            SELECT {self.schema}.SPEAKER_BATCH_MATCH(
                '{query_json}', '{profiles_json}', {threshold}
            )
        """
        result = self._run_query(sql)
        return MatchResult.from_dict(self._parse_result(result))
    
    def get_speaker_profiles(self) -> List[Dict[str, Any]]:
        """
        Get all speaker voiceprints as profiles for matching.
        
        Returns:
            List of {speaker_id, speaker_name, embedding} dicts
        """
        sql = f"""
            SELECT speaker_id, speaker_name, embedding::VARCHAR as embedding
            FROM {self.schema}.SPEAKER_VOICEPRINTS
            WHERE embedding IS NOT NULL
        """
        results = self.session.sql(sql).collect()
        
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
                logger.warning(f"Failed to parse profile for {row['SPEAKER_ID']}: {e}")
        
        return profiles
    
    def identify_speaker(
        self,
        embedding: List[float],
        threshold: float = 0.75
    ) -> MatchResult:
        """
        Identify a speaker by matching embedding against all known voiceprints.
        
        This is a convenience method that fetches all profiles and matches.
        
        Args:
            embedding: 192-dim embedding to identify
            threshold: Match threshold (default 0.75)
            
        Returns:
            MatchResult with identified speaker or no match
        """
        profiles = self.get_speaker_profiles()
        if not profiles:
            return MatchResult(matched=False, error="No speaker profiles available")
        
        return self.batch_match(embedding, profiles, threshold)
    
    def get_presigned_url(
        self,
        stage: str,
        file_path: str,
        expiration_seconds: int = 3600
    ) -> Optional[str]:
        """
        Get a presigned URL for a file in a Snowflake stage.
        
        Args:
            stage: Stage name (e.g., "@CALL_RECORDINGS")
            file_path: Path within the stage
            expiration_seconds: URL expiration time
            
        Returns:
            Presigned URL or None
        """
        sql = f"""
            SELECT GET_PRESIGNED_URL('{stage}', '{file_path}', {expiration_seconds})
        """
        result = self._run_query(sql)
        return result if result else None
    
    def extract_contribution_embedding(
        self,
        call_id: str,
        contribution_id: str
    ) -> EmbeddingResult:
        """
        Extract embedding for a specific contribution.
        
        Args:
            call_id: Call ID
            contribution_id: Contribution ID
            
        Returns:
            EmbeddingResult with embedding or error
        """
        sql = f"""
            CALL {self.schema}.EXTRACT_CONTRIBUTION_EMBEDDING(
                '{call_id}', '{contribution_id}'
            )
        """
        result = self._run_query(sql)
        return EmbeddingResult.from_dict(self._parse_result(result))
    
    def create_voiceprint(
        self,
        speaker_id: str,
        call_id: str,
        diarization_label: str
    ) -> dict:
        """
        Create or update a speaker voiceprint from a contribution.
        
        Args:
            speaker_id: Speaker ID
            call_id: Call ID containing the audio
            diarization_label: Diarization label to use
            
        Returns:
            Result dict with status
        """
        sql = f"""
            CALL {self.schema}.CREATE_SPEAKER_VOICEPRINT_FROM_CONTRIBUTION(
                '{speaker_id}', '{call_id}', '{diarization_label}'
            )
        """
        result = self._run_query(sql)
        return self._parse_result(result)
    
    def auto_identify_call(
        self,
        call_id: str,
        threshold: float = 0.75
    ) -> dict:
        """
        Automatically identify speakers in a call.
        
        Args:
            call_id: Call ID to process
            threshold: Match threshold
            
        Returns:
            Result dict with identification summary
        """
        sql = f"""
            CALL {self.schema}.AUTO_IDENTIFY_CALL_SPEAKERS(
                '{call_id}', {threshold}
            )
        """
        result = self._run_query(sql)
        return self._parse_result(result)


class SpeakerRestClient:
    """
    Client for speaker embedding service via REST API.
    
    Use this client for external access to the Model Registry service.
    Requires a PAT token for authentication.
    """
    
    def __init__(
        self,
        endpoint: str,
        token: str,
        timeout: int = 120
    ):
        """
        Initialize the REST client.
        
        Args:
            endpoint: Service endpoint URL (from SHOW ENDPOINTS IN SERVICE)
            token: Snowflake PAT token for authentication
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint.rstrip('/')
        self.token = token
        self.timeout = timeout
        
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("requests library required for REST client")
    
    def _make_request(self, path: str, data: dict) -> dict:
        """Make authenticated request to service"""
        url = f"{self.endpoint}/{path.lstrip('/')}"
        headers = {
            "Authorization": f'Snowflake Token="{self.token}"',
            "Content-Type": "application/json"
        }
        
        response = self.requests.post(
            url,
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            return {"status": "error", "error": f"HTTP {response.status_code}"}
        
        result = response.json()
        # Model Registry returns {"data": [[row_idx, result], ...]}
        if "data" in result and result["data"]:
            return result["data"][0][1] if len(result["data"][0]) > 1 else result["data"][0][0]
        return result
    
    def health_check(self) -> dict:
        """Check service health"""
        return self._make_request("health", {"data": [["dummy"]]})
    
    def extract_embedding_b64(
        self,
        audio_base64: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> EmbeddingResult:
        """Extract embedding from base64 audio"""
        data = {
            "data": [[audio_base64, start_time, end_time]]
        }
        result = self._make_request("extract-embedding", data)
        return EmbeddingResult.from_dict(result)
    
    def extract_embedding_url(
        self,
        audio_url: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> EmbeddingResult:
        """Extract embedding from presigned URL"""
        data = {
            "data": [[audio_url, start_time, end_time]]
        }
        result = self._make_request("extract-embedding-url", data)
        return EmbeddingResult.from_dict(result)
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
        threshold: float = 0.75
    ) -> SimilarityResult:
        """Compute similarity between embeddings"""
        data = {
            "data": [[json.dumps(embedding1), json.dumps(embedding2), threshold]]
        }
        result = self._make_request("compute-similarity", data)
        return SimilarityResult.from_dict(result)
    
    def batch_match(
        self,
        query_embedding: List[float],
        profiles: List[Dict[str, Any]],
        threshold: float = 0.75
    ) -> MatchResult:
        """Match embedding against profiles"""
        data = {
            "data": [[json.dumps(query_embedding), json.dumps(profiles), threshold]]
        }
        result = self._make_request("batch-match", data)
        return MatchResult.from_dict(result)


# ============================================================================
# Utility Functions
# ============================================================================

def cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings locally.
    
    Useful for quick comparisons without calling the service.
    """
    import numpy as np
    vec1 = np.array(emb1)
    vec2 = np.array(emb2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def find_best_match(
    query_embedding: List[float],
    profiles: List[Dict[str, Any]],
    threshold: float = 0.75
) -> MatchResult:
    """
    Find best matching speaker from profiles locally.
    
    Useful when you want to avoid service calls for simple matching.
    """
    import numpy as np
    
    query = np.array(query_embedding)
    best_match = None
    best_score = 0.0
    
    for profile in profiles:
        try:
            profile_emb = np.array(profile.get("embedding", []))
            if len(profile_emb) != 192:
                continue
            
            score = float(np.dot(query, profile_emb))
            if score > best_score:
                best_score = score
                best_match = profile
        except:
            continue
    
    if best_match and best_score >= threshold:
        return MatchResult(
            matched=True,
            speaker_id=best_match.get("speaker_id"),
            speaker_name=best_match.get("speaker_name"),
            confidence=best_score
        )
    
    return MatchResult(matched=False, confidence=best_score if best_score > 0 else None)
