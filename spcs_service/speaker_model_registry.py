"""
Speaker Embedding Model for Snowflake Model Registry

This module provides a CustomModel wrapper for the SpeechBrain ECAPA-TDNN
speaker embedding model, designed for deployment via Snowflake Model Registry
with Real-time Inference REST API.

Uses soundfile for audio processing with torchaudio fallback.

Features:
- extract_embedding: Extract 192-dim speaker embeddings from audio
- extract_embedding_url: Extract embeddings from presigned URLs (for Snowflake stages)
- compute_similarity: Compare two embeddings
- batch_match: Match embeddings against a list of known speaker profiles

Usage:
    from snowflake.snowpark import Session
    from speaker_model_registry import register_model, deploy_service
    
    session = Session.builder.configs(...).create()
    mv = register_model(session, "SPEAKER_EMBEDDING", "v1")
    
    # Deploy as real-time service
    deploy_service(mv, "speaker_embedding_svc")
"""

# Patch torchaudio before importing speechbrain (compatibility fix for torchaudio 2.10+)
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['default']

import io
import base64
import logging
import json
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Custom Model for Snowflake Model Registry
# ============================================================================

def create_model_class():
    """
    Factory function to create the CustomModel class.
    This avoids import errors when snowflake-ml-python is not installed.
    """
    from snowflake.ml.model import custom_model
    
    class SpeakerEmbeddingModel(custom_model.CustomModel):
        """
        Speaker embedding extraction model using SpeechBrain ECAPA-TDNN.
        
        Produces 192-dimensional normalized speaker embeddings suitable for:
        - Speaker verification (1:1 comparison)
        - Speaker identification (1:N matching)
        - Speaker diarization (clustering)
        
        Input: Base64-encoded audio (WAV format recommended, 16kHz mono)
        Output: 192-dimensional normalized embedding vector
        """
        
        EMBEDDING_DIM = 192
        DEFAULT_THRESHOLD = 0.75
        
        def __init__(self, context: custom_model.ModelContext) -> None:
            super().__init__(context)
            self._classifier = None
        
        def _load_model(self):
            """Lazy load the SpeechBrain model"""
            if self._classifier is None:
                import os
                
                # Set HuggingFace cache to writable temp directory
                # SPCS containers may not have write access to /root/.cache
                os.environ['HF_HOME'] = '/tmp/huggingface_cache'
                os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/huggingface_cache'
                os.makedirs('/tmp/huggingface_cache', exist_ok=True)
                
                # CRITICAL: Patch torchaudio BEFORE importing speechbrain
                # torchaudio 2.1+ removed list_audio_backends(), which SpeechBrain requires
                import torchaudio
                if not hasattr(torchaudio, 'list_audio_backends'):
                    torchaudio.list_audio_backends = lambda: ['default']
                    logger.info("Patched torchaudio.list_audio_backends for compatibility")
                
                from speechbrain.inference.speaker import EncoderClassifier
                
                logger.info("Loading SpeechBrain ECAPA-TDNN model...")
                self._classifier = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="/tmp/speechbrain_cache/spkrec-ecapa-voxceleb"
                )
                logger.info("Model loaded successfully")
            return self._classifier
        
        def _process_audio_bytes(
            self, 
            audio_bytes: bytes, 
            start_time: Optional[float] = None,
            end_time: Optional[float] = None
        ) -> Optional[np.ndarray]:
            """
            Process audio bytes and extract speaker embedding.
            
            Args:
                audio_bytes: Raw audio file bytes (WAV format)
                start_time: Optional segment start time in seconds
                end_time: Optional segment end time in seconds
                
            Returns:
                192-dimensional normalized embedding or None on error
            """
            import torch
            import numpy as np
            
            try:
                # Try loading with soundfile first (more reliable)
                waveform = None
                sample_rate = None
                try:
                    import soundfile as sf
                    audio_buffer = io.BytesIO(audio_bytes)
                    data, sample_rate = sf.read(audio_buffer)
                    waveform = torch.from_numpy(data.astype(np.float32))
                    # Ensure correct shape (channels, samples)
                    if len(waveform.shape) == 1:
                        waveform = waveform.unsqueeze(0)
                    elif waveform.shape[0] > waveform.shape[1]:  # If (samples, channels), transpose
                        waveform = waveform.T
                    logger.info(f"Loaded audio with soundfile: {waveform.shape}, {sample_rate}Hz")
                except Exception as sf_err:
                    logger.warning(f"soundfile failed: {sf_err}, trying torchaudio")
                    # Fallback to torchaudio
                    try:
                        import torchaudio
                        audio_buffer = io.BytesIO(audio_bytes)
                        waveform, sample_rate = torchaudio.load(audio_buffer, format='wav')
                        logger.info(f"Loaded audio with torchaudio: {waveform.shape}, {sample_rate}Hz")
                    except Exception as ta_err:
                        logger.error(f"Both soundfile and torchaudio failed. soundfile: {sf_err}, torchaudio: {ta_err}")
                        raise ValueError(f"Failed to load audio: soundfile={sf_err}, torchaudio={ta_err}")
                
                if waveform is None:
                    raise ValueError("Failed to load audio: waveform is None")
                
                logger.debug(f"Loaded audio: {waveform.shape}, {sample_rate}Hz")
                
                # Extract segment if specified
                if start_time is not None and end_time is not None:
                    start_sample = int(start_time * sample_rate)
                    end_sample = int(end_time * sample_rate)
                    waveform = waveform[:, start_sample:end_sample]
                    logger.debug(f"Extracted segment {start_time}-{end_time}s")
                
                # Resample to 16kHz if needed (model requirement)
                if sample_rate != 16000:
                    try:
                        import torchaudio
                        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                        waveform = resampler(waveform)
                    except (ImportError, Exception) as e:
                        # Simple linear interpolation resampling as fallback
                        logger.info(f"Using fallback resampling: {e}")
                        import torch.nn.functional as F
                        ratio = 16000 / sample_rate
                        new_length = int(waveform.shape[1] * ratio)
                        waveform = F.interpolate(waveform.unsqueeze(0), size=new_length, mode='linear', align_corners=False).squeeze(0)
                    sample_rate = 16000
                
                # Convert stereo to mono
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Check minimum length (0.5 seconds at 16kHz = 8000 samples)
                min_samples = int(0.5 * 16000)
                if waveform.shape[1] < min_samples:
                    logger.warning(f"Audio too short: {waveform.shape[1]} samples")
                    return None
                
                # Load model and extract embedding
                classifier = self._load_model()
                
                with torch.no_grad():
                    embedding = classifier.encode_batch(waveform)
                    embedding = embedding.squeeze().cpu().numpy()
                
                # L2 normalize
                embedding = embedding / np.linalg.norm(embedding)
                
                return embedding
                
            except Exception as e:
                import traceback
                logger.error(f"Error processing audio: {e}\n{traceback.format_exc()}")
                return None
        
        def _download_from_url(self, url: str, timeout: int = 120) -> Optional[bytes]:
            """Download audio from a URL (presigned URL from Snowflake stage)"""
            import requests
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200:
                    return response.content
                logger.error(f"HTTP {response.status_code} downloading from URL")
                return None
            except Exception as e:
                logger.error(f"Error downloading from URL: {e}")
                return None
        
        @staticmethod
        def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
            """Compute cosine similarity between normalized embeddings"""
            return float(np.dot(emb1, emb2))
        
        # ====================================================================
        # Inference APIs
        # ====================================================================
        
        @custom_model.inference_api
        def extract_embedding(self, input_df: pd.DataFrame) -> pd.DataFrame:
            """
            Extract speaker embeddings from base64-encoded audio.
            
            Input DataFrame columns:
                - audio_base64 (str): Base64-encoded audio bytes (WAV format)
                - start_time (float, optional): Segment start time in seconds
                - end_time (float, optional): Segment end time in seconds
            
            Output DataFrame columns:
                - embedding (str): JSON array of 192-dimensional embedding
                - status (str): "success" or "error"
                - error (str): Error message if status is "error"
            """
            results = []
            
            for idx, row in input_df.iterrows():
                audio_b64 = row.get("audio_base64") or (row.iloc[0] if len(row) > 0 else None)
                start_time = row.get("start_time") or (row.iloc[1] if len(row) > 1 else None)
                end_time = row.get("end_time") or (row.iloc[2] if len(row) > 2 else None)
                
                if not audio_b64:
                    results.append({
                        "embedding": None,
                        "status": "error",
                        "error": "No audio_base64 provided"
                    })
                    continue
                
                try:
                    audio_bytes = base64.b64decode(audio_b64)
                    start_t = float(start_time) if start_time is not None else None
                    end_t = float(end_time) if end_time is not None else None
                    
                    embedding = self._process_audio_bytes(audio_bytes, start_t, end_t)
                    
                    if embedding is not None:
                        results.append({
                            "embedding": json.dumps(embedding.tolist()),
                            "status": "success",
                            "error": None
                        })
                    else:
                        results.append({
                            "embedding": None,
                            "status": "error",
                            "error": "Extraction failed - audio may be too short"
                        })
                        
                except Exception as e:
                    results.append({
                        "embedding": None,
                        "status": "error",
                        "error": str(e)
                    })
            
            return pd.DataFrame(results)
        
        @custom_model.inference_api
        def extract_embedding_url(self, input_df: pd.DataFrame) -> pd.DataFrame:
            """
            Extract speaker embeddings from audio at presigned URLs.
            
            This is the primary method for Snowflake integration - it accepts
            presigned URLs from GET_PRESIGNED_URL() for stage files.
            
            Input DataFrame columns:
                - audio_url (str): Presigned URL to audio file
                - start_time (float, optional): Segment start time in seconds
                - end_time (float, optional): Segment end time in seconds
            
            Output DataFrame columns:
                - embedding (str): JSON array of 192-dimensional embedding
                - status (str): "success" or "error"
                - error (str): Error message if status is "error"
            """
            results = []
            
            for idx, row in input_df.iterrows():
                audio_url = row.get("audio_url") or (row.iloc[0] if len(row) > 0 else None)
                start_time = row.get("start_time") or (row.iloc[1] if len(row) > 1 else None)
                end_time = row.get("end_time") or (row.iloc[2] if len(row) > 2 else None)
                
                if not audio_url:
                    results.append({
                        "embedding": None,
                        "status": "error",
                        "error": "No audio_url provided"
                    })
                    continue
                
                try:
                    # Download audio from URL
                    audio_bytes = self._download_from_url(audio_url)
                    if audio_bytes is None:
                        results.append({
                            "embedding": None,
                            "status": "error",
                            "error": "Failed to download audio from URL"
                        })
                        continue
                    
                    start_t = float(start_time) if start_time is not None else None
                    end_t = float(end_time) if end_time is not None else None
                    
                    embedding = self._process_audio_bytes(audio_bytes, start_t, end_t)
                    
                    if embedding is not None:
                        results.append({
                            "embedding": json.dumps(embedding.tolist()),
                            "status": "success",
                            "error": None
                        })
                    else:
                        results.append({
                            "embedding": None,
                            "status": "error",
                            "error": "Extraction failed - segment may be too short"
                        })
                        
                except Exception as e:
                    results.append({
                        "embedding": None,
                        "status": "error",
                        "error": str(e)
                    })
            
            return pd.DataFrame(results)
        
        @custom_model.inference_api
        def extract_embedding_batch(self, input_df: pd.DataFrame) -> pd.DataFrame:
            """
            Extract speaker embeddings for multiple segments from a single audio URL.
            
            Downloads audio ONCE and extracts embeddings for all segments - much faster
            than calling extract_embedding_url multiple times.
            
            Input DataFrame columns:
                - audio_url (str): Presigned URL to audio file
                - segments_json (str): JSON array of segments: [{"id": "...", "start": 0.0, "end": 5.0}, ...]
            
            Output DataFrame columns:
                - embeddings_json (str): JSON dict mapping segment id to embedding array
                - processed (int): Number of segments successfully processed
                - total (int): Total number of segments
                - status (str): "success" or "error"
                - error (str): Error message if status is "error"
            """
            results = []
            
            for idx, row in input_df.iterrows():
                audio_url = row.get("audio_url") or (row.iloc[0] if len(row) > 0 else None)
                segments_json = row.get("segments_json") or (row.iloc[1] if len(row) > 1 else None)
                
                if not audio_url:
                    results.append({
                        "embeddings_json": None,
                        "processed": 0,
                        "total": 0,
                        "status": "error",
                        "error": "No audio_url provided"
                    })
                    continue
                
                if not segments_json:
                    results.append({
                        "embeddings_json": None,
                        "processed": 0,
                        "total": 0,
                        "status": "error",
                        "error": "No segments_json provided"
                    })
                    continue
                
                try:
                    # Parse segments
                    segments = json.loads(segments_json) if isinstance(segments_json, str) else segments_json
                    
                    if not segments:
                        results.append({
                            "embeddings_json": None,
                            "processed": 0,
                            "total": 0,
                            "status": "error",
                            "error": "Empty segments list"
                        })
                        continue
                    
                    # Download audio ONCE
                    logger.info(f"Batch: downloading audio for {len(segments)} segments...")
                    audio_bytes = self._download_from_url(audio_url, timeout=300)
                    
                    if audio_bytes is None:
                        results.append({
                            "embeddings_json": None,
                            "processed": 0,
                            "total": len(segments),
                            "status": "error",
                            "error": "Failed to download audio from URL"
                        })
                        continue
                    
                    logger.info(f"Batch: downloaded {len(audio_bytes)} bytes, processing {len(segments)} segments")
                    
                    # Process all segments from the same audio bytes
                    embeddings = {}
                    processed = 0
                    errors = []
                    
                    for seg in segments:
                        seg_id = seg.get("id", f"seg_{processed}")
                        start_time = float(seg.get("start", 0))
                        end_time = float(seg.get("end", 0))
                        
                        # Skip very short segments
                        if (end_time - start_time) < 0.5:
                            errors.append(f"{seg_id}: too short")
                            continue
                        
                        try:
                            embedding = self._process_audio_bytes(audio_bytes, start_time, end_time)
                            
                            if embedding is not None:
                                embeddings[seg_id] = embedding.tolist()
                                processed += 1
                            else:
                                errors.append(f"{seg_id}: extraction failed")
                        except Exception as e:
                            errors.append(f"{seg_id}: {str(e)}")
                    
                    logger.info(f"Batch complete: {processed}/{len(segments)} segments processed")
                    
                    results.append({
                        "embeddings_json": json.dumps(embeddings),
                        "processed": processed,
                        "total": len(segments),
                        "status": "success",
                        "error": json.dumps(errors[:10]) if errors else None
                    })
                    
                except Exception as e:
                    results.append({
                        "embeddings_json": None,
                        "processed": 0,
                        "total": 0,
                        "status": "error",
                        "error": str(e)
                    })
            
            return pd.DataFrame(results)
        
        @custom_model.inference_api
        def compute_similarity(self, input_df: pd.DataFrame) -> pd.DataFrame:
            """
            Compute cosine similarity between two embeddings.
            
            Input DataFrame columns:
                - embedding1 (str): JSON array of first embedding
                - embedding2 (str): JSON array of second embedding
                - threshold (float, optional): Match threshold (default 0.75)
            
            Output DataFrame columns:
                - similarity (float): Cosine similarity score (-1.0 to 1.0)
                - match (bool): True if similarity >= threshold
                - error (str): Error message if failed
            """
            results = []
            
            for idx, row in input_df.iterrows():
                emb1_str = row.get("embedding1") or (row.iloc[0] if len(row) > 0 else None)
                emb2_str = row.get("embedding2") or (row.iloc[1] if len(row) > 1 else None)
                threshold = row.get("threshold") or (row.iloc[2] if len(row) > 2 else self.DEFAULT_THRESHOLD)
                
                if emb1_str is None or emb2_str is None:
                    results.append({
                        "similarity": None,
                        "match": False,
                        "error": "Both embeddings required"
                    })
                    continue
                
                try:
                    # Parse JSON if string
                    emb1 = json.loads(emb1_str) if isinstance(emb1_str, str) else emb1_str
                    emb2 = json.loads(emb2_str) if isinstance(emb2_str, str) else emb2_str
                    
                    vec1 = np.array(emb1)
                    vec2 = np.array(emb2)
                    
                    similarity = self._cosine_similarity(vec1, vec2)
                    threshold_val = float(threshold) if threshold else self.DEFAULT_THRESHOLD
                    
                    results.append({
                        "similarity": similarity,
                        "match": similarity >= threshold_val,
                        "error": None
                    })
                except Exception as e:
                    results.append({
                        "similarity": None,
                        "match": False,
                        "error": str(e)
                    })
            
            return pd.DataFrame(results)
        
        @custom_model.inference_api
        def batch_match(self, input_df: pd.DataFrame) -> pd.DataFrame:
            """
            Match an embedding against multiple known speaker profiles.
            
            This is the core speaker identification function - given an embedding
            and a list of known speaker profiles, find the best match.
            
            Input DataFrame columns:
                - query_embedding (str): JSON array of embedding to match
                - profiles (str): JSON array of {speaker_id, speaker_name, embedding}
                - threshold (float, optional): Match threshold (default 0.75)
            
            Output DataFrame columns:
                - matched (bool): True if a match was found above threshold
                - speaker_id (str): ID of matched speaker or null
                - speaker_name (str): Name of matched speaker or null
                - confidence (float): Similarity score of best match
                - error (str): Error message if failed
            """
            results = []
            
            for idx, row in input_df.iterrows():
                query_str = row.get("query_embedding") or (row.iloc[0] if len(row) > 0 else None)
                profiles_str = row.get("profiles") or (row.iloc[1] if len(row) > 1 else None)
                threshold = row.get("threshold") or (row.iloc[2] if len(row) > 2 else self.DEFAULT_THRESHOLD)
                
                if not query_str or not profiles_str:
                    results.append({
                        "matched": False,
                        "speaker_id": None,
                        "speaker_name": None,
                        "confidence": None,
                        "error": "query_embedding and profiles required"
                    })
                    continue
                
                try:
                    query_emb = np.array(json.loads(query_str) if isinstance(query_str, str) else query_str)
                    profiles = json.loads(profiles_str) if isinstance(profiles_str, str) else profiles_str
                    threshold_val = float(threshold) if threshold else self.DEFAULT_THRESHOLD
                    
                    best_match = None
                    best_score = 0.0
                    
                    for profile in profiles:
                        profile_emb = np.array(profile.get("embedding", []))
                        if len(profile_emb) != self.EMBEDDING_DIM:
                            continue
                        
                        score = self._cosine_similarity(query_emb, profile_emb)
                        if score > best_score:
                            best_score = score
                            best_match = profile
                    
                    if best_match and best_score >= threshold_val:
                        results.append({
                            "matched": True,
                            "speaker_id": best_match.get("speaker_id"),
                            "speaker_name": best_match.get("speaker_name"),
                            "confidence": best_score,
                            "error": None
                        })
                    else:
                        results.append({
                            "matched": False,
                            "speaker_id": None,
                            "speaker_name": None,
                            "confidence": best_score if best_score > 0 else None,
                            "error": None
                        })
                        
                except Exception as e:
                    results.append({
                        "matched": False,
                        "speaker_id": None,
                        "speaker_name": None,
                        "confidence": None,
                        "error": str(e)
                    })
            
            return pd.DataFrame(results)
        
        @custom_model.inference_api
        def health(self, input_df: pd.DataFrame) -> pd.DataFrame:
            """
            Health check endpoint.
            
            Input: Any DataFrame (ignored)
            Output: Status information
            """
            model_loaded = self._classifier is not None
            return pd.DataFrame([{
                "status": "healthy",
                "model_loaded": model_loaded,
                "embedding_dim": self.EMBEDDING_DIM,
                "default_threshold": self.DEFAULT_THRESHOLD
            }])
    
    return SpeakerEmbeddingModel


# ============================================================================
# Registration and Deployment Helper Functions
# ============================================================================

def register_model(
    session,
    model_name: str = "SPEAKER_EMBEDDING",
    version_name: str = "v1",
    database: str = "CALL_TRANSCRIPTS_DB",
    schema: str = "TRANSCRIPTS"
):
    """
    Register the speaker embedding model in Snowflake Model Registry.
    
    Args:
        session: Snowpark Session
        model_name: Name for the model in registry
        version_name: Version identifier
        database: Target database
        schema: Target schema
        
    Returns:
        ModelVersion object for deployment
    """
    from snowflake.ml.registry import Registry
    from snowflake.ml.model import model_signature
    
    # Create model class and instance
    SpeakerEmbeddingModel = create_model_class()
    from snowflake.ml.model import custom_model
    
    model_context = custom_model.ModelContext()
    model_instance = SpeakerEmbeddingModel(model_context)
    
    # Define signatures for all inference APIs
    extract_sig = model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(name="audio_base64", dtype=model_signature.DataType.STRING),
            model_signature.FeatureSpec(name="start_time", dtype=model_signature.DataType.DOUBLE),
            model_signature.FeatureSpec(name="end_time", dtype=model_signature.DataType.DOUBLE),
        ],
        outputs=[
            model_signature.FeatureSpec(name="embedding", dtype=model_signature.DataType.STRING),
            model_signature.FeatureSpec(name="status", dtype=model_signature.DataType.STRING),
            model_signature.FeatureSpec(name="error", dtype=model_signature.DataType.STRING),
        ]
    )
    
    extract_url_sig = model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(name="audio_url", dtype=model_signature.DataType.STRING),
            model_signature.FeatureSpec(name="start_time", dtype=model_signature.DataType.DOUBLE),
            model_signature.FeatureSpec(name="end_time", dtype=model_signature.DataType.DOUBLE),
        ],
        outputs=[
            model_signature.FeatureSpec(name="embedding", dtype=model_signature.DataType.STRING),
            model_signature.FeatureSpec(name="status", dtype=model_signature.DataType.STRING),
            model_signature.FeatureSpec(name="error", dtype=model_signature.DataType.STRING),
        ]
    )
    
    # Batch extraction signature - downloads audio once, extracts all segments
    extract_batch_sig = model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(name="audio_url", dtype=model_signature.DataType.STRING),
            model_signature.FeatureSpec(name="segments_json", dtype=model_signature.DataType.STRING),
        ],
        outputs=[
            model_signature.FeatureSpec(name="embeddings_json", dtype=model_signature.DataType.STRING),
            model_signature.FeatureSpec(name="processed", dtype=model_signature.DataType.INT64),
            model_signature.FeatureSpec(name="total", dtype=model_signature.DataType.INT64),
            model_signature.FeatureSpec(name="status", dtype=model_signature.DataType.STRING),
            model_signature.FeatureSpec(name="error", dtype=model_signature.DataType.STRING),
        ]
    )
    
    similarity_sig = model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(name="embedding1", dtype=model_signature.DataType.STRING),
            model_signature.FeatureSpec(name="embedding2", dtype=model_signature.DataType.STRING),
            model_signature.FeatureSpec(name="threshold", dtype=model_signature.DataType.DOUBLE),
        ],
        outputs=[
            model_signature.FeatureSpec(name="similarity", dtype=model_signature.DataType.DOUBLE),
            model_signature.FeatureSpec(name="match", dtype=model_signature.DataType.BOOL),
            model_signature.FeatureSpec(name="error", dtype=model_signature.DataType.STRING),
        ]
    )
    
    batch_match_sig = model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(name="query_embedding", dtype=model_signature.DataType.STRING),
            model_signature.FeatureSpec(name="profiles", dtype=model_signature.DataType.STRING),
            model_signature.FeatureSpec(name="threshold", dtype=model_signature.DataType.DOUBLE),
        ],
        outputs=[
            model_signature.FeatureSpec(name="matched", dtype=model_signature.DataType.BOOL),
            model_signature.FeatureSpec(name="speaker_id", dtype=model_signature.DataType.STRING),
            model_signature.FeatureSpec(name="speaker_name", dtype=model_signature.DataType.STRING),
            model_signature.FeatureSpec(name="confidence", dtype=model_signature.DataType.DOUBLE),
            model_signature.FeatureSpec(name="error", dtype=model_signature.DataType.STRING),
        ]
    )
    
    health_sig = model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(name="dummy", dtype=model_signature.DataType.STRING),
        ],
        outputs=[
            model_signature.FeatureSpec(name="status", dtype=model_signature.DataType.STRING),
            model_signature.FeatureSpec(name="model_loaded", dtype=model_signature.DataType.BOOL),
            model_signature.FeatureSpec(name="embedding_dim", dtype=model_signature.DataType.INT64),
            model_signature.FeatureSpec(name="default_threshold", dtype=model_signature.DataType.DOUBLE),
        ]
    )
    
    # Open registry
    reg = Registry(
        session=session,
        database_name=database,
        schema_name=schema
    )
    
    logger.info(f"Logging model {model_name}/{version_name} to registry...")
    
    mv = reg.log_model(
        model=model_instance,
        model_name=model_name,
        version_name=version_name,
        pip_requirements=[
            "speechbrain>=1.0.0",
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "soundfile>=0.12.0",  # For reliable audio loading (requires libsndfile)
            "numpy>=1.20.0",
            "requests>=2.25.0",
            "huggingface_hub<0.25",  # Fix for use_auth_token deprecation
        ],
        signatures={
            "extract_embedding": extract_sig,
            "extract_embedding_url": extract_url_sig,
            "extract_embedding_batch": extract_batch_sig,
            "compute_similarity": similarity_sig,
            "batch_match": batch_match_sig,
            "health": health_sig,
        },
        comment="SpeechBrain ECAPA-TDNN speaker embedding model (192-dim) with URL support",
    )
    
    logger.info(f"Model registered: {model_name}/{version_name}")
    return mv


def deploy_service(
    model_version,
    service_name: str = "SPEAKER_EMBEDDING_SERVICE",
    compute_pool: str = "SYSTEM_COMPUTE_POOL_GPU",
    gpu_requests: str = "1",
    max_instances: int = 3,
    min_instances: int = 1
):
    """
    Deploy the registered model as a real-time inference service.
    
    Args:
        model_version: ModelVersion object from register_model()
        service_name: Name for the SPCS service
        compute_pool: Compute pool (GPU recommended for performance)
        gpu_requests: Number of GPUs per instance
        max_instances: Maximum auto-scaling instances
        min_instances: Minimum instances (for availability)
        
    Returns:
        Service info dict
    """
    logger.info(f"Deploying service {service_name}...")
    
    model_version.create_service(
        service_name=service_name,
        service_compute_pool=compute_pool,
        ingress_enabled=True,
        gpu_requests=gpu_requests,
        max_instances=max_instances,
        min_instances=min_instances,
    )
    
    services = model_version.list_services()
    logger.info(f"Service deployed: {services}")
    
    return services


def get_service_info(session, service_name: str = "SPEAKER_EMBEDDING_SERVICE"):
    """Get information about a deployed service."""
    try:
        result = session.sql(f"SHOW ENDPOINTS IN SERVICE {service_name}").collect()
        if result:
            return {
                "service_name": service_name,
                "endpoints": [dict(r) for r in result]
            }
    except Exception as e:
        logger.error(f"Error getting service info: {e}")
    return None


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """
    Example: Register and deploy the speaker embedding model.
    
    Run:
        export SNOWFLAKE_CONNECTION_NAME=your_connection
        python speaker_model_registry.py
    """
    from snowflake.snowpark import Session
    import os
    
    connection_name = os.environ.get("SNOWFLAKE_CONNECTION_NAME", "default")
    session = Session.builder.config("connection_name", connection_name).create()
    
    print(f"Connected to Snowflake: {session.get_current_account()}")
    
    # Register model
    mv = register_model(
        session=session,
        model_name="SPEAKER_EMBEDDING",
        version_name="v1"
    )
    
    # Deploy service
    services = deploy_service(
        model_version=mv,
        service_name="SPEAKER_EMBEDDING_SERVICE",
        compute_pool="SYSTEM_COMPUTE_POOL_GPU",
        gpu_requests="1"
    )
    
    print("\n" + "=" * 60)
    print("DEPLOYMENT COMPLETE")
    print("=" * 60)
    print(f"\nService info: {services}")
    print("\nAvailable endpoints:")
    print("  - /extract-embedding      : Extract from base64 audio")
    print("  - /extract-embedding-url  : Extract from presigned URL")
    print("  - /compute-similarity     : Compare two embeddings")
    print("  - /batch-match            : Match against speaker profiles")
    print("  - /health                 : Health check")
    
    session.close()


if __name__ == "__main__":
    example_usage()
