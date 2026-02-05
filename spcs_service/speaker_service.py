"""
Speaker Identification Service - Using SpeechBrain ECAPA-TDNN
Pre-baked model version that doesn't require HuggingFace gated model access
"""

import os
import json
import logging
import tempfile
import numpy as np
from typing import Optional, Dict, Any, List
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model instance
CLASSIFIER = None

def load_model():
    """Load the pre-baked speechbrain model"""
    global CLASSIFIER
    
    if CLASSIFIER is not None:
        return True
    
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        
        logger.info("Loading speechbrain ECAPA-TDNN model...")
        CLASSIFIER = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/root/.cache/speechbrain/spkrec-ecapa-voxceleb"
        )
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def extract_embedding(audio_file_path: str) -> Optional[np.ndarray]:
    """Extract speaker embedding from audio file"""
    try:
        if not load_model():
            return None
        
        import torch
        import torchaudio
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_file_path)
        
        # Resample if needed (model expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Extract embedding
        with torch.no_grad():
            embedding = CLASSIFIER.encode_batch(waveform)
            embedding = embedding.squeeze().cpu().numpy()
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
        
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        return None


def extract_embedding_segment(audio_file_path: str, start_time: float, end_time: float) -> Optional[np.ndarray]:
    """Extract embedding from a specific segment of audio"""
    try:
        if not load_model():
            return None
        
        import torch
        import torchaudio
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_file_path)
        
        # Calculate sample positions
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # Extract segment
        waveform = waveform[:, start_sample:end_sample]
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Minimum length check (need at least 0.5 seconds)
        min_samples = int(0.5 * 16000)
        if waveform.shape[1] < min_samples:
            logger.warning(f"Segment too short ({waveform.shape[1]} samples)")
            return None
        
        # Extract embedding
        with torch.no_grad():
            embedding = CLASSIFIER.encode_batch(waveform)
            embedding = embedding.squeeze().cpu().numpy()
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
        
    except Exception as e:
        logger.error(f"Error extracting segment embedding: {e}")
        return None


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    return float(np.dot(emb1, emb2))


def get_snowflake_connection():
    """Create Snowflake connection using SPCS OAuth tokens or Snowpark session"""
    # In SPCS, credentials are mounted at /snowflake/session/token
    token_path = '/snowflake/session/token'
    
    if os.path.exists(token_path):
        # Running inside SPCS - use Snowpark Session which handles auth automatically
        from snowflake.snowpark import Session
        
        return Session.builder.configs({
            "connection_name": "spcs"  # This uses built-in SPCS credentials
        }).create()
    else:
        # Local testing - use regular connector
        import snowflake.connector
        return snowflake.connector.connect(
            account=os.environ.get('SNOWFLAKE_ACCOUNT'),
            user=os.environ.get('SNOWFLAKE_USER'),
            password=os.environ.get('SNOWFLAKE_PASSWORD'),
            database=os.environ.get('SNOWFLAKE_DATABASE', 'MEETING_AGENT_DB'),
            schema=os.environ.get('SNOWFLAKE_SCHEMA', 'MEETING_AGENT'),
            warehouse=os.environ.get('SNOWFLAKE_WAREHOUSE', 'CAPSTONE_LOADING_WH')
        )


def download_from_stage(stage_path: str, local_path: str) -> bool:
    """Download file from Snowflake stage - only works locally, not in SPCS"""
    try:
        # Local - use connector
        import snowflake.connector
        conn = snowflake.connector.connect(
            account=os.environ.get('SNOWFLAKE_ACCOUNT'),
            user=os.environ.get('SNOWFLAKE_USER'),
            password=os.environ.get('SNOWFLAKE_PASSWORD'),
            database=os.environ.get('SNOWFLAKE_DATABASE', 'MEETING_AGENT_DB'),
            schema=os.environ.get('SNOWFLAKE_SCHEMA', 'MEETING_AGENT'),
            warehouse=os.environ.get('SNOWFLAKE_WAREHOUSE', 'CAPSTONE_LOADING_WH')
        )
        cursor = conn.cursor()
        cursor.execute(f"GET {stage_path} file://{local_path}")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to download from stage: {e}")
        return False


def extract_embedding_from_bytes(audio_bytes: bytes, start_time: float = None, end_time: float = None) -> Optional[np.ndarray]:
    """Extract embedding from audio bytes - supports MP3, WAV, and other formats via pydub"""
    try:
        if not load_model():
            return None
        
        import torch
        from pydub import AudioSegment
        import io
        from scipy.io import wavfile
        
        # Load audio using pydub (handles MP3, WAV, etc.)
        try:
            # Try to detect format from bytes
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            logger.info(f"Loaded audio: {len(audio)}ms, {audio.frame_rate}Hz, {audio.channels} channels")
        except Exception as e:
            logger.error(f"Failed to load audio with pydub: {e}")
            return None
        
        # Extract segment if specified
        if start_time is not None and end_time is not None:
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            audio = audio[start_ms:end_ms]
            logger.info(f"Extracted segment {start_time}-{end_time}s: {len(audio)}ms")
        
        # Convert to mono if needed
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Resample to 16kHz if needed
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        
        # Check minimum length (0.5 seconds = 500ms)
        if len(audio) < 500:
            logger.warning(f"Audio segment too short ({len(audio)}ms)")
            return None
        
        # Export to WAV bytes and load with scipy
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        # Read WAV using scipy (no torchaudio backend needed)
        sample_rate, audio_data = wavfile.read(wav_buffer)
        logger.info(f"Loaded WAV: sample_rate={sample_rate}, shape={audio_data.shape}, dtype={audio_data.dtype}")
        
        # Convert to float and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Convert to torch tensor with shape [1, num_samples]
        waveform = torch.from_numpy(audio_data).unsqueeze(0)
        logger.info(f"Waveform tensor: shape={waveform.shape}")
        
        # Extract embedding
        with torch.no_grad():
            embedding = CLASSIFIER.encode_batch(waveform)
            embedding = embedding.squeeze().cpu().numpy()
        
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        logger.info(f"Extracted embedding: shape={embedding.shape}")
        return embedding
            
    except Exception as e:
        logger.error(f"Error extracting embedding from bytes: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_loaded = CLASSIFIER is not None
    return jsonify({
        "status": "healthy" if model_loaded else "initializing",
        "model_loaded": model_loaded,
        "embedding_dim": 192
    })


@app.route('/extract-embedding', methods=['POST'])
def extract_embedding_endpoint():
    """Extract embedding from audio - accepts base64 audio data or stage path"""
    try:
        data = request.get_json()
        audio_base64 = data.get('audio_base64')
        audio_url = data.get('audio_url')
        start_time = data.get('start_time')
        end_time = data.get('end_time')
        
        if audio_base64:
            # Decode base64 audio
            import base64
            audio_bytes = base64.b64decode(audio_base64)
            
            # Extract embedding from bytes
            if start_time is not None and end_time is not None:
                embedding = extract_embedding_from_bytes(audio_bytes, float(start_time), float(end_time))
            else:
                embedding = extract_embedding_from_bytes(audio_bytes)
            
            if embedding is None:
                return jsonify({"status": "error", "message": "Failed to extract embedding"}), 500
            
            return jsonify({
                "status": "success",
                "embedding": embedding.tolist(),
                "embedding_dim": len(embedding)
            })
            
        elif audio_url:
            # Stage path (only works outside SPCS)
            with tempfile.TemporaryDirectory() as tmpdir:
                if audio_url.startswith('@') or audio_url.startswith('MEETING_'):
                    if not download_from_stage(audio_url, tmpdir):
                        return jsonify({"status": "error", "message": "Failed to download from stage"}), 500
                    files = os.listdir(tmpdir)
                    if files:
                        local_file = os.path.join(tmpdir, files[0])
                    else:
                        return jsonify({"status": "error", "message": "No file downloaded"}), 500
                else:
                    return jsonify({"status": "error", "message": "Invalid audio_url"}), 400
                
                if start_time is not None and end_time is not None:
                    embedding = extract_embedding_segment(local_file, float(start_time), float(end_time))
                else:
                    embedding = extract_embedding(local_file)
                
                if embedding is None:
                    return jsonify({"status": "error", "message": "Failed to extract embedding"}), 500
                
                return jsonify({
                    "status": "success",
                    "embedding": embedding.tolist(),
                    "embedding_dim": len(embedding)
                })
        else:
            return jsonify({"status": "error", "message": "audio_base64 or audio_url required"}), 400
            
    except Exception as e:
        logger.error(f"Error in extract-embedding: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/extract-embedding-b64', methods=['POST'])
def extract_embedding_b64():
    """Snowflake service function endpoint for base64 audio extraction"""
    try:
        data = request.get_json()
        rows = data.get('data', [])
        
        results = []
        for row in rows:
            row_idx = row[0]
            audio_base64 = row[1] if len(row) > 1 else None
            start_time = float(row[2]) if len(row) > 2 and row[2] is not None else None
            end_time = float(row[3]) if len(row) > 3 and row[3] is not None else None
            
            if not audio_base64:
                results.append([row_idx, None, "No audio data"])
                continue
            
            try:
                import base64
                audio_bytes = base64.b64decode(audio_base64)
                
                embedding = extract_embedding_from_bytes(audio_bytes, start_time, end_time)
                
                if embedding is not None:
                    results.append([row_idx, embedding.tolist(), None])
                else:
                    results.append([row_idx, None, "Extraction failed"])
            except Exception as e:
                results.append([row_idx, None, str(e)])
        
        return jsonify({"data": results})
        
    except Exception as e:
        logger.error(f"Error in extract-embedding-b64: {e}")
        return jsonify({"data": [[0, None, str(e)]]}), 500


@app.route('/extract-embedding-sf', methods=['POST'])
def extract_embedding_sf():
    """Snowflake service function compatible endpoint"""
    try:
        data = request.get_json()
        rows = data.get('data', [])
        
        results = []
        for row in rows:
            row_idx = row[0]
            audio_url = row[1] if len(row) > 1 else None
            
            if not audio_url:
                results.append([row_idx, None, "No audio_url provided"])
                continue
            
            with tempfile.TemporaryDirectory() as tmpdir:
                if audio_url.startswith('@'):
                    if not download_from_stage(audio_url, tmpdir):
                        results.append([row_idx, None, "Failed to download"])
                        continue
                    files = os.listdir(tmpdir)
                    if files:
                        local_file = os.path.join(tmpdir, files[0])
                    else:
                        results.append([row_idx, None, "No file downloaded"])
                        continue
                else:
                    results.append([row_idx, None, "Invalid path"])
                    continue
                
                embedding = extract_embedding(local_file)
                if embedding is not None:
                    results.append([row_idx, embedding.tolist(), None])
                else:
                    results.append([row_idx, None, "Extraction failed"])
        
        return jsonify({"data": results})
        
    except Exception as e:
        logger.error(f"Error in extract-embedding-sf: {e}")
        return jsonify({"data": [[0, None, str(e)]]}), 500


@app.route('/extract-embedding-url', methods=['POST'])
def extract_embedding_url():
    """Extract embedding from audio at a presigned URL with segment support
    
    Request format:
    {
        "data": [[row_idx, presigned_url, start_time, end_time], ...]
    }
    
    Response format for Snowflake service function (single return value):
    {
        "data": [[row_idx, result_dict], ...]
    }
    where result_dict is {"embedding": [...], "status": "success"} or {"error": "...", "status": "error"}
    """
    try:
        import requests as http_requests  # Avoid name collision with flask request
        
        data = request.get_json()
        rows = data.get('data', [])
        
        results = []
        for row in rows:
            row_idx = row[0]
            audio_url = row[1] if len(row) > 1 else None
            start_time = float(row[2]) if len(row) > 2 and row[2] is not None else None
            end_time = float(row[3]) if len(row) > 3 and row[3] is not None else None
            
            if not audio_url:
                results.append([row_idx, {"status": "error", "error": "No audio_url provided"}])
                continue
            
            try:
                # Download audio from presigned URL
                logger.info(f"Downloading audio from URL (segment: {start_time}-{end_time}s)")
                response = http_requests.get(audio_url, timeout=120)
                
                if response.status_code != 200:
                    results.append([row_idx, {"status": "error", "error": f"HTTP {response.status_code}"}])
                    continue
                
                audio_bytes = response.content
                logger.info(f"Downloaded {len(audio_bytes)} bytes")
                
                # Extract embedding from bytes with segment
                embedding = extract_embedding_from_bytes(audio_bytes, start_time, end_time)
                
                if embedding is not None:
                    results.append([row_idx, {"status": "success", "embedding": embedding.tolist()}])
                else:
                    results.append([row_idx, {"status": "error", "error": "Extraction failed - segment may be too short"}])
                    
            except Exception as e:
                logger.error(f"Error processing row {row_idx}: {e}")
                results.append([row_idx, {"status": "error", "error": str(e)}])
        
        return jsonify({"data": results})
        
    except Exception as e:
        logger.error(f"Error in extract-embedding-url: {e}")
        return jsonify({"data": [[0, {"status": "error", "error": str(e)}]]}), 500


@app.route('/match', methods=['POST'])
def match_embedding():
    """Match an embedding against stored profiles"""
    try:
        data = request.get_json()
        embedding = data.get('embedding')
        threshold = data.get('threshold', 0.75)
        
        if not embedding:
            return jsonify({"status": "error", "message": "embedding required"}), 400
        
        query_emb = np.array(embedding)
        
        # Get all speaker profiles with embeddings
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT speaker_id, speaker_name, embedding 
            FROM SPEAKERS s
            JOIN SPEAKER_VOICEPRINTS v ON s.speaker_id = v.speaker_id
            WHERE v.embedding IS NOT NULL
        """)
        
        best_match = None
        best_score = 0
        
        for row in cursor:
            speaker_id, speaker_name, emb_json = row
            try:
                profile_emb = np.array(json.loads(emb_json))
                score = cosine_similarity(query_emb, profile_emb)
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = {"speaker_id": speaker_id, "speaker_name": speaker_name}
            except:
                continue
        
        cursor.close()
        conn.close()
        
        if best_match:
            return jsonify({
                "status": "success",
                "matched": True,
                "speaker_id": best_match["speaker_id"],
                "speaker_name": best_match["speaker_name"],
                "confidence": best_score
            })
        else:
            return jsonify({
                "status": "success",
                "matched": False,
                "message": "No match above threshold"
            })
            
    except Exception as e:
        logger.error(f"Error in match: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/identify', methods=['POST'])
def identify_speakers():
    """Identify speakers in a meeting by extracting embeddings and matching profiles"""
    try:
        data = request.get_json()
        rows = data.get('data', [])
        
        results = []
        for row in rows:
            row_idx = row[0]
            meeting_id = row[1] if len(row) > 1 else None
            audio_path = row[2] if len(row) > 2 else None
            threshold = float(row[3]) if len(row) > 3 else 0.75
            
            if not meeting_id or not audio_path:
                results.append([row_idx, json.dumps({"status": "error", "message": "Missing parameters"})])
                continue
            
            try:
                # Load model if needed
                if not load_model():
                    results.append([row_idx, json.dumps({"status": "error", "message": "Failed to load models"})])
                    continue
                
                # Download audio
                with tempfile.TemporaryDirectory() as tmpdir:
                    if not download_from_stage(audio_path, tmpdir):
                        results.append([row_idx, json.dumps({"status": "error", "message": "Failed to download audio"})])
                        continue
                    
                    files = os.listdir(tmpdir)
                    if not files:
                        results.append([row_idx, json.dumps({"status": "error", "message": "No file downloaded"})])
                        continue
                    
                    local_file = os.path.join(tmpdir, files[0])
                    
                    # Get contributions for this meeting
                    conn = get_snowflake_connection()
                    cursor = conn.cursor()
                    cursor.execute(f"""
                        SELECT contribution_id, diarization_label, start_time_seconds, end_time_seconds
                        FROM MEETING_CONTRIBUTIONS
                        WHERE meeting_id = '{meeting_id}'
                        AND classification_status = 'pending'
                        ORDER BY segment_number
                    """)
                    contributions = cursor.fetchall()
                    
                    # Get existing speaker profiles
                    cursor.execute("""
                        SELECT s.speaker_id, s.speaker_name, v.embedding
                        FROM SPEAKERS s
                        JOIN SPEAKER_VOICEPRINTS v ON s.speaker_id = v.speaker_id
                        WHERE v.embedding IS NOT NULL
                    """)
                    profiles = []
                    for sp_row in cursor:
                        try:
                            profiles.append({
                                "speaker_id": sp_row[0],
                                "speaker_name": sp_row[1],
                                "embedding": np.array(json.loads(sp_row[2]))
                            })
                        except:
                            pass
                    
                    identified = 0
                    unidentified = 0
                    
                    for contrib in contributions:
                        contrib_id, label, start_t, end_t = contrib
                        
                        # Extract embedding for this segment
                        if start_t is not None and end_t is not None:
                            emb = extract_embedding_segment(local_file, start_t, end_t)
                        else:
                            emb = None
                        
                        if emb is None:
                            unidentified += 1
                            # Still mark for manual review
                            cursor.execute(f"""
                                UPDATE MEETING_CONTRIBUTIONS 
                                SET classification_status = 'needs_review'
                                WHERE contribution_id = '{contrib_id}'
                            """)
                            continue
                        
                        # Match against profiles
                        best_match = None
                        best_score = 0
                        
                        for profile in profiles:
                            score = cosine_similarity(emb, profile["embedding"])
                            if score > best_score and score >= threshold:
                                best_score = score
                                best_match = profile
                        
                        if best_match:
                            # Auto-identify
                            cursor.execute(f"""
                                UPDATE MEETING_CONTRIBUTIONS 
                                SET identified_speaker_id = '{best_match["speaker_id"]}',
                                    identification_method = 'voice_embedding',
                                    identification_confidence = {best_score},
                                    classification_status = 'auto_identified'
                                WHERE contribution_id = '{contrib_id}'
                            """)
                            identified += 1
                        else:
                            # Mark for manual review
                            cursor.execute(f"""
                                UPDATE MEETING_CONTRIBUTIONS 
                                SET classification_status = 'needs_review',
                                    embedding = '{json.dumps(emb.tolist())}'
                                WHERE contribution_id = '{contrib_id}'
                            """)
                            unidentified += 1
                    
                    conn.commit()
                    cursor.close()
                    conn.close()
                    
                    results.append([row_idx, json.dumps({
                        "status": "success",
                        "meeting_id": meeting_id,
                        "identified": identified,
                        "needs_review": unidentified,
                        "total_profiles": len(profiles)
                    })])
                    
            except Exception as e:
                logger.error(f"Error processing meeting {meeting_id}: {e}")
                results.append([row_idx, json.dumps({"status": "error", "message": str(e)})])
        
        return jsonify({"data": results})
        
    except Exception as e:
        logger.error(f"Error in identify: {e}")
        return jsonify({"data": [[0, json.dumps({"status": "error", "message": str(e)})]]}), 500


@app.route('/enroll', methods=['POST'])
def enroll_speaker():
    """Enroll a new speaker profile from audio"""
    try:
        data = request.get_json()
        speaker_id = data.get('speaker_id')
        speaker_name = data.get('speaker_name')
        audio_url = data.get('audio_url')
        
        if not speaker_id or not audio_url:
            return jsonify({"status": "error", "message": "speaker_id and audio_url required"}), 400
        
        with tempfile.TemporaryDirectory() as tmpdir:
            if audio_url.startswith('@'):
                if not download_from_stage(audio_url, tmpdir):
                    return jsonify({"status": "error", "message": "Failed to download"}), 500
                files = os.listdir(tmpdir)
                if files:
                    local_file = os.path.join(tmpdir, files[0])
                else:
                    return jsonify({"status": "error", "message": "No file"}), 500
            else:
                return jsonify({"status": "error", "message": "Invalid audio_url"}), 400
            
            embedding = extract_embedding(local_file)
            if embedding is None:
                return jsonify({"status": "error", "message": "Failed to extract embedding"}), 500
            
            # Store in database
            conn = get_snowflake_connection()
            cursor = conn.cursor()
            
            # Check if speaker exists
            cursor.execute(f"SELECT speaker_id FROM SPEAKERS WHERE speaker_id = '{speaker_id}'")
            if not cursor.fetchone():
                cursor.execute(f"""
                    INSERT INTO SPEAKERS (speaker_id, speaker_name, created_at)
                    VALUES ('{speaker_id}', '{speaker_name or speaker_id}', CURRENT_TIMESTAMP)
                """)
            
            # Add voiceprint
            import uuid
            voiceprint_id = f"VP_{uuid.uuid4().hex[:12].upper()}"
            cursor.execute(f"""
                INSERT INTO SPEAKER_VOICEPRINTS (voiceprint_id, speaker_id, embedding, created_at)
                VALUES ('{voiceprint_id}', '{speaker_id}', '{json.dumps(embedding.tolist())}', CURRENT_TIMESTAMP)
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return jsonify({
                "status": "success",
                "speaker_id": speaker_id,
                "voiceprint_id": voiceprint_id,
                "embedding_dim": len(embedding)
            })
            
    except Exception as e:
        logger.error(f"Error in enroll: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    # Pre-load model
    load_model()
    app.run(host='0.0.0.0', port=8080, debug=False)
