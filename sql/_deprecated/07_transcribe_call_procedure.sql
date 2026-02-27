-- ============================================================================
-- Snowflake Call Transcript and Speaker Detection - Transcribe Call Procedure
-- ============================================================================
-- Creates the TRANSCRIBE_CALL procedure that uses Cortex AI_TRANSCRIBE
-- with speaker diarization and merges consecutive segments from the same speaker
-- ============================================================================

USE SCHEMA CALL_TRANSCRIPTS_DB.TRANSCRIPTS;

-- ============================================================================
-- Procedure: Transcribe a call recording using Cortex AI_TRANSCRIBE
-- ============================================================================
-- This procedure:
-- 1. Takes a call_id and retrieves the recording path
-- 2. Calls AI_TRANSCRIBE with speaker diarization enabled
-- 3. Handles both old format (paragraphs) and new format (segments)
-- 4. Merges consecutive segments from the same speaker into single contributions
-- 5. Inserts merged segments into CALL_CONTRIBUTIONS table
-- 6. Updates the call status and speaker count
--
-- The merging logic ensures that if speaker A says multiple consecutive
-- paragraphs before speaker B speaks, all of speaker A's text is combined
-- into a single contribution with the full time span.

CREATE OR REPLACE PROCEDURE TRANSCRIBE_CALL(P_CALL_ID VARCHAR)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.11'
PACKAGES = ('snowflake-snowpark-python')
HANDLER = 'transcribe_call'
EXECUTE AS OWNER
AS $$
import json
import hashlib
from datetime import datetime

def transcribe_call(session, p_call_id):
    try:
        call_result = session.sql(f"SELECT recording_path, title FROM CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS WHERE call_id = '{p_call_id}'").collect()
        
        if not call_result:
            return {"status": "error", "message": f"Call {p_call_id} not found"}
        
        recording_path = call_result[0]["RECORDING_PATH"]
        title = call_result[0]["TITLE"] or p_call_id
        
        if not recording_path:
            return {"status": "error", "message": "No recording path specified"}
        
        session.sql(f"UPDATE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS SET transcription_status = 'processing', updated_at = CURRENT_TIMESTAMP() WHERE call_id = '{p_call_id}'").collect()
        
        stage_path = recording_path.lstrip("@")
        parts = stage_path.split("/")
        stage_name = "@" + "/".join(parts[:-1])
        file_name = parts[-1]
        
        transcription_result = session.sql(f"SELECT SNOWFLAKE.CORTEX.AI_TRANSCRIBE(TO_FILE('{stage_name}', '{file_name}'), OBJECT_CONSTRUCT('timestamp_granularity', 'speaker')) as transcription").collect()
        
        if not transcription_result:
            session.sql(f"UPDATE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS SET transcription_status = 'error', updated_at = CURRENT_TIMESTAMP() WHERE call_id = '{p_call_id}'").collect()
            return {"status": "error", "message": "Transcription failed"}
        
        transcript_data = transcription_result[0]["TRANSCRIPTION"]
        if isinstance(transcript_data, str):
            transcript_data = json.loads(transcript_data)
        
        # Handle both old format (results.channels.alternatives.paragraphs) and new format (segments)
        segments = []
        
        # Try new format first (direct segments array)
        if "segments" in transcript_data and transcript_data["segments"]:
            raw_segments = transcript_data["segments"]
            for seg in raw_segments:
                segments.append({
                    "speaker": seg.get("speaker_label", "unknown"),
                    "text": seg.get("text", ""),
                    "start": float(seg.get("start", 0)),
                    "end": float(seg.get("end", 0))
                })
        # Fall back to old format (nested paragraphs)
        elif "results" in transcript_data:
            paragraphs = transcript_data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("paragraphs", {}).get("paragraphs", [])
            for seg in paragraphs:
                text = " ".join([s.get("text", "") for s in seg.get("sentences", [])])
                segments.append({
                    "speaker": seg.get("speaker", "unknown"),
                    "text": text,
                    "start": float(seg.get("start", 0)),
                    "end": float(seg.get("end", 0))
                })
        
        original_count = len(segments)
        
        # Merge consecutive segments from the same speaker
        # This ensures that if speaker A has multiple paragraphs before speaker B speaks,
        # all of A's text is combined into a single contribution
        merged_segments = []
        current_segment = None
        
        for seg in segments:
            speaker_label = seg["speaker"]
            text = seg["text"]
            start_time = seg["start"]
            end_time = seg["end"]
            
            if current_segment is None:
                current_segment = {
                    "speaker": speaker_label,
                    "text": text,
                    "start_time": start_time,
                    "end_time": end_time
                }
            elif current_segment["speaker"] == speaker_label:
                current_segment["text"] += " " + text
                current_segment["end_time"] = end_time
            else:
                merged_segments.append(current_segment)
                current_segment = {
                    "speaker": speaker_label,
                    "text": text,
                    "start_time": start_time,
                    "end_time": end_time
                }
        
        # Do not forget the last segment
        if current_segment is not None:
            merged_segments.append(current_segment)
        
        segment_count = 0
        speaker_labels = set()
        
        for seg in merged_segments:
            segment_count += 1
            speaker_label = seg["speaker"]
            speaker_labels.add(speaker_label)
            
            text = seg["text"]
            text_escaped = text.replace("'", "''")
            start_time = seg["start_time"]
            end_time = seg["end_time"]
            duration = end_time - start_time
            word_count = len(text.split())
            
            contrib_id = hashlib.md5(f"{p_call_id}_{segment_count}_{start_time}".encode()).hexdigest()[:12]
            contrib_id = f"CONTRIB_{contrib_id.upper()}"
            
            session.sql(f"INSERT INTO CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALL_CONTRIBUTIONS (contribution_id, call_id, segment_number, diarization_label, text_content, start_time_seconds, end_time_seconds, duration_seconds, word_count) VALUES ('{contrib_id}', '{p_call_id}', {segment_count}, '{speaker_label}', '{text_escaped}', {start_time}, {end_time}, {duration}, {word_count})").collect()
        
        num_speakers = len(speaker_labels)
        session.sql(f"UPDATE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS SET transcription_status = 'completed', total_speakers = {num_speakers}, unidentified_speakers = {num_speakers}, updated_at = CURRENT_TIMESTAMP() WHERE call_id = '{p_call_id}'").collect()
        
        return {
            "status": "success", 
            "call_id": p_call_id, 
            "segments": segment_count, 
            "speakers_detected": num_speakers, 
            "original_segments": original_count, 
            "merged_segments": len(merged_segments)
        }
        
    except Exception as e:
        session.sql(f"UPDATE CALL_TRANSCRIPTS_DB.TRANSCRIPTS.CALLS SET transcription_status = 'error', updated_at = CURRENT_TIMESTAMP() WHERE call_id = '{p_call_id}'").collect()
        return {"status": "error", "message": str(e)}
$$;

-- ============================================================================
-- Usage Examples
-- ============================================================================
-- Transcribe a call:
-- CALL TRANSCRIBE_CALL('CALL_12345');
--
-- The result will show:
-- - status: success/error
-- - segments: number of merged segments created
-- - speakers_detected: number of unique speakers
-- - original_segments: raw segment count from AI_TRANSCRIBE
-- - merged_segments: count after merging consecutive same-speaker segments
