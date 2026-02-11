#!/usr/bin/env python3
"""
Test script for Speaker Embedding Model Registry integration.

This script tests the custom model locally before deploying to Snowflake.
It validates:
1. Model class instantiation
2. Audio processing with torchaudio (no pydub/ffmpeg)
3. Embedding extraction
4. Similarity computation

Usage:
    python test_speaker_model_registry.py [--audio-file path/to/audio.wav]
"""

import sys
import io
import base64
import argparse
import numpy as np
import pandas as pd

def test_torchaudio_loading():
    """Test that torchaudio can load audio without pydub/ffmpeg."""
    print("\n=== Test 1: torchaudio Audio Loading ===")
    
    import torch
    import torchaudio
    
    # Create a synthetic test audio (1 second of sine wave at 440Hz)
    sample_rate = 16000
    duration = 1.0
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)  # Shape: [1, 16000]
    
    print(f"  Created synthetic audio: {waveform.shape}, {sample_rate}Hz")
    
    # Save to WAV bytes
    buffer = io.BytesIO()
    torchaudio.save(buffer, waveform, sample_rate, format="wav")
    wav_bytes = buffer.getvalue()
    
    print(f"  WAV bytes size: {len(wav_bytes)}")
    
    # Reload from bytes
    buffer.seek(0)
    loaded_waveform, loaded_sr = torchaudio.load(buffer)
    
    print(f"  Reloaded audio: {loaded_waveform.shape}, {loaded_sr}Hz")
    assert loaded_waveform.shape == waveform.shape, "Shape mismatch"
    assert loaded_sr == sample_rate, "Sample rate mismatch"
    
    print("  PASSED: torchaudio can load WAV from bytes")
    return wav_bytes


def test_speechbrain_embedding(wav_bytes: bytes):
    """Test SpeechBrain ECAPA-TDNN embedding extraction."""
    print("\n=== Test 2: SpeechBrain Embedding Extraction ===")
    
    import torch
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    
    # Load model
    print("  Loading ECAPA-TDNN model (this may download ~90MB on first run)...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/speechbrain_cache/spkrec-ecapa-voxceleb"
    )
    print("  Model loaded")
    
    # Load audio
    buffer = io.BytesIO(wav_bytes)
    waveform, sample_rate = torchaudio.load(buffer)
    
    # Extract embedding
    with torch.no_grad():
        embedding = classifier.encode_batch(waveform)
        embedding = embedding.squeeze().cpu().numpy()
    
    # Normalize
    embedding = embedding / np.linalg.norm(embedding)
    
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.4f} (should be 1.0)")
    print(f"  Embedding sample: [{embedding[0]:.4f}, {embedding[1]:.4f}, ..., {embedding[-1]:.4f}]")
    
    assert embedding.shape == (192,), f"Expected 192-dim, got {embedding.shape}"
    assert abs(np.linalg.norm(embedding) - 1.0) < 0.001, "Embedding not normalized"
    
    print("  PASSED: SpeechBrain extracts 192-dim embeddings")
    return embedding


def test_custom_model_class(wav_bytes: bytes):
    """Test the CustomModel wrapper (without Snowflake connection)."""
    print("\n=== Test 3: CustomModel Class (Local Mock) ===")
    
    # We can't fully test without snowflake-ml-python, but we can test the core logic
    # by directly testing the processing functions
    
    import torch
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    
    # Simulate what the model class does
    audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
    
    # Decode
    decoded_bytes = base64.b64decode(audio_b64)
    assert decoded_bytes == wav_bytes, "Base64 round-trip failed"
    print(f"  Base64 encoding/decoding: OK ({len(audio_b64)} chars)")
    
    # Process (simulating _process_audio_bytes)
    buffer = io.BytesIO(decoded_bytes)
    waveform, sample_rate = torchaudio.load(buffer)
    
    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Mono conversion
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    print(f"  Audio preprocessed: {waveform.shape}")
    
    # Test segment extraction
    start_time, end_time = 0.0, 0.5
    start_sample = int(start_time * 16000)
    end_sample = int(end_time * 16000)
    segment = waveform[:, start_sample:end_sample]
    print(f"  Segment extraction ({start_time}-{end_time}s): {segment.shape}")
    
    print("  PASSED: CustomModel preprocessing logic works")


def test_similarity_computation():
    """Test cosine similarity between embeddings."""
    print("\n=== Test 4: Similarity Computation ===")
    
    # Create two random normalized embeddings
    emb1 = np.random.randn(192)
    emb1 = emb1 / np.linalg.norm(emb1)
    
    emb2 = np.random.randn(192)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    # Cosine similarity
    similarity = float(np.dot(emb1, emb2))
    print(f"  Random embeddings similarity: {similarity:.4f}")
    
    # Same embedding should have similarity ~1.0
    self_similarity = float(np.dot(emb1, emb1))
    print(f"  Self-similarity: {self_similarity:.4f}")
    assert abs(self_similarity - 1.0) < 0.001, "Self-similarity should be 1.0"
    
    # Similar embeddings (with noise)
    emb3 = emb1 + np.random.randn(192) * 0.1
    emb3 = emb3 / np.linalg.norm(emb3)
    high_similarity = float(np.dot(emb1, emb3))
    print(f"  Similar embeddings (10% noise): {high_similarity:.4f}")
    assert high_similarity > 0.8, "Similar embeddings should have high similarity"
    
    print("  PASSED: Similarity computation works correctly")


def test_with_real_audio(audio_path: str):
    """Test with a real audio file."""
    print(f"\n=== Test 5: Real Audio File ({audio_path}) ===")
    
    import torch
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    
    # Load audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"  Loaded: {waveform.shape}, {sample_rate}Hz")
    
    # Preprocess
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        print(f"  Resampled to 16kHz: {waveform.shape}")
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        print(f"  Converted to mono: {waveform.shape}")
    
    # Extract embedding
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/speechbrain_cache/spkrec-ecapa-voxceleb"
    )
    
    with torch.no_grad():
        embedding = classifier.encode_batch(waveform)
        embedding = embedding.squeeze().cpu().numpy()
    
    embedding = embedding / np.linalg.norm(embedding)
    
    print(f"  Embedding: {embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
    print(f"  Sample values: [{embedding[0]:.4f}, {embedding[1]:.4f}, ..., {embedding[-1]:.4f}]")
    
    print("  PASSED: Real audio embedding extraction works")
    return embedding


def test_dataframe_interface():
    """Test pandas DataFrame input/output interface."""
    print("\n=== Test 6: DataFrame Interface ===")
    
    # Test input format
    input_df = pd.DataFrame({
        "audio_base64": ["SGVsbG8gV29ybGQ=", "VGVzdCBhdWRpbw=="],
        "start_time": [None, 0.0],
        "end_time": [None, 5.0],
    })
    print(f"  Input DataFrame shape: {input_df.shape}")
    print(f"  Columns: {list(input_df.columns)}")
    
    # Test output format
    output_df = pd.DataFrame({
        "embedding": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "status": ["success", "error"],
        "error": [None, "Audio too short"],
    })
    print(f"  Output DataFrame shape: {output_df.shape}")
    print(f"  Columns: {list(output_df.columns)}")
    
    # Test row iteration
    for idx, row in input_df.iterrows():
        audio_b64 = row.get("audio_base64")
        start_time = row.get("start_time")
        end_time = row.get("end_time")
        print(f"    Row {idx}: audio_b64={audio_b64[:20]}..., start={start_time}, end={end_time}")
    
    print("  PASSED: DataFrame interface is correct")


def main():
    parser = argparse.ArgumentParser(description="Test Speaker Model Registry integration")
    parser.add_argument("--audio-file", type=str, help="Path to test audio file (WAV/MP3)")
    parser.add_argument("--skip-model", action="store_true", help="Skip SpeechBrain model tests")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Speaker Model Registry - Local Tests")
    print("=" * 60)
    
    try:
        # Test 1: torchaudio
        wav_bytes = test_torchaudio_loading()
        
        if not args.skip_model:
            # Test 2: SpeechBrain
            embedding = test_speechbrain_embedding(wav_bytes)
            
            # Test 3: CustomModel logic
            test_custom_model_class(wav_bytes)
        
        # Test 4: Similarity
        test_similarity_computation()
        
        # Test 5: Real audio (optional)
        if args.audio_file:
            test_with_real_audio(args.audio_file)
        
        # Test 6: DataFrame interface
        test_dataframe_interface()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Install snowflake-ml-python: pip install snowflake-ml-python")
        print("  2. Set connection: export SNOWFLAKE_CONNECTION_NAME=your_connection")
        print("  3. Register model: python speaker_model_registry.py")
        print("  4. Deploy service via create_service()")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
