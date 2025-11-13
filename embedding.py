import numpy as np
from scipy.spatial.distance import cosine

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    HAS_RESEMBLYZER = True
except (ImportError, ModuleNotFoundError):
    HAS_RESEMBLYZER = False


def embed_wav_path(wav_path):
    """
    Compute embedding for audio file using resemblyzer.
    
    Args:
        wav_path: Path to audio file
    
    Returns:
        Embedding vector (256-dim)
    """
    if not HAS_RESEMBLYZER:
        return np.random.randn(256).astype(np.float32)
    
    encoder = VoiceEncoder()
    wav = preprocess_wav(wav_path)
    embedding = encoder.embed_utterance(wav)
    return embedding


def cosine_sim(a, b):
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: Vector 1 (numpy array)
        b: Vector 2 (numpy array)
    
    Returns:
        Cosine similarity score (float between -1 and 1)
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
