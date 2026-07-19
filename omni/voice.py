"""
Voice — Kokoro ONNX TTS + Whisper STT for Unison's audio pipeline.

Uses kokoro-onnx for fastest TTS inference and Whisper (via whisper.cpp
or transformers) for speech-to-text transcription.

Kokoro voice: bm_fable (British Male Fable) — Unison's assigned voice.

Audio is encoded via the AudioEncoder from modalities.py into character
sequences that are banked into the SynapticGraph alongside text.
"""
import os
import io
import time
import wave
import struct
import numpy as np
from omni.logging_config import get_logger

voice_logger = get_logger("OmniVoice", "voice.log")

# ── Model paths ──
_SSD_BASE = "/Volumes/One Touch/models library"
_KOKORO_MODEL = f"{_SSD_BASE}/Audio_TTS/Kokoro-82M"
_KOKORO_ONNX_SRC = "/Volumes/One Touch/ernos-archive/programs/kokoro-onnx"
_WHISPER_MODEL = f"{_SSD_BASE}/Audio_TTS/Whisper-Large-v3-Turbo"
_VOICE_FILE = f"{_KOKORO_MODEL}/voices/bm_fable.pt"

# Output directory for generated audio
_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "generated_audio")


class KokoroSpeaker:
    """
    Text-to-speech via Kokoro-82M ONNX runtime.
    
    Uses bm_fable voice (British Male Fable) as Unison's canonical voice.
    Returns WAV bytes that can be sent as Discord voice messages or
    encoded into audio orbits.
    """
    def __init__(self, voice="bm_fable", speed=1.0):
        self.voice_name = voice
        self.speed = speed
        self.kokoro = None
        self.available = False

        os.makedirs(_AUDIO_DIR, exist_ok=True)
        # Removal-proof replay record (paper Sec 8.6): a sentence SPOKEN ONCE is
        # thereafter re-spoken from the engine's own held record with NO synthesis
        # model. Kokoro is the teacher (scaffolding); none is load-bearing.
        self._record_dir = os.path.join(_AUDIO_DIR, "spoken_record")
        os.makedirs(self._record_dir, exist_ok=True)
        self._init_kokoro()

    def _record_path(self, text, voice):
        import hashlib
        key = hashlib.sha256(f"{voice}\x00{text}".encode("utf-8")).hexdigest()[:32]
        return os.path.join(self._record_dir, f"{key}.wav")

    @staticmethod
    def _wav_duration_ms(wav_bytes):
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                return wf.getnframes() / wf.getframerate() * 1000.0
        except Exception:
            return 0.0
    
    def _init_kokoro(self):
        """Initialise the Kokoro ONNX runtime."""
        try:
            import kokoro_onnx
            
            # Find the ONNX model file
            onnx_model = os.path.join(_KOKORO_MODEL, "kokoro-v1_0.onnx")
            voices_dir = os.path.join(os.path.expanduser("~"), ".ernosagent", "models", "voices-v1.0.bin")
            
            # If ONNX model doesn't exist, check the kokoro-onnx source
            if not os.path.exists(onnx_model):
                # Look in the kokoro-onnx project
                alt_paths = [
                    os.path.join(_KOKORO_ONNX_SRC, "kokoro-v1_0.onnx"),
                    os.path.join(os.path.expanduser("~"), ".cache", "kokoro-onnx", "kokoro-v1_0.onnx"),
                    os.path.join(os.path.expanduser("~"), ".ernosagent", "models", "kokoro-v1.0.onnx"),
                ]
                for alt in alt_paths:
                    if os.path.exists(alt):
                        onnx_model = alt
                        break
            
            if not os.path.exists(onnx_model):
                voice_logger.warning(
                    f"Kokoro ONNX model not found. Checked: {onnx_model}. "
                    f"TTS will use fallback. Install kokoro-onnx and download model."
                )
                self.available = False
                return
            
            self.kokoro = kokoro_onnx.Kokoro(onnx_model, voices_dir)
            self.available = True
            voice_logger.info(
                f"Kokoro TTS initialised: model={onnx_model}, "
                f"voice={self.voice_name}, speed={self.speed}"
            )
            
        except ImportError:
            voice_logger.warning(
                "kokoro_onnx not installed. Install with: pip install kokoro-onnx"
            )
            self.available = False
        except Exception as e:
            voice_logger.error(f"Kokoro initialisation failed: {e}", exc_info=True)
            self.available = False
    
    def speak(self, text, voice=None):
        """
        Speak text. Removal-proof (paper Sec 8.6): if this sentence has been
        spoken before, it is replayed from the HELD RECORD with no synthesis model
        touched -- so it still speaks even when Kokoro is unavailable. Only a
        never-before-spoken sentence goes to the teacher (Kokoro).

        Returns (success: bool, wav_bytes or error: str, duration_ms: float).
        """
        voice = voice or self.voice_name

        # Record-first: re-speak from the engine's own record, no synthesis.
        rec = self._record_path(text, voice)
        if os.path.exists(rec):
            try:
                with open(rec, "rb") as f:
                    wav_bytes = f.read()
                dur = self._wav_duration_ms(wav_bytes)
                voice_logger.info(f"TTS from RECORD (no synthesis): \"{text[:50]}\" | {len(wav_bytes)} bytes")
                return True, wav_bytes, dur
            except Exception as e:
                voice_logger.warning(f"record replay failed ({e}); re-synthesising.")

        if not self.available or self.kokoro is None:
            return False, "TTS not available", 0
        
        try:
            t0 = time.perf_counter()
            
            # Generate audio
            samples, sample_rate = self.kokoro.create(
                text, voice=voice, speed=self.speed
            )
            
            elapsed_ms = (time.perf_counter() - t0) * 1000
            duration_ms = len(samples) / sample_rate * 1000
            
            # Convert numpy array to WAV bytes
            wav_bytes = self._samples_to_wav(samples, sample_rate)

            # Hold the record: spoken once, thereafter removal-proof (replayed).
            try:
                with open(rec, "wb") as f:
                    f.write(wav_bytes)
            except Exception as e:
                voice_logger.warning(f"could not hold spoken record: {e}")

            voice_logger.info(
                f"TTS generated (held to record): \"{text[:50]}\" | "
                f"voice={voice} | {duration_ms:.0f}ms audio | "
                f"{elapsed_ms:.0f}ms compute | {len(wav_bytes)} bytes"
            )

            return True, wav_bytes, duration_ms
            
        except Exception as e:
            voice_logger.error(f"TTS failed: {e}", exc_info=True)
            return False, f"TTS error: {e}", 0
    
    def speak_to_file(self, text, filename=None, voice=None):
        """Generate speech and save to a WAV file. Returns the file path."""
        success, result, duration = self.speak(text, voice)
        if not success:
            return None
        
        if filename is None:
            filename = f"tts_{int(time.time() * 1000)}.wav"
        
        filepath = os.path.join(_AUDIO_DIR, filename)
        with open(filepath, "wb") as f:
            f.write(result)
        
        return filepath
    
    def _samples_to_wav(self, samples, sample_rate):
        """Convert numpy float32 samples to WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            # Convert float32 [-1, 1] to int16
            int_samples = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
            wf.writeframes(int_samples.tobytes())
        return buf.getvalue()


class WhisperListener:
    """
    Speech-to-text via Whisper Large v3 Turbo.
    
    Transcribes audio input (voice messages, audio clips) into text
    that can be processed by the Omni Engine.
    """
    def __init__(self):
        self.model = None
        self.processor = None
        # Whisper is an optional input organ. Loading its multi-gigabyte local
        # weights here blocked Discord text service before connection even
        # when no audio was requested. Advertise the locally present organ and
        # materialise it on the first actual listen instead.
        self.available = os.path.exists(_WHISPER_MODEL)
    
    def _init_whisper(self):
        """Initialise Whisper model from local weights."""
        self.available = False
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            import torch
            
            if not os.path.exists(_WHISPER_MODEL):
                voice_logger.warning(f"Whisper model not found at {_WHISPER_MODEL}")
                return
            
            voice_logger.info(f"Loading Whisper from {_WHISPER_MODEL}...")
            t0 = time.perf_counter()
            
            self.processor = WhisperProcessor.from_pretrained(_WHISPER_MODEL)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                _WHISPER_MODEL,
                dtype=torch.float16,
            )
            # Use MPS (Metal) on Apple Silicon
            if torch.backends.mps.is_available():
                self.model = self.model.to("mps")
                voice_logger.info("Whisper loaded on MPS (Metal).")
            
            self.available = True
            elapsed = (time.perf_counter() - t0) * 1000
            voice_logger.info(f"Whisper initialised in {elapsed:.0f}ms")
            
        except ImportError:
            voice_logger.warning(
                "transformers/torch not installed for Whisper. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            voice_logger.error(f"Whisper initialisation failed: {e}", exc_info=True)
    
    def listen(self, audio_bytes, sample_rate=16000):
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_bytes: Raw WAV or PCM audio bytes.
            sample_rate: Sample rate of the audio.
            
        Returns (success: bool, text or error: str).
        """
        if self.model is None and self.available:
            self._init_whisper()
        if not self.available or self.model is None or self.processor is None:
            return False, "Whisper not available"
        
        try:
            import torch
            
            t0 = time.perf_counter()
            
            # Parse WAV if it has a header
            audio_array = self._parse_audio(audio_bytes, sample_rate)
            
            input_features = self.processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features
            
            if torch.backends.mps.is_available():
                input_features = input_features.to("mps").half()
            
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)
            
            text = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            
            elapsed_ms = (time.perf_counter() - t0) * 1000
            voice_logger.info(
                f"Transcribed: \"{text[:80]}\" | {elapsed_ms:.0f}ms"
            )
            
            return True, text.strip()
            
        except Exception as e:
            voice_logger.error(f"Transcription failed: {e}", exc_info=True)
            return False, f"Transcription error: {e}"
    
    def _parse_audio(self, audio_bytes, default_rate=16000):
        """Parse audio bytes (WAV or raw PCM) to a float32 numpy array."""
        try:
            # Try WAV first
            buf = io.BytesIO(audio_bytes)
            with wave.open(buf, "rb") as wf:
                n_frames = wf.getnframes()
                sample_rate = wf.getframerate()
                raw = wf.readframes(n_frames)
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Resample to 16000 if needed
                if sample_rate != 16000:
                    ratio = 16000 / sample_rate
                    new_len = int(len(samples) * ratio)
                    samples = np.interp(
                        np.linspace(0, len(samples), new_len),
                        np.arange(len(samples)),
                        samples
                    )
                return samples
        except Exception:
            # Assume raw 16-bit PCM
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            return samples
