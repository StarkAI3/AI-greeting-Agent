"""
Indic Parler TTS Service for Indian Accent Voice Greetings
Handles text-to-speech conversion using Indic Parler TTS API
"""
import os
import time
import logging
import requests
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import threading
import queue

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("pygame not available, audio playback will be disabled")

logger = logging.getLogger(__name__)

class IndicParlerTTS:
    """
    Indic Parler TTS API Integration for Indian accent voice synthesis
    """
    
    def __init__(self, api_url: str = None, api_key: str = None):
        """
        Initialize Indic Parler TTS service
        
        Args:
            api_url: API endpoint URL (default from env)
            api_key: API key for authentication (default from env)
        """
        # Get API configuration from environment or use defaults
        # Prefer an officially hosted Parler TTS model unless overridden via env
        # Note: The previous default (ai4bharat/indic-parler-tts) returned 404.
        self.api_url = api_url or os.getenv(
            "INDIC_PARLER_API_URL",
            "https://api-inference.huggingface.co/models/parler-tts/parler-tts-mini-v1"
        )
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY", "")
        self.force_fallback = str(os.getenv("TTS_FORCE_FALLBACK", "false")).lower() in ("1","true","yes","on")
        # Azure Neural TTS (optional, premium high-quality)
        self.azure_key = os.getenv("AZURE_SPEECH_KEY", "")
        self.azure_region = os.getenv("AZURE_SPEECH_REGION", "")
        self.azure_voice = os.getenv("AZURE_SPEECH_VOICE", "en-IN-NeerjaNeural")
        
        # TTS configuration for Indian accent
        self.voice_config = {
            "voice": "indian_female",  # Options: indian_female, indian_male
            "language": "en-IN",  # Indian English
            "speed": 1.0,
            "pitch": 1.0,
        }
        
        # Audio cache directory
        self.cache_dir = Path("audio_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize audio player
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        self.is_playing = False
        
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                logger.info("✅ Audio player initialized successfully")
                
                # Start audio playback thread
                self.audio_thread = threading.Thread(target=self._audio_playback_worker, daemon=True)
                self.audio_thread.start()
            except Exception as e:
                logger.error(f"Failed to initialize pygame mixer: {e}")
        else:
            logger.warning("pygame not available, audio playback disabled")
        
        logger.info(f"Indic Parler TTS service initialized")
        logger.info(f"API URL: {self.api_url}")
        logger.info(f"TTS force fallback: {self.force_fallback}")
        logger.info(f"Voice: {self.voice_config['voice']}, Language: {self.voice_config['language']}")
    
    def _build_speaker_prompt(self) -> str:
        """
        Build a Parler-TTS speaker prompt emphasizing Indian English accent and clear Indian names.
        """
        base_gender = "female" if self.voice_config.get("voice") == "indian_female" else "male"
        # Parler prompt describing an Indian English speaker
        return (
            f"A {base_gender} speaker with Indian English accent, clear diction, warm and friendly tone. "
            f"Pronounces Indian names accurately. Speaking rate {self.voice_config.get('speed', 1.0)}x."
        )

    def _audio_playback_worker(self):
        """Background worker for audio playback"""
        while True:
            try:
                audio_file = self.audio_queue.get(timeout=1)
                if audio_file is None:  # Sentinel value to stop
                    break
                
                self.is_playing = True
                self._play_audio_sync(audio_file)
                self.is_playing = False
                
                # Small delay between greetings
                time.sleep(0.5)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio playback worker: {e}")
                self.is_playing = False
    
    def _play_audio_sync(self, audio_file: str):
        """
        Play audio file synchronously using pygame
        
        Args:
            audio_file: Path to audio file
        """
        try:
            if not PYGAME_AVAILABLE:
                logger.warning("Cannot play audio: pygame not available")
                return
            
            # Convert mp3 to wav at runtime if needed for pygame
            if audio_file.endswith('.mp3'):
                try:
                    import subprocess, os
                    wav_path = str(Path(audio_file).with_suffix('.wav'))
                    # Use ffmpeg if available to convert
                    subprocess.run(
                        ["ffmpeg", "-y", "-loglevel", "error", "-i", audio_file, wav_path],
                        check=True
                    )
                    audio_file = wav_path
                except Exception:
                    # Fallback: let pygame try mp3 directly
                    pass

            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            logger.info(f"✅ Finished playing audio: {audio_file}")
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
    
    def synthesize_speech(self, text: str, cache_key: str = None) -> Optional[str]:
        """
        Convert text to speech using Indic Parler TTS API
        
        Args:
            text: Text to convert to speech
            cache_key: Optional cache key for reusing audio
            
        Returns:
            Path to generated audio file or None if failed
        """
        try:
            # Check cache first
            if cache_key:
                cached_file = self.cache_dir / f"{cache_key}.wav"
                if cached_file.exists():
                    logger.info(f"Using cached audio: {cached_file}")
                    return str(cached_file)
            
            logger.info(f"Synthesizing speech: '{text}'")

            # If forced, skip API and use fallback immediately
            if self.force_fallback or not self.api_url:
                logger.info("Skipping API synthesis, using fallback by configuration")
                # Try Azure first if configured for best quality, else gTTS/pyttsx3
                if self.azure_key and self.azure_region:
                    audio = self._azure_tts(text, cache_key)
                    if audio:
                        return audio
                return self._fallback_tts(text, cache_key)
            
            # Prepare API request
            headers = {"Accept": "audio/wav"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            # Parler-TTS expects a JSON with nested inputs: {"inputs": {"text": ..., "speaker": ...}}
            payload = {
                "inputs": {
                    "text": text,
                    "speaker": self._build_speaker_prompt(),
                }
            }
            
            # Make API request with retry logic
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        json=payload,
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        # Ensure we received audio bytes
                        content_type = response.headers.get("content-type", "")
                        if "audio" in content_type:
                            audio_file = self._save_audio(response.content, cache_key)
                            logger.info(f"✅ Speech synthesized successfully: {audio_file}")
                            return audio_file
                        else:
                            logger.error(f"TTS API returned non-audio response: {response.text[:200]}")
                            break
                    elif response.status_code == 503:
                        # Model loading, retry
                        logger.warning(f"Model loading, retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        logger.error(f"TTS API error: {response.status_code} - {response.text}")
                        break
                except requests.exceptions.Timeout:
                    logger.warning(f"Request timeout, attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
            
            # Fallbacks: Azure → gTTS/pyttsx3
            logger.warning("API synthesis failed, using fallback TTS")
            if self.azure_key and self.azure_region:
                audio = self._azure_tts(text, cache_key)
                if audio:
                    return audio
            return self._fallback_tts(text, cache_key)
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            return self._fallback_tts(text, cache_key)
    
    def _save_audio(self, audio_content: bytes, cache_key: str = None) -> str:
        """
        Save audio content to file
        
        Args:
            audio_content: Audio data in bytes
            cache_key: Optional cache key
            
        Returns:
            Path to saved audio file
        """
        try:
            if cache_key:
                audio_file = self.cache_dir / f"{cache_key}.wav"
            else:
                # Create temporary file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_file = self.cache_dir / f"tts_{timestamp}.wav"
            
            with open(audio_file, "wb") as f:
                f.write(audio_content)
            
            return str(audio_file)
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
            return None
    
    def _fallback_tts(self, text: str, cache_key: str = None) -> Optional[str]:
        """
        Fallback TTS using gTTS (online, Indian accent) then pyttsx3 (offline)
        
        Args:
            text: Text to convert
            cache_key: Optional cache key
            
        Returns:
            Path to audio file or None
        """
        # 1) Try gTTS (Google) with Indian accent via tld='co.in'
        try:
            from gtts import gTTS
            logger.info("Using fallback TTS (gTTS, Indian accent)")
            if cache_key:
                audio_file = self.cache_dir / f"{cache_key}_fallback.mp3"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                audio_file = self.cache_dir / f"tts_fallback_{timestamp}.mp3"
            tts = gTTS(text=text, lang='en', tld='co.in', slow=False)
            tts.save(str(audio_file))
            return str(audio_file)
        except Exception as e:
            logger.error(f"gTTS fallback error: {e}")
            # 2) Try pyttsx3 offline
            try:
                import pyttsx3
                logger.info("Using fallback TTS (pyttsx3)")
                if cache_key:
                    audio_file = self.cache_dir / f"{cache_key}_fallback.wav"
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    audio_file = self.cache_dir / f"tts_fallback_{timestamp}.wav"
                engine = pyttsx3.init()
                voices = engine.getProperty('voices')
                for voice in voices:
                    if 'india' in voice.name.lower() or 'english' in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 0.9)
                engine.save_to_file(text, str(audio_file))
                engine.runAndWait()
                return str(audio_file)
            except Exception as e2:
                logger.error(f"pyttsx3 fallback error: {e2}")
                return None

    def _azure_tts(self, text: str, cache_key: str = None) -> Optional[str]:
        """
        Azure Neural TTS for high-quality natural voice (e.g., en-IN-NeerjaNeural)
        """
        try:
            import azure.cognitiveservices.speech as speechsdk
            if not (self.azure_key and self.azure_region):
                return None
            if cache_key:
                out_path = self.cache_dir / f"{cache_key}_azure.wav"
            else:
                out_path = self.cache_dir / f"tts_azure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

            speech_config = speechsdk.SpeechConfig(subscription=self.azure_key, region=self.azure_region)
            speech_config.speech_synthesis_voice_name = self.azure_voice
            # Slightly slower for receptionist warmth
            speech_config.set_speech_synthesis_output_format(
                speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
            )
            audio_config = speechsdk.audio.AudioOutputConfig(filename=str(out_path))
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

            # SSML for richer control could be added here if needed
            result = synthesizer.speak_text_async(text).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info(f"✅ Azure TTS synthesized: {out_path}")
                return str(out_path)
            else:
                logger.error(f"Azure TTS failed: {result.reason}")
                return None
        except Exception as e:
            logger.error(f"Azure TTS error: {e}")
            return None
    
    def greet_person(self, name: str, time_of_day: str = None) -> bool:
        """
        Generate and play greeting for a person
        
        Args:
            name: Person's name to greet
            time_of_day: Optional time of day (morning, afternoon, evening)
            
        Returns:
            True if greeting was queued successfully
        """
        try:
            # Generate greeting text
            greeting_text = self._generate_greeting(name, time_of_day)
            
            # Create cache key from name
            cache_key = f"greeting_{name.lower().replace(' ', '_')}"
            
            # Synthesize speech
            audio_file = self.synthesize_speech(greeting_text, cache_key)
            
            if audio_file:
                # Queue for playback
                self.audio_queue.put(audio_file)
                logger.info(f"✅ Greeting queued for {name}: '{greeting_text}'")
                return True
            else:
                logger.error(f"Failed to generate greeting for {name}")
                return False
        except Exception as e:
            logger.error(f"Error greeting person: {e}")
            return False
    
    def _generate_greeting(self, name: str, time_of_day: str = None) -> str:
        """
        Generate personalized greeting message
        
        Args:
            name: Person's name
            time_of_day: Time of day (morning, afternoon, evening)
            
        Returns:
            Greeting text
        """
        # Determine time of day if not provided
        if not time_of_day:
            hour = datetime.now().hour
            if hour < 12:
                time_of_day = "morning"
            elif hour < 17:
                time_of_day = "afternoon"
            else:
                time_of_day = "evening"
        
        # Greeting templates with Indian touch
        greetings = {
            "morning": [
                f"Namaste {name}, Good morning!",
                f"Good morning {name}, have a wonderful day!",
                f"Shuprabhaat {name}, welcome!",
            ],
            "afternoon": [
                f"Good afternoon {name}, welcome!",
                f"Namaste {name}, hope you're having a great day!",
                f"Hello {name}, good to see you!",
            ],
            "evening": [
                f"Good evening {name}, welcome!",
                f"Namaste {name}, how was your day?",
                f"Shubraatri {name}, good evening!",
            ]
        }
        
        # Select greeting
        import random
        greeting_list = greetings.get(time_of_day, greetings["morning"])
        return random.choice(greeting_list)
    
    def set_voice_config(self, voice: str = None, speed: float = None):
        """
        Update voice configuration
        
        Args:
            voice: Voice type (indian_female, indian_male)
            speed: Speech speed (0.5 to 2.0)
        """
        if voice:
            self.voice_config["voice"] = voice
            logger.info(f"Voice changed to: {voice}")
        
        if speed:
            self.voice_config["speed"] = max(0.5, min(2.0, speed))
            logger.info(f"Speech speed changed to: {self.voice_config['speed']}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Stop audio thread
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_queue.put(None)  # Sentinel to stop thread
                self.audio_thread.join(timeout=2)
            
            # Cleanup pygame
            if PYGAME_AVAILABLE:
                pygame.mixer.quit()
            
            logger.info("TTS service cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class GreetingManager:
    """
    Manages greeting logic with cooldown to avoid repetitive greetings
    """
    
    def __init__(self, tts_service: IndicParlerTTS, cooldown_minutes: int = 5):
        """
        Initialize greeting manager
        
        Args:
            tts_service: TTS service instance
            cooldown_minutes: Cooldown period between greetings for same person
        """
        self.tts_service = tts_service
        self.cooldown_minutes = cooldown_minutes
        self.last_greeted: Dict[str, datetime] = {}
        self.greeting_lock = threading.Lock()
        
        logger.info(f"Greeting Manager initialized with {cooldown_minutes} min cooldown")
    
    def should_greet(self, name: str) -> bool:
        """
        Check if person should be greeted based on cooldown
        
        Args:
            name: Person's name
            
        Returns:
            True if person should be greeted
        """
        if name == "Unknown":
            return False
        
        with self.greeting_lock:
            now = datetime.now()
            
            if name not in self.last_greeted:
                return True
            
            last_time = self.last_greeted[name]
            time_diff = now - last_time
            
            return time_diff >= timedelta(minutes=self.cooldown_minutes)
    
    def greet_if_needed(self, name: str, confidence: float = 0.0) -> bool:
        """
        Greet person if needed based on cooldown
        
        Args:
            name: Person's name
            confidence: Recognition confidence (0.0 to 1.0)
            
        Returns:
            True if greeting was triggered
        """
        try:
            # Only greet if confidence is high enough
            if confidence < 0.7:
                return False
            
            if self.should_greet(name):
                # Trigger greeting
                success = self.tts_service.greet_person(name)
                
                if success:
                    with self.greeting_lock:
                        self.last_greeted[name] = datetime.now()
                    logger.info(f"✅ Greeted: {name} (confidence: {confidence:.2f})")
                    return True
            else:
                logger.debug(f"Skipping greeting for {name} (cooldown active)")
            
            return False
        except Exception as e:
            logger.error(f"Error in greet_if_needed: {e}")
            return False
    
    def reset_cooldown(self, name: str = None):
        """
        Reset cooldown for a person or all people
        
        Args:
            name: Person's name (None for all)
        """
        with self.greeting_lock:
            if name:
                if name in self.last_greeted:
                    del self.last_greeted[name]
                    logger.info(f"Reset cooldown for: {name}")
            else:
                self.last_greeted.clear()
                logger.info("Reset all greeting cooldowns")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get greeting statistics
        
        Returns:
            Dictionary with greeting stats
        """
        with self.greeting_lock:
            return {
                "total_people_greeted": len(self.last_greeted),
                "last_greeted": {
                    name: time.isoformat() 
                    for name, time in self.last_greeted.items()
                },
                "cooldown_minutes": self.cooldown_minutes
            }

