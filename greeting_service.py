"""
Greeting Service using Gemini 2.5 Flash for greeting generation 
and Sarvam AI TTS for voice output
"""

import os
import logging
import time
import requests
import tempfile
from datetime import datetime, timedelta
from typing import Optional, Dict
from pathlib import Path

# Audio playback
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logging.warning("pygame not available - audio playback disabled")

# Gemini API
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("google-generativeai not available - greeting generation disabled")


logger = logging.getLogger(__name__)


def _safe_filename(name: str) -> str:
    allowed = "-.()_"
    return "".join(c if c.isalnum() or c in allowed else "_" for c in name)


class SarvamTTS:
    """
    Sarvam AI Text-to-Speech service
    API Documentation: https://docs.sarvam.ai/api-reference-docs/introduction
    """
    
    def __init__(self, api_key: str):
        # Ensure no stray whitespace in API key
        self.api_key = (api_key or "").strip()
        self.base_url = "https://api.sarvam.ai/text-to-speech"
        self.default_config = {
            "target_language_code": "en-IN",  # English (India)
            # Valid speakers per Sarvam docs (compatible with bulbul:v2)
            "speaker": "vidya",
            "pitch": 0,
            "pace": 1.0,
            "loudness": 1.5,
            "speech_sample_rate": 22050,
            "enable_preprocessing": True,
            # Valid models: 'bulbul:v2' or 'bulbul:v3-beta'
            "model": "bulbul:v2"
        }
        
    def generate_speech(self, text: str, language_code: str = "en-IN") -> Optional[bytes]:
        """
        Generate speech from text using Sarvam AI TTS API
        
        Args:
            text: Text to convert to speech
            language_code: Target language code (default: en-IN for English India)
            
        Returns:
            Audio bytes or None if failed
        """
        try:
            headers = {
                # Send all common header variants to maximize compatibility
                "api-subscription-key": self.api_key,
                "subscription-key": self.api_key,
                "x-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            # Payload format that returns JSON with base64 'audios'
            payload = {
                "inputs": [text],
                "target_language_code": language_code,
                "speaker": self.default_config["speaker"],
                "pitch": self.default_config["pitch"],
                "pace": self.default_config["pace"],
                "loudness": self.default_config["loudness"],
                "speech_sample_rate": self.default_config["speech_sample_rate"],
                "enable_preprocessing": self.default_config["enable_preprocessing"],
                "model": self.default_config["model"]
            }
            
            logger.info(f"🎤 Generating speech with Sarvam AI for: '{text[:50]}...'")
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                try:
                    # Primary: JSON with base64 'audios'
                    result = response.json()
                    if isinstance(result, dict) and "audios" in result and result["audios"]:
                        import base64
                        audio_base64 = result["audios"][0]
                        audio_bytes = base64.b64decode(audio_base64)
                        logger.info(f"✅ Speech generated successfully ({len(audio_bytes)} bytes)")
                        return audio_bytes
                except ValueError:
                    # Fallback: binary audio content
                    ct = response.headers.get("Content-Type", "")
                    if ct.startswith("audio/") or ct == "application/octet-stream":
                        logger.info(f"✅ Speech generated (binary response, {len(response.content)} bytes)")
                        return response.content
                logger.error("No audio data in Sarvam AI response")
                return None
            else:
                logger.error(f"Sarvam AI API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating speech with Sarvam AI: {e}")
            return None
    
    def set_voice_config(self, speaker: str = None, pace: float = None, language: str = None):
        """Update voice configuration"""
        if speaker:
            self.default_config["speaker"] = speaker
        if pace:
            self.default_config["pace"] = pace
        if language:
            self.default_config["target_language_code"] = language
        logger.info(f"Voice config updated: {self.default_config}")


class GeminiGreetingGenerator:
    """
    Gemini 2.5 Flash for personalized greeting generation
    """
    
    def __init__(self, api_key: str):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info("✅ Gemini 2.5 Flash initialized for greeting generation")
    
    def generate_greeting(self, name: str, context: Optional[Dict] = None) -> str:
        """
        Generate a personalized greeting using Gemini 2.5 Flash
        
        Args:
            name: Person's name
            context: Additional context (time of day, previous interactions, etc.)
            
        Returns:
            Generated greeting text
        """
        try:
            # Determine time of day
            hour = datetime.now().hour
            if hour < 12:
                time_greeting = "Good morning"
            elif hour < 17:
                time_greeting = "Good afternoon"
            else:
                time_greeting = "Good evening"
            
            # Build context-aware prompt
            prompt = f"""Generate a warm, natural, and brief greeting for {name}.

Context:
- Time: {time_greeting}
- This is a face recognition system greeting
- Keep it friendly, professional, and concise (1-2 sentences max)
- Make it sound natural and welcoming

Example styles:
- "{time_greeting}, {name}! Great to see you again!"
- "Hello {name}! Welcome back!"
- "Hi {name}, {time_greeting.lower()}! How are you doing?"

Generate ONE natural greeting (just the greeting text, no quotes or extra formatting):"""

            response = self.model.generate_content(prompt)
            greeting = response.text.strip()
            
            # Clean up any quotes or extra formatting
            greeting = greeting.strip('"\'')
            
            logger.info(f"✨ Generated greeting: '{greeting}'")
            return greeting
            
        except Exception as e:
            logger.error(f"Error generating greeting with Gemini: {e}")
            # Fallback to simple greeting
            hour = datetime.now().hour
            if hour < 12:
                return f"Good morning, {name}! Great to see you!"
            elif hour < 17:
                return f"Good afternoon, {name}! Welcome back!"
            else:
                return f"Good evening, {name}! Nice to see you again!"


class AudioPlayer:
    """Simple audio player using pygame"""
    
    def __init__(self):
        if not PYGAME_AVAILABLE:
            logger.warning("pygame not available - audio playback disabled")
            self.enabled = False
            return
        
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self.enabled = True
            logger.info("✅ Audio player initialized")
        except Exception as e:
            logger.error(f"Failed to initialize pygame mixer: {e}")
            self.enabled = False
    
    def play_audio(self, audio_bytes: bytes) -> bool:
        """Play audio from bytes"""
        if not self.enabled:
            logger.warning("Audio playback disabled")
            return False
        
        try:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio_bytes)
            
            # Play audio
            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Cleanup
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            return False
    
    def cleanup(self):
        """Cleanup audio resources"""
        if self.enabled:
            try:
                pygame.mixer.quit()
            except Exception:
                pass


class GreetingManager:
    """
    Manages greeting generation and playback with cooldown logic
    """
    
    def __init__(
        self, 
        gemini_api_key: str,
        sarvam_api_key: str,
        cooldown_minutes: int = 5
    ):
        self.greeting_generator = GeminiGreetingGenerator(gemini_api_key)
        self.tts_service = SarvamTTS(sarvam_api_key)
        self.audio_player = AudioPlayer()
        self.cooldown_minutes = cooldown_minutes
        
        # Track last greeting time for each person
        self.last_greeted: Dict[str, datetime] = {}
        self.greeting_count: Dict[str, int] = {}
        
        logger.info(f"✅ GreetingManager initialized (cooldown: {cooldown_minutes} min)")
    
    def should_greet(self, name: str) -> bool:
        """Check if we should greet this person (based on cooldown)"""
        if name not in self.last_greeted:
            return True
        
        time_since_last = datetime.now() - self.last_greeted[name]
        return time_since_last > timedelta(minutes=self.cooldown_minutes)
    
    def greet_person(self, name: str, confidence: float = 1.0) -> bool:
        """
        Generate and play a personalized greeting
        
        Args:
            name: Person's name
            confidence: Recognition confidence score
            
        Returns:
            True if greeting was successful
        """
        try:
            # Generate greeting with Gemini
            greeting_text = self.greeting_generator.generate_greeting(name)
            
            # Generate speech with Sarvam AI
            audio_bytes = self.tts_service.generate_speech(greeting_text)
            
            if audio_bytes:
                # Persist audio to outputs directory
                try:
                    outputs_dir = Path("outputs")
                    outputs_dir.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    clean_name = _safe_filename(name or "person")
                    filename = f"greeting_{timestamp}_{clean_name}.wav"
                    file_path = outputs_dir / filename
                    with open(file_path, "wb") as f:
                        f.write(audio_bytes)
                    logger.info(f"💾 Saved greeting audio to {file_path}")
                except Exception as save_err:
                    logger.error(f"Failed to save greeting audio: {save_err}")

                # Play audio
                success = self.audio_player.play_audio(audio_bytes)
                
                if success:
                    # Update tracking
                    self.last_greeted[name] = datetime.now()
                    self.greeting_count[name] = self.greeting_count.get(name, 0) + 1
                    
                    logger.info(f"🎉 Successfully greeted {name} (confidence: {confidence:.2f})")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error greeting {name}: {e}")
            return False
    
    def greet_if_needed(self, name: str, confidence: float = 1.0) -> bool:
        """Greet person only if cooldown has expired"""
        if self.should_greet(name):
            return self.greet_person(name, confidence)
        return False
    
    def reset_cooldown(self, name: Optional[str] = None):
        """Reset cooldown for a person or all people"""
        if name:
            if name in self.last_greeted:
                del self.last_greeted[name]
                logger.info(f"Cooldown reset for {name}")
        else:
            self.last_greeted.clear()
            logger.info("Cooldown reset for all people")
    
    def get_stats(self) -> Dict:
        """Get greeting statistics"""
        return {
            "total_people_greeted": len(self.greeting_count),
            "greeting_counts": self.greeting_count,
            "last_greeted": {
                name: dt.isoformat() 
                for name, dt in self.last_greeted.items()
            },
            "cooldown_minutes": self.cooldown_minutes
        }
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.audio_player.cleanup()
        except Exception:
            pass


# Backward compatibility - expose classes for camera_app.py
IndicParlerTTS = GreetingManager  # For compatibility with existing camera_app.py


def test_greeting_system():
    """Test function for the greeting system"""
    from dotenv import load_dotenv
    load_dotenv()
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    sarvam_key = os.getenv("SARVAM_API_KEY")
    
    if not gemini_key or not sarvam_key:
        print("❌ Error: GEMINI_API_KEY and SARVAM_API_KEY must be set in .env file")
        return
    
    print("🚀 Testing Greeting System...\n")
    
    # Initialize
    manager = GreetingManager(
        gemini_api_key=gemini_key,
        sarvam_api_key=sarvam_key,
        cooldown_minutes=1  # Short cooldown for testing
    )
    
    # Test greeting
    test_name = "John"
    print(f"Testing greeting for '{test_name}'...")
    success = manager.greet_person(test_name)
    
    if success:
        print(f"✅ Successfully greeted {test_name}!")
    else:
        print(f"❌ Failed to greet {test_name}")
    
    # Show stats
    stats = manager.get_stats()
    print(f"\n📊 Stats: {stats}")
    
    # Cleanup
    manager.cleanup()
    print("\n✅ Test complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    test_greeting_system()

