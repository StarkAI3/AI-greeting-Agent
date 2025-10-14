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

# Indic transliteration for Marathi names using aksharamukha
try:
    from aksharamukha import transliterate as aksharamukha_transliterate
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    TRANSLITERATION_AVAILABLE = False
    logging.warning("aksharamukha not available - Marathi name conversion disabled")

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


def convert_name_to_marathi(name: str, person_details: Optional[Dict] = None) -> str:
    """
    Convert a name from English/Latin script to Marathi Devanagari script
    for better pronunciation by TTS systems.
    
    Priority:
    1. Use 'marathi_name' from person_details if provided
    2. Use common name mappings
    3. Use phonetic transliteration as fallback
    
    Args:
        name: Name in English/Latin script (e.g., "Shubham Kamble", "Prabhuraj Dhondge")
        person_details: Optional person metadata that may contain 'marathi_name' field
        
    Returns:
        Name in Marathi Devanagari script (e.g., "शुभम कांबळे", "प्रभुराज धोंडगे")
    """
    # Priority 1: Use user-provided Marathi name from metadata
    if person_details and person_details.get('marathi_name'):
        marathi_name = person_details['marathi_name']
        logger.info(f"📝 Using metadata Marathi name: '{name}' → '{marathi_name}'")
        return marathi_name
    
    # Priority 2: Common Marathi name mappings (add more as needed)
    common_names = {
        # First names
        'Shubham': 'शुभम', 'Prabhuraj': 'प्रभुराज', 'Soham': 'सोहम',
        'Priya': 'प्रिया', 'Raj': 'राज', 'Amit': 'अमित',
        'Sneha': 'स्नेहा', 'Rahul': 'राहुल', 'Ananya': 'अनन्या',
        'Arjun': 'अर्जुन', 'Aarav': 'आरव', 'Vivaan': 'विवान',
        'Aditya': 'आदित्य', 'Vihaan': 'विहान', 'Reyansh': 'रेयांश',
        'Sai': 'साई', 'Krishna': 'कृष्ण', 'Rudra': 'रुद्र',
        'Ishaan': 'ईशान', 'Atharva': 'अथर्व', 'Arnav': 'अर्णव',
        'Aadhya': 'आध्या', 'Anvi': 'अन्वी', 'Diya': 'दिया',
        'Pari': 'परी', 'Saanvi': 'सान्वी', 'Sara': 'सारा',
        
        # Last names (Marathi surnames)
        'Kamble': 'कांबळे', 'Dhondge': 'धोंडगे', 'Pawar': 'पवार',
        'Sharma': 'शर्मा', 'Patel': 'पाटील', 'Kumar': 'कुमार',
        'Desai': 'देसाई', 'Patil': 'पाटील', 'Singh': 'सिंह',
        'Kulkarni': 'कुलकर्णी', 'Joshi': 'जोशी', 'More': 'मोरे',
        'Jadhav': 'जाधव', 'Bhosale': 'भोसले', 'Gaikwad': 'गायकवाड',
        'Shinde': 'शिंदे', 'Sawant': 'सावंत', 'Chavan': 'चव्हाण',
        'Raut': 'राऊत', 'Ghag': 'घाग', 'Naik': 'नायक',
    }
    
    # Try to convert each part of the name
    parts = name.split()
    converted_parts = []
    
    for part in parts:
        # Check if we have a mapping for this part
        if part in common_names:
            converted_parts.append(common_names[part])
        else:
            # Try case-insensitive match
            part_lower = {k.lower(): v for k, v in common_names.items()}.get(part.lower())
            if part_lower:
                converted_parts.append(part_lower)
            else:
                # No mapping found, use original
                converted_parts.append(part)
    
    marathi_name = ' '.join(converted_parts)
    
    # If we successfully converted any part, use it
    if marathi_name != name:
        logger.info(f"📝 Converted name using mappings: '{name}' → '{marathi_name}'")
        return marathi_name
    
    # Fallback: return original name if no conversion available
    logger.info(f"📝 No Marathi mapping found for '{name}', using original")
    return name


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
            # ACTUAL valid speakers for bulbul:v2: anushka, abhilash, manisha, vidya, arya, karun, hitesh
            "speaker": "anushka",
            "pitch": 0,
            "pace": 0.95,  # Slightly slower for clearer pronunciation
            "loudness": 1.2,  # Reduced loudness to prevent distortion
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
            
            # Build payload; omit unsupported fields for Bulbul v3
            model_name = self.default_config.get("model", "bulbul:v2")
            payload = {
                "inputs": [text],
                "target_language_code": language_code,
                "speaker": self.default_config.get("speaker", "vidya"),
                "pace": self.default_config.get("pace", 1.0),
                "speech_sample_rate": self.default_config.get("speech_sample_rate", 22050),
                "enable_preprocessing": self.default_config.get("enable_preprocessing", True),
                "model": model_name,
            }
            # pitch and loudness not supported in bulbul v3
            if not model_name.startswith("bulbul:v3"):
                payload["pitch"] = self.default_config.get("pitch", 0)
                payload["loudness"] = self.default_config.get("loudness", 1.5)
            
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
    
    def set_voice_config(
        self,
        speaker: str = None,
        pace: float = None,
        language: str = None,
        pitch: float = None,
        loudness: float = None,
        speech_sample_rate: int = None,
    ):
        """Update voice configuration"""
        if speaker:
            self.default_config["speaker"] = speaker
        if pace:
            self.default_config["pace"] = pace
        if pitch is not None:
            self.default_config["pitch"] = pitch
        if loudness is not None:
            self.default_config["loudness"] = loudness
        if speech_sample_rate is not None:
            self.default_config["speech_sample_rate"] = speech_sample_rate
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
    
    def generate_greeting(self, name: str, person_details: Optional[Dict] = None) -> str:
        """
        Generate a personalized, fun greeting using Gemini 2.5 Flash with all person details
        
        Args:
            name: Person's name (will be converted to Marathi Devanagari for better TTS pronunciation)
            person_details: Full enrollment details (position, department, interests, etc.)
            
        Returns:
            Generated greeting text
        """
        try:
            # Convert name to Marathi Devanagari for better pronunciation
            marathi_name = convert_name_to_marathi(name, person_details)
            logger.info(f"🗣️ Using Marathi name for TTS: {marathi_name}")
            
            # Determine time of day
            hour = datetime.now().hour
            if hour < 12:
                time_greeting = "Good morning"
            elif hour < 17:
                time_greeting = "Good afternoon"
            else:
                time_greeting = "Good evening"
            
            # Build comprehensive context from person details
            context_info = []
            if person_details:
                if person_details.get('position'):
                    context_info.append(f"Position: {person_details['position']}")
                if person_details.get('department'):
                    context_info.append(f"Department: {person_details['department']}")
                if person_details.get('current_project'):
                    context_info.append(f"Current Project: {person_details['current_project']}")
                if person_details.get('interests'):
                    interests = person_details['interests']
                    if isinstance(interests, list):
                        interests = ', '.join(interests)
                    context_info.append(f"Interests: {interests}")
                if person_details.get('skills'):
                    skills = person_details['skills']
                    if isinstance(skills, list):
                        skills = ', '.join(skills)
                    context_info.append(f"Skills: {skills}")
                if person_details.get('office_location'):
                    context_info.append(f"Office Location: {person_details['office_location']}")
                if person_details.get('work_schedule'):
                    context_info.append(f"Work Schedule: {person_details['work_schedule']}")
                if person_details.get('team_size'):
                    context_info.append(f"Team Size: {person_details['team_size']}")
                if person_details.get('person_type'):
                    context_info.append(f"Type: {person_details['person_type']}")
                if person_details.get('special_notes'):
                    context_info.append(f"Notes: {person_details['special_notes']}")
            
            context_str = "\n".join(context_info) if context_info else "No additional details available"
            
            # Build fun, personalized prompt for creative greetings
            # IMPORTANT: Use Marathi Devanagari name for proper pronunciation by TTS
            prompt = f"""You are a SUPER ENTHUSIASTIC, cheerful, witty AI assistant greeting employees and visitors at an office.
Generate an ENERGETIC, FUN, PLAYFUL, and PERSONALIZED greeting for {marathi_name}.

Person Details:
{context_str}

Time: {time_greeting}

IMPORTANT: The person's name is provided in Marathi Devanagari script ({marathi_name}) for proper pronunciation by text-to-speech. Use this name exactly as provided in the greeting. The REST of the greeting should be in ENGLISH.

Guidelines:
- Use the name {marathi_name} EXACTLY as provided (Marathi Devanagari script)
- Make the REST of the greeting in ENGLISH - SUPER warm, friendly, ENERGETIC and LIGHT-HEARTED (2-3 sentences max)
- Use LOTS of exclamation marks to show enthusiasm and energy!
- Use their details creatively - reference their projects, interests, skills, or department
- Add a touch of HUMOR, gentle jokes, or playful compliments (keep it workplace-appropriate)
- Include light FLIRTY banter if appropriate (charming, not creepy - think friendly colleague vibes)
- Make them SMILE and feel PUMPED UP and energized for the day
- Keep it natural and conversational, but ENTHUSIASTIC, not overly formal
- Vary the greeting style - be creative and spontaneous!
- Add energy words like "awesome", "amazing", "fantastic", "brilliant", "superstar", "rockstar"
- Make it sound like you're genuinely EXCITED to see them!

Example styles (be MORE creative and ENERGETIC than these):
- "Hey {marathi_name}! WOW! Looking absolutely fantastic today! How's that AI project treating you? Those neural networks don't stand a chance against your brilliance! You're crushing it! 🚀"
- "Well well well! If it isn't the AMAZING {marathi_name} from Marketing! Your energy could power this whole building! Ready to absolutely DOMINATE those campaigns today? Let's GO! 💪"
- "Good morning SUPERSTAR! {marathi_name}! The team can't stop raving about your incredible work! Keep that genius brain caffeinated and keep being AWESOME! You're on fire! ☕🔥"
- "YES! {marathi_name}! The office just got 100% more AWESOME! Those Python skills are LEGENDARY! Are you writing code or pure magic today? Keep rocking it! 😄✨"

Generate ONE creative, ENERGETIC, fun greeting with LOTS of enthusiasm (just the greeting text, no quotes or extra formatting). Remember: Use {marathi_name} for the name, and English for everything else:"""

            response = self.model.generate_content(prompt)
            greeting = response.text.strip()
            
            # Clean up any quotes or extra formatting
            greeting = greeting.strip('"\'')
            
            logger.info(f"✨ Generated fun greeting: '{greeting}'")
            return greeting
            
        except Exception as e:
            logger.error(f"Error generating greeting with Gemini: {e}")
            # Fallback to energetic greeting with enthusiasm
            # Use Marathi name for better pronunciation
            marathi_name = convert_name_to_marathi(name, person_details)
            hour = datetime.now().hour
            if hour < 12:
                return f"Good morning, {marathi_name}! WOW! Looking absolutely fantastic today! Let's make it an AMAZING day! You're going to crush it! 🚀😊"
            elif hour < 17:
                return f"Good afternoon, {marathi_name}! You're doing AWESOME! Hope your day is going GREAT! Keep shining bright, superstar! ✨💫"
            else:
                return f"Good evening, {marathi_name}! YES! You made it through another day - you're an absolute ROCKSTAR! Way to go! 🌟🎉"


class AudioPlayer:
    """Simple audio player using pygame"""
    
    def __init__(self):
        if not PYGAME_AVAILABLE:
            logger.warning("pygame not available - audio playback disabled")
            self.enabled = False
            return
        
        try:
            # Increased buffer size for smoother playback, stereo channels for better quality
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096)
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
            
            # Wait for playback to finish with better timing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Add a small delay to ensure complete playback
            time.sleep(0.3)
            
            # Cleanup
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            
            logger.info("✅ Audio playback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            return False
    
    def cleanup(self):
        """Cleanup audio resources"""
        if self.enabled:
            try:
                # Stop any ongoing playback before cleanup
                pygame.mixer.music.stop()
                time.sleep(0.1)  # Small delay to ensure stop completes
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
    
    def greet_person(self, name: str, confidence: float = 1.0, person_details: Optional[Dict] = None) -> bool:
        """
        Generate and play a personalized greeting with full person details
        
        Args:
            name: Person's name
            confidence: Recognition confidence score
            person_details: Full enrollment details (position, interests, skills, etc.)
            
        Returns:
            True if greeting was successful
        """
        try:
            # Generate greeting with Gemini using all person details
            greeting_text = self.greeting_generator.generate_greeting(name, person_details)
            
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
    
    def greet_if_needed(self, name: str, confidence: float = 1.0, person_details: Optional[Dict] = None) -> bool:
        """Greet person only if cooldown has expired"""
        if self.should_greet(name):
            return self.greet_person(name, confidence, person_details)
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
    """Test function for the greeting system with personalized details"""
    from dotenv import load_dotenv
    load_dotenv()
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    sarvam_key = os.getenv("SARVAM_API_KEY")
    
    if not gemini_key or not sarvam_key:
        print("❌ Error: GEMINI_API_KEY and SARVAM_API_KEY must be set in .env file")
        return
    
    print("🚀 Testing Enhanced Greeting System with Fun Personalized Greetings!\n")
    
    # Initialize
    manager = GreetingManager(
        gemini_api_key=gemini_key,
        sarvam_api_key=sarvam_key,
        cooldown_minutes=1  # Short cooldown for testing
    )
    
    # Test with multiple personas to show different greeting styles
    test_people = [
        {
            "name": "Alex Chen",
            "details": {
                "position": "Senior Software Engineer",
                "department": "AI Research",
                "current_project": "Neural Network Optimization",
                "interests": ["Machine Learning", "Rock Climbing", "Coffee"],
                "skills": ["Python", "TensorFlow", "Deep Learning"],
                "person_type": "employee",
                "office_location": "Building A, Floor 5",
                "special_notes": "Always brings amazing coffee beans to share"
            }
        },
        {
            "name": "Priya Sharma",
            "details": {
                "position": "Marketing Director",
                "department": "Marketing",
                "current_project": "Q4 Brand Campaign",
                "interests": ["Photography", "Travel", "Yoga"],
                "skills": ["Content Strategy", "Brand Management", "Social Media"],
                "person_type": "employee",
                "team_size": "8",
                "special_notes": "Just won Best Campaign Award last month"
            }
        },
        {
            "name": "Raj Patel",
            "details": {
                "position": "Data Scientist",
                "department": "Analytics",
                "current_project": "Customer Behavior Prediction Model",
                "interests": ["Cricket", "Gaming", "Stand-up Comedy"],
                "skills": ["Python", "R", "Statistical Analysis", "Machine Learning"],
                "person_type": "employee",
                "work_schedule": "Flexible (10 AM - 7 PM)",
                "special_notes": "Office comedian, always lightens the mood"
            }
        }
    ]
    
    print("=" * 70)
    for person in test_people:
        print(f"\n🎭 Testing greeting for: {person['name']}")
        print(f"📋 Role: {person['details'].get('position', 'N/A')}")
        print(f"🏢 Department: {person['details'].get('department', 'N/A')}")
        print(f"💼 Project: {person['details'].get('current_project', 'N/A')}")
        print(f"❤️  Interests: {', '.join(person['details'].get('interests', []))}")
        print(f"✨ Notes: {person['details'].get('special_notes', 'N/A')}")
        print("\n🎤 Generating personalized greeting...")
        print("-" * 70)
        
        success = manager.greet_person(person['name'], 1.0, person['details'])
        
        if success:
            print(f"✅ Successfully greeted {person['name']}!")
        else:
            print(f"❌ Failed to greet {person['name']}")
        
        print("=" * 70)
        
        # Small delay between greetings
        import time
        time.sleep(2)
    
    # Show stats
    print("\n📊 Greeting Statistics:")
    stats = manager.get_stats()
    print(f"   Total people greeted: {stats['total_people_greeted']}")
    print(f"   Greeting counts: {stats['greeting_counts']}")
    print(f"   Cooldown: {stats['cooldown_minutes']} minutes")
    
    # Cleanup
    manager.cleanup()
    print("\n✅ Test complete! Check the 'outputs' folder for generated audio files.")
    print("🎉 The greetings are now fun, personalized, and include jokes/flirty banter!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    test_greeting_system()

