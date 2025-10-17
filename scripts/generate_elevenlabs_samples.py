import os
import sys
import json
from pathlib import Path
import logging
import requests


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        # dotenv is optional; continue if unavailable
        pass

    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if not api_key:
        logging.error("ELEVENLABS_API_KEY not found in environment (.env). Aborting.")
        sys.exit(1)

    text = (
        "Hi, welcome to Stark Digital!\n"
        "A very warm welcome to प्रभुराज धोंडगे, शुभम कांबळे, यश अमृतकर, आणि अमित मोहोळ sir."
    )

    voices = {
        #"Sia": "6JsmTroalVewG1gA6Jmw",
        "Koku": "UYoWPkHjaRgjWccloxC5",
        "karishma": "Zjz30d9v1e5xCxNVTni6",
        #"Anika_b": "9cStOzc7aNcCkPNah5jS",
        #"Natasha": "iWq9tCtTw1SYmVpvsRjY",
        #"Anika_c": "90ipbRoKi4CpHXvKVtl0",
        #"Tara": "fA4eSDsx5xNZ1RV7zEu5",
        #"Anika_d": "CoQByuTrT9gbKYx6QFL6",
        #"Tarini": "aAzJB4KWAzCZrJSxdt5x",
        #"Diana": "F2OOWcJMWhX8wCsyT0oR",
        #"Ananya": "ack0QsRaQyDLnVyMQTSd",
        #"Halle": "26C3KPrSUdSuPVU42fFc",
        #"Nina": "GUskjoz2EB74Wamu3r3D",
        #"Zara": "MmQVkVZnQ0dUbfWzcW6f",
        #"Meesha": "brHdTxI2cvSTiRe1fQlH",
        #"Saaavi": "H8bdWZHK2OgZwTN7ponr",
        #"Drashya": "qZCqzgsdxAHYGdLgJhis",
    }

    # Output directory '@Test Outputs/' relative to project root
    project_root = Path(__file__).resolve().parents[1]
    out_dir = project_root / "@Test Outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ElevenLabs config
    model_id = "eleven_turbo_v2_5"
    voice_settings = {
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.5,
        "use_speaker_boost": True,
    }

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }

    successes = 0
    failures = {}

    for name, voice_id in voices.items():
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        payload = {
            "text": text,
            "model_id": model_id,
            "voice_settings": voice_settings,
        }

        logging.info(f"Generating: {name} ({voice_id})")
        try:
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
            if resp.status_code == 200:
                audio_bytes = resp.content
                file_path = out_dir / f"{name}_sample.mp3"
                with open(file_path, "wb") as f:
                    f.write(audio_bytes)
                successes += 1
                logging.info(f"Saved: {file_path}")
            else:
                failures[name] = f"{resp.status_code}: {resp.text[:200]}"
                logging.error(f"Failed {name}: {failures[name]}")
        except Exception as e:
            failures[name] = str(e)
            logging.error(f"Error {name}: {e}")

    logging.info(f"Completed. Success: {successes}, Failures: {len(failures)}")
    if failures:
        for k, v in failures.items():
            logging.info(f"- {k}: {v}")


if __name__ == "__main__":
    main()


