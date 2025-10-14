
import os
import io
import torch
import soundfile as sf
from fastapi import FastAPI, Response, Request
from pydantic import BaseModel
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

# Model setup
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_ID = os.getenv('INDIC_PARLER_MODEL', 'ai4bharat/indic-parler-tts')

description_tokenizer = None
prompt_tokenizer = None
model = None

app = FastAPI(title='Local Indic Parler TTS')

class TTSRequest(BaseModel):
    text: str
    speaker: str | None = None  # description/caption

@app.on_event('startup')
async def startup():
    global model, prompt_tokenizer, description_tokenizer
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
    prompt_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    # Warmup
    desc = 'A friendly Indian English receptionist voice, warm and clear, medium pace and pitch.'
    prompt = 'Hello.'
    with torch.no_grad():
        di = description_tokenizer(desc, return_tensors='pt').to(DEVICE)
        pi = prompt_tokenizer(prompt, return_tensors='pt').to(DEVICE)
        _ = model.generate(input_ids=di.input_ids, attention_mask=di.attention_mask,
                           prompt_input_ids=pi.input_ids, prompt_attention_mask=pi.attention_mask)

@app.get('/health')
async def health():
    return {
        'ready': model is not None,
        'device': DEVICE,
        'model_id': MODEL_ID
    }

@app.post('/tts')
async def tts(request: Request):
    # Accept either {"text":..., "speaker":...} or {"inputs": {"text":..., "speaker":...}}
    body = await request.json()
    if isinstance(body, dict) and 'inputs' in body and isinstance(body['inputs'], dict):
        text = body['inputs'].get('text', '')
        speaker = body['inputs'].get('speaker')
    else:
        text = body.get('text', '') if isinstance(body, dict) else ''
        speaker = body.get('speaker') if isinstance(body, dict) else None

    if not text:
        return Response(content=b'', media_type='audio/wav', status_code=400)

    desc = speaker or 'A female Indian English receptionist voice, warm, friendly, clear studio-quality, natural prosody.'
    with torch.no_grad():
        di = description_tokenizer(desc, return_tensors='pt').to(DEVICE)
        pi = prompt_tokenizer(text, return_tensors='pt').to(DEVICE)
        gen = model.generate(input_ids=di.input_ids, attention_mask=di.attention_mask,
                             prompt_input_ids=pi.input_ids, prompt_attention_mask=pi.attention_mask)
        audio = gen.cpu().numpy().squeeze()
        sr = model.config.sampling_rate
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format='WAV')
    buf.seek(0)
    return Response(content=buf.read(), media_type='audio/wav')
