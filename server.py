import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import tempfile
import os
import subprocess

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables to store the model and pipeline
global_model = None
global_pipe = None

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
local_model_dir = "./whisper_large_v3_turbo"

def load_model():
    global global_model, global_pipe
    
    # Load the model
    global_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        local_model_dir, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    global_model.to(device)

    # Load the processor
    processor = AutoProcessor.from_pretrained(local_model_dir)

    # Create the pipeline
    global_pipe = pipeline(
        "automatic-speech-recognition",
        model=global_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')



@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    original_extension = os.path.splitext(file.filename)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_extension) as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name

    try:
        wav_path = f"{temp_file_path}.wav"
        subprocess.run([
            'ffmpeg', '-i', temp_file_path, '-ar', '16000', '-ac', '1', wav_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        result = global_pipe(wav_path)
        return {"transcription": result["text"]}
    
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode()
        return {"error": f"Audio conversion failed: {error_message}"}
    
    except Exception as e:
        return {"error": str(e)}
    
    finally:
        os.unlink(temp_file_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)