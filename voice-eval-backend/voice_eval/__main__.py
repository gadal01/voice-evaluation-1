import uvicorn

from voice_eval.config import get_settings

if __name__ == "__main__":
    uvicorn.run("voice_eval.app:app", host="127.0.0.1", port=get_settings().port)
