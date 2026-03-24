# main.py
import base64
import imghdr
import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="OCR API")

# Update origins to include your Vercel URL after deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://ocr-frontend.vercel.app",     # your Vercel URL
        "https://ocr-frontend-*.vercel.app",   # preview deployments
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


@app.get("/")
def health():
    return {"status": "V1.0.0"}


@app.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")

    if imghdr.what(None, contents) is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image data",
        )

    b64_image = base64.b64encode(contents).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{file.content_type};base64,{b64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Extract all text visible in this image. "
                                "Return only the extracted text, nothing else."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=1024,
        )
        extracted = response.choices[0].message.content
        return {"message": extracted, "filename": file.filename}

    except Exception as e:
        err_msg = str(e)
        if "invalid image data" in err_msg.lower():
            raise HTTPException(status_code=400, detail="Invalid image data")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
