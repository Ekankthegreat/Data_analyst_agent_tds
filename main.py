from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import os, io, base64, requests, tempfile
import google.generativeai as genai
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# System prompt for Gemini
SYSTEM_PROMPT = """
You are a data analysis assistant.
You will be given a user’s question along with optional context such as text, documents, spreadsheets, web links, or images.

Rules:
1. Always answer the question directly.
2. Output must ONLY contain the final answer, nothing else.
3. If the question requires visualization (bar chart, line plot, histogram, scatter plot, etc.),
   return valid Python code that generates the chart using matplotlib.
   Do not explain the code, just output the code itself.
4. If the answer is text-only, return the text as is.
"""

def extract_file_content(file: UploadFile) -> str:
    try:
        content = file.file.read()
        if file.filename.endswith(".txt"):
            return content.decode("utf-8")
        elif file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            return df.to_string()
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content))
            return df.to_string()
        elif file.filename.endswith((".png", ".jpg", ".jpeg")):
            return f"[Image uploaded: {file.filename}]"
        else:
            return f"[Unsupported file type: {file.filename}]"
    except Exception as e:
        return f"[Error reading file {file.filename}: {str(e)}]"

def fetch_webpage_text(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text()[:2000]
    except Exception as e:
        return f"[Error fetching {url}: {str(e)}]"

@app.post("/")
async def process_question(
    question: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    links: Optional[List[str]] = Form(None)
):
    try:
        # Collect all context
        context_parts = [f"User Question: {question}"]

        if files:
            for file in files:
                context_parts.append(extract_file_content(file))

        if links:
            for link in links:
                context_parts.append(fetch_webpage_text(link))

        full_context = "\n".join(context_parts)

        # Send prompt + context to Gemini
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content([SYSTEM_PROMPT, full_context])
        ai_output = response.text.strip()

        # Check if AI returned Python code for plotting
        if ai_output.startswith("```") and "matplotlib" in ai_output:
            code = ai_output.strip("`")  # remove markdown fences
            # Execute the code safely
            tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            exec_globals, exec_locals = {}, {}
            exec(code, {"plt": plt, "io": io, "base64": base64}, exec_locals)
            plt.savefig(tmpfile.name, format="png")
            with open(tmpfile.name, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            return {"answer": img_base64}

        # Otherwise return plain text answer
        return {ai_output}

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

