import json
import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel


app = FastAPI()


# CORS - required by the assignment grader
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Request / response models
# -----------------------------

class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# -----------------------------
# Python execution tool
# -----------------------------

def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return the exact stdout or traceback.
    """

    old_stdout = sys.stdout
    stdout = StringIO()

    sys.stdout = stdout

    try:
        exec(code)

        output = stdout.getvalue()

        return {
            "success": True,
            "output": output
        }

    except Exception:
        output = traceback.format_exc()

        return {
            "success": False,
            "output": output
        }

    finally:
        sys.stdout = old_stdout


# -----------------------------
# AI error analysis
# -----------------------------
def analyze_error_with_ai(
    code: str,
    error_traceback: str
) -> List[int]:

    token = os.environ.get("AIPIPE_TOKEN")

    if not token:
        raise RuntimeError("AIPIPE_TOKEN environment variable is not set")

    client = OpenAI(
        api_key=token,
        base_url="https://aipipe.org/openrouter/v1"
    )

    prompt = f"""
Analyze the following Python code and traceback.

Identify the exact source-code line number or line numbers
where the error occurred.

Return ONLY valid JSON in exactly this format:

{{"error_lines":[3]}}

Do not include markdown.
Do not include explanations.

CODE:
{code}

TRACEBACK:
{error_traceback}
"""

    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-lite-001",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("AI returned an empty response")

    content = content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "", 1)
        content = content.replace("```", "")
        content = content.strip()

    parsed = json.loads(content)

    result = ErrorAnalysis.model_validate(parsed)

    return result.error_lines

# -----------------------------
# Root / health endpoint
# -----------------------------

@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok"}


# -----------------------------
# Main assignment endpoint
# -----------------------------

@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):

    # Step 1: Execute the submitted Python code
    execution = execute_python_code(request.code)

    # Step 2: Successful execution
    # IMPORTANT: AI is NOT called here.
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # Step 3: Error occurred.
    # Only now call the AI.
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    # Step 4: Return exact traceback unchanged
    return {
        "error": error_lines,
        "result": execution["output"]
    }
