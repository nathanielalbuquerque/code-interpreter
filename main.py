import json
import os
import sys
import traceback
from io import StringIO
from typing import List

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    stdout = StringIO()
    sys.stdout = stdout

    try:
        exec(code)

        return {
            "success": True,
            "output": stdout.getvalue()
        }

    except Exception:
        return {
            "success": False,
            "output": traceback.format_exc()
        }

    finally:
        sys.stdout = old_stdout


def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:

    token = os.environ.get("AIPIPE_TOKEN")

    if not token:
        raise RuntimeError("AIPIPE_TOKEN environment variable is not set")

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

    response = requests.post(
        "https://aipipe.org/openrouter/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openai/gpt-4.1-nano",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0
        },
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"AI Pipe HTTP {response.status_code}: {response.text}"
        )

    data = response.json()

    content = data["choices"][0]["message"]["content"]

    if not content:
        raise RuntimeError("AI Pipe returned empty content")

    content = content.strip()

    # Handle accidental markdown code fences
    if content.startswith("```"):
        content = content.replace("```json", "", 1)
        content = content.replace("```", "")
        content = content.strip()

    parsed = json.loads(content)

    result = ErrorAnalysis.model_validate(parsed)

    return result.error_lines


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok"}


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):

    execution = execute_python_code(request.code)

    # Successful execution: do NOT call AI
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # Error: invoke AI
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    return {
        "error": error_lines,
        "result": execution["output"]
    }
