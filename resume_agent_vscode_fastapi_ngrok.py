"""
Resume Screening Agent
VSCode -> FastAPI -> ngrok

Based on the user's Sample10oct.ipynb flow:
- OpenAI ChatOpenAI (gpt-4o-mini)
- LangChain create_agent + job-description tool
- LangSmith tracing via .env
- Base64 encode/decode for API payloads
- FastAPI REST API
- ngrok public URL
- Final candidate analysis from interview Q&A

Run locally:
    python resume_agent_vscode_fastapi_ngrok.py

Or:
    uvicorn resume_agent_vscode_fastapi_ngrok:app --host 0.0.0.0 --port 8000

Required .env:
    OPENAI_API_KEY=...
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=...
    LANGSMITH_PROJECT=IIT-Madras-Customer-Support
    NGROK_AUTHTOKEN=...

NGROK_AUTH_TOKEN is also accepted for compatibility with the notebook.
"""

from __future__ import annotations

import base64 as _b64
import os
import threading
import time
from pathlib import Path
from typing import List

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")

# Support the existing notebook name as well as current ngrok SDK naming.
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN") or os.getenv("NGROK_AUTH_TOKEN")

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in .env")

if not LANGSMITH_API_KEY:
    raise EnvironmentError("LANGSMITH_API_KEY not found in .env")

if not LANGSMITH_PROJECT:
    raise EnvironmentError("LANGSMITH_PROJECT not found in .env")

if not NGROK_AUTHTOKEN:
    print("WARNING: NGROK_AUTHTOKEN / NGROK_AUTH_TOKEN not found.")
    print("FastAPI can still run locally, but ngrok will not start.")
else:
    # Current ngrok Python SDK reads NGROK_AUTHTOKEN from the environment.
    os.environ["NGROK_AUTHTOKEN"] = NGROK_AUTHTOKEN

# Keep compatibility with the tracing environment used in the notebook.
os.environ["LANGSMITH_TRACING"] = str(LANGSMITH_TRACING).lower()
os.environ["LANGCHAIN_TRACING_V2"] = str(LANGSMITH_TRACING).lower()
os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT

print("OpenAI configuration loaded.")
print("LangSmith configuration loaded.")
print(f"LangSmith project: {LANGSMITH_PROJECT}")
print(f"LangSmith tracing: {LANGSMITH_TRACING}")


# ============================================================
# 2. BASE64 HELPERS
# ============================================================

def encode_base64(text: str) -> str:
    """Encode UTF-8 text into Base64."""
    return _b64.b64encode(text.encode("utf-8")).decode("ascii")


def decode_base64(encoded_text: str) -> str:
    """Decode Base64 text back into UTF-8 text."""
    try:
        decoded = _b64.b64decode(encoded_text, validate=True)
        return decoded.decode("utf-8")
    except (_b64.binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError("Invalid Base64 or UTF-8 text") from exc


# ============================================================
# 3. OPTIONAL: CREATE BASE64-ENCODED APPLICATION SOURCE FILE
# ============================================================

def create_source_base64(
    source_file: str = "resume_agent_vscode_fastapi_ngrok.py",
    output_file: str = "resume_agent_vscode_fastapi_ngrok.py.b64",
) -> None:
    """Create a Base64 copy of the complete Python source file."""
    source_path = Path(source_file)
    output_path = Path(output_file)

    source_bytes = source_path.read_bytes()
    encoded = _b64.b64encode(source_bytes).decode("ascii")
    output_path.write_text(encoded, encoding="ascii")

    print(f"Base64 source created: {output_path.resolve()}")
    print(f"Base64 length: {len(encoded):,} characters")


# ============================================================
# 4. CREATE OPENAI LLM
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

print("OpenAI LLM created successfully.")


# ============================================================
# 5. JOB DESCRIPTION TOOL
# ============================================================

@tool
def analyze_job_description(job_description: str) -> str:
    """
    Analyze a job description and identify key skills, experience
    requirements, responsibilities, qualifications, and keywords.
    """
    prompt = f"""
Analyze the following Job Description.

Identify:
1. Required technical skills
2. Required soft skills
3. Required years of experience
4. Key responsibilities
5. Required qualifications
6. Important keywords

Job Description:
{job_description}

Return the analysis in a clear structured format.
"""

    response = llm.invoke(prompt)
    return response.content


# ============================================================
# 6. LANGCHAIN RESUME SCREENING AGENT
# ============================================================

agent = create_agent(
    model=llm,
    tools=[analyze_job_description],
    system_prompt="""
You are a Resume Screening Agent.

Your job is to help analyze a candidate's resume against a Job Description.

You have access to a tool called analyze_job_description.

When the user provides a Job Description:
1. Understand what the user is asking.
2. Use the appropriate tool when necessary.
3. Do not invent information.
4. Ask for additional information when it is required.
5. Keep your responses clear and concise.
""",
)

print("Resume Screening Agent created successfully.")


# ============================================================
# 7. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Resume Screening Agent",
    description="AI-powered Resume Screening API using LangChain, OpenAI and LangSmith",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Resume Screening Agent API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "resume-screening-agent",
        "langsmith_project": LANGSMITH_PROJECT,
    }


# ============================================================
# 8. REQUEST MODELS
# ============================================================

class JobDescriptionRequest(BaseModel):
    job_description: str = Field(..., min_length=1)


class Base64JobDescriptionRequest(BaseModel):
    job_description_base64: str = Field(..., min_length=1)


class InterviewItem(BaseModel):
    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)


class InterviewQuestionRequest(BaseModel):
    job_description: str = Field(..., min_length=1)
    previous_interview: List[InterviewItem] = Field(default_factory=list)


class FinalAnalysisRequest(BaseModel):
    job_description: str = Field(..., min_length=1)
    interview: List[InterviewItem] = Field(..., min_length=1)


# ============================================================
# 9. JD ANALYSIS ENDPOINT
# ============================================================

@app.post("/analyze-job-description")
def analyze_job_description_api(request: JobDescriptionRequest):
    try:
        response = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"""
Analyze the following Job Description.

Job Description:
{request.job_description}
""",
                }
            ]
        })

        result = response["messages"][-1].content

        return {
            "job_description": request.job_description,
            "analysis": result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ============================================================
# 10. BASE64 JD ANALYSIS ENDPOINT
# ============================================================

@app.post("/analyze-job-description-base64")
def analyze_job_description_base64(
    request: Base64JobDescriptionRequest,
):
    try:
        job_description = decode_base64(request.job_description_base64)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        response = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": f"""
Analyze the following Job Description.

Job Description:
{job_description}
""",
                }
            ]
        })

        result = response["messages"][-1].content

        return {
            "job_description": job_description,
            "analysis": result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ============================================================
# 11. GENERATE NEXT INTERVIEW QUESTION
# ============================================================

def build_interview_history(interview: List[InterviewItem]) -> str:
    if not interview:
        return "No previous interview answers."

    return "\n".join(
        f"Question: {item.question}\nCandidate Answer: {item.answer}\n"
        for item in interview
    )


@app.post("/generate-interview-question")
def generate_interview_question(request: InterviewQuestionRequest):
    conversation = build_interview_history(request.previous_interview)

    prompt = f"""
You are conducting a candidate screening interview.

Job Description:
{request.job_description}

Previous interview:
{conversation}

Generate the next interview question.

Rules:
- Ask exactly ONE question.
- Do not repeat a previous question.
- Do not ask something that has already been clearly answered.
- Focus on an important requirement from the Job Description.
- The question should help determine whether the candidate matches the Job Description.
- Do not provide analysis.
- Keep the question concise.
"""

    try:
        response = agent.invoke({
            "messages": [{"role": "user", "content": prompt}]
        })
        question = response["messages"][-1].content
        return {"question": question}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ============================================================
# 12. FINAL CANDIDATE ANALYSIS
# ============================================================

@app.post("/final-analysis")
def final_analysis(request: FinalAnalysisRequest):
    interview_transcript = "\n".join(
        f"""
Question {i}:
{item.question}

Candidate Answer:
{item.answer}

--------------------------------------
"""
        for i, item in enumerate(request.interview, start=1)
    )

    final_prompt = f"""
You are a professional Resume Screening Agent.

Analyze the candidate against the Job Description using ONLY the information provided below.

================ JOB DESCRIPTION ================

{request.job_description}

================ INTERVIEW ======================

{interview_transcript}

==================================================

Prepare a structured candidate screening report.

Include:

1. Candidate Profile Summary

2. Key Requirements from the Job Description

3. Matching Skills
   - Skills clearly demonstrated by the candidate

4. Missing or Unclear Skills
   - Requirements that were not demonstrated
   - Clearly distinguish between "missing" and "not enough information"

5. Experience Assessment
   - Relevant experience
   - Industry/domain experience
   - Project experience

6. Key Strengths

7. Potential Gaps

8. Resume Improvement Suggestions
   - Specific keywords or experience the candidate should highlight
   - Do not invent experience

9. Interview Follow-up Areas
   - Areas that may require further verification

Important:
- Do not invent candidate experience.
- Do not assume a skill just because it is mentioned in the Job Description.
- Base the analysis only on the candidate's answers.
- Be factual and concise.
"""

    try:
        final_response = llm.invoke(final_prompt)
        return {
            "job_description": request.job_description,
            "interview": [item.model_dump() for item in request.interview],
            "analysis": final_response.content,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ============================================================
# 13. START LOCAL FASTAPI + NGROK
# ============================================================

APP_HOST = "0.0.0.0"
APP_PORT = 8000


def start_fastapi_server():
    """Start Uvicorn in the current process/thread."""
    import uvicorn

    uvicorn.run(
        app,
        host=APP_HOST,
        port=APP_PORT,
        reload=False,
    )


def start_ngrok():
    """Create a public ngrok tunnel to the local FastAPI server."""
    if not NGROK_AUTHTOKEN:
        print("ngrok not started because no NGROK_AUTHTOKEN was supplied.")
        return None

    try:
        import ngrok
    except ImportError as exc:
        print("Install the ngrok Python SDK with: pip install ngrok")
        raise exc

    forwarder = ngrok.forward(
        f"localhost:{APP_PORT}",
        authtoken_from_env=True,
    )

    public_url = forwarder.url()
    print("\n======================================")
    print("          NGROK PUBLIC URL")
    print("======================================")
    print(public_url)
    print(f"Swagger docs: {public_url}/docs")
    print(f"Health check: {public_url}/health")
    print("======================================\n")

    return forwarder


def wait_for_server(timeout_seconds: int = 20):
    """Wait until the local FastAPI health endpoint responds."""
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        try:
            response = requests.get(
                f"http://127.0.0.1:{APP_PORT}/health",
                timeout=2,
            )
            if response.status_code == 200:
                print("FastAPI server is ready.")
                print(response.json())
                return True
        except requests.RequestException:
            pass

        time.sleep(0.5)

    raise TimeoutError(
        f"FastAPI did not become ready within {timeout_seconds} seconds."
    )


def main():
    print("\n======================================")
    print("     RESUME SCREENING AGENT API")
    print("======================================")
    print(f"Starting FastAPI on port {APP_PORT}...")

    server_thread = threading.Thread(
        target=start_fastapi_server,
        daemon=True,
    )
    server_thread.start()

    wait_for_server()

    print(f"Local API: http://127.0.0.1:{APP_PORT}")
    print(f"Swagger:   http://127.0.0.1:{APP_PORT}/docs")

    start_ngrok()

    print("Server is running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping application...")


if __name__ == "__main__":
    main()
