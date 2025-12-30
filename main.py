from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# Rate Limiting Imports
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# 1. Initialize Rate Limiter (Tracks by Client IP)
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(debug=True)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 2. Add CORS Middleware
# Replace "*" with your Java Server's IP/Domain in production for better security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["POST"], # We only need POST for analysis
    allow_headers=["*"],
)

class AnalysisReport(BaseModel):
    time_complexity: str
    space_complexity: str
    logic_faults: list[str]
    improvement_explanation: str
    corrected_code: str

class CodeInput(BaseModel):
    code: str
    language: str


client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY")).aio

@app.post("/analyze", response_model=AnalysisReport)
@limiter.limit("5/minute") # Limit: 5 requests per minute per user/IP
async def analyze_code(input_data: CodeInput, request: Request):
    try:
        response = await client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"LANGUAGE: {input_data.language}\nCODE:\n{input_data.code}",
            config=types.GenerateContentConfig(
                system_instruction="You are a Senior Software Architect and Coding Mentor. Your goal is to help users improve their technical depth by providing rigorous code analysis.For every code snippet provided, follow these logical steps:Dry Run: Mentally execute the code with an edge-case input to find logic gaps.Metric Extraction: Calculate Big O Time and Space complexity.Fault Identification: Look for anti-patterns, security risks, or inefficiencies.Synthesis: Write a corrected version and explain the 'why' behind the changes.Output Requirements:Format the response in valid JSON (for API parsing).Use Markdown inside the JSON strings for readability.Be technically precise but encouraging.",
                response_mime_type="application/json",
                response_schema=AnalysisReport,
                temperature=0.2,
            ),
        )
        return response.parsed
    except Exception as e:
        # ADD THIS LINE: It will print the real error to your VS Code terminal
        print(f"DEBUG ERROR: {e}")
        raise HTTPException(status_code=500, detail="AI Analysis Service is currently unavailable.")