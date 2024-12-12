# main.py
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List
from motor.motor_asyncio import AsyncIOMotorClient
import os
import yt_dlp
import re
import json
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security, Depends
import jwt
from dotenv import load_dotenv

security = HTTPBearer()

if(type(os.getenv('JWT_SECRET_KEY')) != str):
    SystemExit

SECRET_KEY = bytes.fromhex(os.getenv('JWT_SECRET_KEY'))
ALGORITHM = "HS256"

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )
            
        return payload
    except jwt.InvalidTokenError as e:
        print(f"JWT Validation Error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication credentials: {str(e)}"
        )


from aws_bedrock_client import generate_questions_from_transcript, evaluate_responses, QuestionPair, EvaluationPair

app = FastAPI()
origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# MongoDB setup
MONGO_DB_URL = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_DB_URL)
db = client.videoService_db
transcriptCollection = db.transcripts
questionCollection = db.questions

# Models 
class YouTubeLinkRequest(BaseModel):
    url: HttpUrl

class TranscriptResponse(BaseModel):
    id: str
    transcript: str

class QuestionGenerationRequest(BaseModel):
    id: str

class EvaluationRequest(BaseModel):
    pairs: List[dict]

class GeneratedQuestionsResponse(BaseModel):
    pairs: List[dict]

class EvaluationReport(BaseModel):
    report_id: str
    submission_date: str
    student_info: dict
    evaluation: dict
    finalized: bool

def extract_video_id(url: str) -> str:
    ydl_opts = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        video_info = ydl.extract_info(url, download=False)
        return video_info['id']

def extract_transcript(url: str) -> str:
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitlesformat": "vtt",
        "outtmpl": "transcripts/%(id)s.%(ext)s",  # save location
        "noprogress": True  
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_id = info['id']

        subtitle_filename = f"transcripts/{video_id}.en.vtt"
        auto_subtitle_filename = f"transcripts/{video_id}.en.auto.vtt"

        if 'en' in info.get('subtitles', {}):
            ydl.download([url])  # Download manual subtitles
            if os.path.exists(subtitle_filename):
                return clean_subtitle_file(subtitle_filename)
            elif os.path.exists(auto_subtitle_filename):
                return clean_subtitle_file(auto_subtitle_filename)
            else:
                raise Exception("Auto-generated subtitles not downloaded")

        elif 'en' in info.get('automatic_captions', {}):
            ydl.download([url])  # Download auto-generated subtitles
            if os.path.exists(auto_subtitle_filename):
                return clean_subtitle_file(auto_subtitle_filename)
            elif os.path.exists(subtitle_filename):
                return clean_subtitle_file(subtitle_filename)
            else:
                raise Exception("Auto-generated subtitles not downloaded")
        else:
            raise Exception("No subtitles available (manual or auto-generated)")

import re

def clean_subtitle_file(subtitle_file: str, min_words=3, max_words=10) -> str:
    cleaned_text = []
    with open(subtitle_file, 'r', encoding='utf-8') as f:
        for line in f:
            if re.match(r'^\d+$', line.strip()):
                continue  
            if '-->' in line:
                continue  
            line = re.sub(r'<[^>]*>|&\w+;', '', line)
            line = re.sub(r'\[.*?\]|\(.*?\)', '', line)
            line = line.strip()
            if line:
                cleaned_text.append(line)
    text = ' '.join(cleaned_text)

    text = remove_repeated_phrases(text, min_words, max_words)
    return text

def remove_repeated_phrases(text: str, min_words: int, max_words: int) -> str:
    words = text.split()
    i = 0
    result_words = []
    while i < len(words):
        found_repeat = False
        for n in range(max_words, min_words - 1, -1):
            if i + n * 2 <= len(words):
                phrase1 = words[i:i + n]
                phrase2 = words[i + n:i + 2 * n]
                if phrase1 == phrase2:
                    repeats = 2
                    while i + repeats * n <= len(words) and words[i + (repeats - 1) * n:i + repeats * n] == phrase1:
                        repeats += 1
                    result_words.extend(phrase1)
                    i += (repeats - 1) * n  
                    found_repeat = True
                    break  
        if not found_repeat:
            result_words.append(words[i])
            i += 1
    return ' '.join(result_words)

@app.post("/generate-transcript", response_model=TranscriptResponse)
async def generate_transcript(data: YouTubeLinkRequest, current_user: dict = Depends(get_current_user)):
    try:
        video_id = extract_video_id(data.url)

        # if transcript already exists in the database
        existing_transcript = await transcriptCollection.find_one({"_id": video_id})
        if existing_transcript:
            return {"id": video_id, "transcript": existing_transcript['transcript']}

        # Extract transcript if not already in the database
        transcript = extract_transcript(data.url)

        # MongoDB save
        await transcriptCollection.insert_one({"_id": video_id, "transcript": transcript})
        return {"id": video_id, "transcript": transcript}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate-questions")
async def generate_questions(data: QuestionGenerationRequest, current_user: dict = Depends(get_current_user)):
    try:
        # if transcript already exists in MongoDB
        existing_transcript = await transcriptCollection.find_one({"_id": data.id})
        if not existing_transcript:
            raise HTTPException(status_code=404, detail="Transcript not found")
        existing_questions = await questionCollection.find_one({"_id": data.id})
        if existing_questions:
            return {"id": data.id, "questions": existing_questions['questions']}

        question_pairs = generate_questions_from_transcript(existing_transcript['transcript'])

        response_pairs = [{"question": pair.question, "topic": pair.topic} for pair in question_pairs]

        response = {"pairs": response_pairs}
        await questionCollection.insert_one({"_id": data.id, "questions": response})

        return {"id": data.id, "questions": response}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/evaluate")
async def evaluate_answers(data: EvaluationRequest, current_user: dict = Depends(get_current_user)):
    try:
        system_prompt = """You are an elementary school teacher who is assigned to evaluate question-answer pairs (answered by students). <instruction>Respond in the following json schema, where reports is an array of report on each question
        {
            "evaluation": { , donot make an array, use this schema as it is strictly
                "score": "Number", out of 10,
                "topic": "String",
                "question_text": "String",
                "student_answer": "String", 
                "insight_gained": boolean, check whether the student has understood the subject material
                "follow_up_required": boolean, make the decision if the student needs to have a followup question
                "feedback": "String" provide a feedback upon the answer provided
            }
        },
    }</instruction>
"""
        evaluation_pairs = [EvaluationPair(question=pair['question'], topic=pair['topic'], answer=pair['answer']) for pair in data.pairs]
        evaluation_response = evaluate_responses(evaluation_pairs, system_prompt)
        json_text = evaluation_response[0]['text']
        parsed_json = json.loads(json_text)

        report_id = str(uuid.uuid4())

        return parsed_json

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
