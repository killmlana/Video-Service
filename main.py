# main.py
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
from motor.motor_asyncio import AsyncIOMotorClient
import os
import yt_dlp
import re
import json

from aws_bedrock_client import generate_questions_from_transcript, evaluate_responses, QuestionPair, EvaluationPair

app = FastAPI()

# MongoDB setup
MONGO_DB_URL = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_DB_URL)
db = client.transcripts_db
collection = db.transcripts

# Models 
class YouTubeLinkRequest(BaseModel):
    url: HttpUrl

class TranscriptResponse(BaseModel):
    id: str
    transcript: str

class QuestionGenerationRequest(BaseModel):
    id: str
    transcript: str

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
            else:
                raise Exception("Manual subtitles not downloaded")

        elif 'en' in info.get('automatic_captions', {}):
            ydl.download([url])  # Download auto-generated subtitles
            if os.path.exists(auto_subtitle_filename):
                return clean_subtitle_file(auto_subtitle_filename)
            else:
                raise Exception("Auto-generated subtitles not downloaded")
        else:
            raise Exception("No subtitles available (manual or auto-generated)")

def clean_subtitle_file(subtitle_file: str) -> str:
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
    return ' '.join(cleaned_text)

@app.post("/generate-transcript", response_model=TranscriptResponse)
async def generate_transcript(data: YouTubeLinkRequest):
    try:
        video_id = extract_video_id(data.url)

        # if transcript already exists in the database
        existing_transcript = await collection.find_one({"id": video_id})
        if existing_transcript:
            return {"_id": video_id, "transcript": existing_transcript['transcript']}

        # Extract transcript if not already in the database
        transcript = extract_transcript(data.url)

        # MongoDB save
        await collection.insert_one({"_id": video_id, "transcript": transcript})
        return {"_id": video_id, "transcript": transcript}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/generate-questions", response_model=GeneratedQuestionsResponse)
async def generate_questions(data: QuestionGenerationRequest):
    try:
        # if transcript already exists in MongoDB
        existing_transcript = await collection.find_one({"_id": data.id})
        if not existing_transcript:
            raise HTTPException(status_code=404, detail="Transcript not found")

        question_pairs = generate_questions_from_transcript(data.transcript)

        response_pairs = [{"question": pair.question, "topic": pair.topic} for pair in question_pairs]

        return {"pairs": response_pairs}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/evaluate")
async def evaluate_answers(data: EvaluationRequest):
    try:
        system_prompt = """You are an elementary school teacher who is assigned to evaluate question-answer pairs (answered by students). Respond in the following json schema, where reports is an array of report on each question
        { "reports" = [
            {
            "student_info": {
                "UserId": "String"
            },
            "evaluation": {
                "total_score": "Float",
                "questions": [
                {
                    "topic": "String",
                    "question_text": "String",
                    "student_answer": "String", 
                    "insight_gained": boolean, check whether the student has understood the subject material
                    "follow_up_required": boolean, make the decision if the student needs to have a followup question
                    "feedback": "String" provide a feedback upon the answer provided
                }
                ]
            },
            "finalized": true
            }]
        }
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
