from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError, Field
from typing import Literal, List
import uvicorn
import os
import subprocess
import random
import string
from dotenv import load_dotenv
import aiohttp
import re
import hashlib
from sqlalchemy import (
    create_engine, Column, Integer, String, ForeignKey, Enum
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, Session
import enum
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError
from google.cloud import texttospeech

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
if OPENAI_API_KEY is None:
    raise Exception("Yo buddy, OPENAI_API_KEY not defined. Check your .env file.")
if BASE_URL is None:
    BASE_URL = "http://localhost:8000"

# Initialize prompts
prompts = {}

with open("prompts/latex_prompt", "r") as f:
    prompts["latex"] = f.read()
with open("prompts/latex_chat_prompt", "r") as f:
    prompts["latex_chat"] = f.read()

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global HTTP client
http_client = None
async def get_client():
    global http_client
    if http_client is None:
        http_client = aiohttp.ClientSession()
    return http_client

# Create necessary directories
os.makedirs("database", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Database setup
DATABASE_URL = "sqlite:///./database/lesson_plans.db"


engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

# Enum for lesson types
class LessonTypeEnum(enum.Enum):
    latex = "latex"
    linux = "linux"

# ORM models
class LessonORM(Base):
    __tablename__ = 'lessons'
    
    id = Column(Integer, primary_key=True, index=True)
    lesson_plan_id = Column(Integer, ForeignKey('lesson_plans.id'), nullable=False)
    title = Column(String, nullable=False)
    lesson_text = Column(String, nullable=False)
    solution = Column(String, nullable=False)
    task = Column(String, nullable=False)
    lesson_type = Column(Enum(LessonTypeEnum), nullable=False)
    solution_boilerplate = Column(String, nullable=False)

class LessonPlanORM(Base):
    __tablename__ = 'lesson_plans'
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False, unique=True)
    author = Column(String, nullable=False)
    
    # Establish relationship with LessonORM
    sublessons = relationship("LessonORM", backref="lesson_plan", cascade="all, delete-orphan")

# Create all tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class Lesson(BaseModel):
    title: str
    lesson_text: str
    solution: str
    task: str
    lesson_type: Literal["latex", "linux"]
    solution_boilerplate: str

class LessonPlan(BaseModel):
    id: int
    title: str
    author: str
    sublessons: list[Lesson]

    @classmethod
    def get(cls, db: Session, lesson_id: int):
        lesson_plan_orm = db.query(LessonPlanORM).filter(LessonPlanORM.id == lesson_id).first()
        
        if not lesson_plan_orm:
            return None

        return cls(
            id=lesson_plan_orm.id,
            title=lesson_plan_orm.title,
            author=lesson_plan_orm.author,
            sublessons=[
                Lesson(
                    title=lesson.title,
                    lesson_text=lesson.lesson_text,
                    solution=lesson.solution,
                    task=lesson.task,
                    lesson_type=lesson.lesson_type.value,
                    solution_boilerplate=lesson.solution_boilerplate
                )
                for lesson in lesson_plan_orm.sublessons
            ]
        )

class LessonChatMessage(BaseModel):
    role: Literal["assistant", "user"] = "assistant"
    content: str

class SubmissionResult(BaseModel):
    success: bool
    problem: str | None = None
    hint: str | None = None
    output_url: str | None = None
    output: str | None = None
    audio_url: str | None = None  # New field for audio

class SubLessonCreate(BaseModel):
    title: str
    prompt: str
    question: str
    type: Literal["latex", "linux"]
    solutionBoilerplate: str

class LessonPlanCreate(BaseModel):
    title: str
    author: str
    sublessons: List[SubLessonCreate]

class SubLessonCreateRequest(BaseModel):
    title: str
    prompt: str
    question: str
    type: Literal["latex", "linux"]
    solutionBoilerplate: str

class AudioResponse(BaseModel):
    audio_url: str

class TTSRequest(BaseModel):
    text: str
    language: str = "en-US"  # Default language code
    voice_name: str = "en-US-Wavenet-D"  # Default voice name

# In-memory cache for TTS (optional)
tts_cache = {}

# Utility function to convert text to speech
async def text_to_speech(text: str, language: str = "en-US", voice_name: str = "en-US-Wavenet-D") -> str:
    """
    Converts text to speech using Google Cloud Text-to-Speech API and returns the URL to the audio file.

    Args:
        text (str): The text to convert to speech.
        language (str): Language code (e.g., 'en-US').
        voice_name (str): The name of the voice.

    Returns:
        str: URL to the generated audio file.
    """
    # Create a unique hash for the text to implement caching
    text_hash = hashlib.md5(text.encode()).hexdigest()

    # Check if the audio URL is already cached
    if text_hash in tts_cache:
        print("Fetching audio URL from cache.")
        return tts_cache[text_hash]

    try:
        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)

        # Build the voice request
        voice = texttospeech.VoiceSelectionParams(
            language_code=language,
            name=voice_name
        )

        # Select the type of audio file you want returned
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        # Perform the text-to-speech request
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # Save the audio to a file
        audio_filename = f"static/{text_hash}.mp3"
        with open(audio_filename, "wb") as out:
            out.write(response.audio_content)
            print(f"Audio content written to file {audio_filename}")

        audio_url = f"{BASE_URL}/static/{text_hash}.mp3"
        # Cache the result
        tts_cache[text_hash] = audio_url
        return audio_url

    except Exception as e:
        print(f"Unexpected Error in TTS: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# Endpoint to get all lesson plans
@app.get("/lessons/")
def get_lesson_plans(db: Session = Depends(get_db)):
    lesson_plans = db.query(LessonPlanORM).all()
    return [{"id": lp.id, "title": lp.title, "author": lp.author} for lp in lesson_plans]

# Endpoint to create a new lesson plan
@app.post("/lessons/", response_model=dict)
def create_lesson(lesson_plan_data: LessonPlanCreate, db: Session = Depends(get_db)):
    try:
        new_lesson_plan = LessonPlanORM(
            title=lesson_plan_data.title,
            author=lesson_plan_data.author,
            sublessons=[
                LessonORM(
                    title=sub.title,
                    lesson_text=sub.prompt,
                    solution=sub.question,
                    task="Refer to LaTeX documentation.",
                    lesson_type=LessonTypeEnum[sub.type],
                    solution_boilerplate=sub.solutionBoilerplate
                ) for sub in lesson_plan_data.sublessons
            ]
        )
        
        db.add(new_lesson_plan)
        db.commit()
        db.refresh(new_lesson_plan)
        print(f"Lesson Plan Created: ID {new_lesson_plan.id}")
        return {"message": "Lesson created successfully!", "lesson_plan_id": new_lesson_plan.id}
    except ValidationError as e:
        db.rollback()
        print(f"Validation Error: {e.errors()}")
        raise HTTPException(status_code=400, detail=f"Validation error: {e.errors()}")
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Database Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    except Exception as e:
        db.rollback()
        print(f"Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Endpoint to get a specific lesson plan
@app.get("/lessons/{lesson_id}")
def lesson(lesson_id: int, db: Session = Depends(get_db)):
    lesson_plan = LessonPlan.get(db, lesson_id)
    if not lesson_plan:
        print(f"Lesson Plan with ID {lesson_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"LessonPlan with id {lesson_id} not found."
        )
    return lesson_plan

@app.post("/lessons/{lesson_id}/sublessons", response_model=dict)
def add_sublesson(
    lesson_id: int,
    sublesson_data: SubLessonCreateRequest,
    db: Session = Depends(get_db)
):
    try:
        # Fetch the lesson plan
        lesson_plan = db.query(LessonPlanORM).filter(LessonPlanORM.id == lesson_id).first()
        if not lesson_plan:
            raise HTTPException(status_code=404, detail="Lesson Plan not found.")

        # Create a new LessonORM instance
        new_sublesson = LessonORM(
            lesson_plan_id=lesson_id,
            title=sublesson_data.title,
            lesson_text=sublesson_data.prompt,
            solution=sublesson_data.question,
            task="Refer to relevant documentation.",
            lesson_type=LessonTypeEnum[sublesson_data.type],
            solution_boilerplate=sublesson_data.solutionBoilerplate
        )

        # Add the sublesson to the lesson plan
        lesson_plan.sublessons.append(new_sublesson)
        db.add(new_sublesson)
        db.commit()
        db.refresh(new_sublesson)

        return {"message": "Sublesson added successfully!", "sublesson_id": new_sublesson.id}
    except ValidationError as e:
        db.rollback()
        print(f"Validation Error: {e.errors()}")
        raise HTTPException(status_code=400, detail=f"Validation error: {e.errors()}")
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Database Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    except Exception as e:
        db.rollback()
        print(f"Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Endpoint to submit a lesson
@app.post("/lessons/{lesson_id}/{sublesson_id}/submit", response_model=SubmissionResult)
async def check_submission(
    lesson_id: int,
    sublesson_id: int,
    submission: str,
    check_solution: bool,
    db: Session = Depends(get_db)
) -> SubmissionResult | None:
    try:
        lesson_plan = LessonPlan.get(db, lesson_id=lesson_id)
        if not lesson_plan or sublesson_id >= len(lesson_plan.sublessons):
            print(f"Sublesson {sublesson_id} not found in Lesson Plan {lesson_id}.")
            raise HTTPException(status_code=404, detail="Sublesson not found.")
        sublesson = lesson_plan.sublessons[sublesson_id]

        # Convert submission to HTML using Pandoc
        filename = "".join(random.choices(string.ascii_letters, k=8)) + ".html"
        pandoc_process = subprocess.run(
            ["pandoc", "--mathml"],
            input=submission.encode(),
            capture_output=True
        )
        if pandoc_process.returncode != 0:
            print("Pandoc conversion failed.")
            return SubmissionResult(success=False, problem="Doesn't compile")
        
        # Save the HTML output to a file
        with open(f"static/{filename}", "w", encoding="utf-8") as f:
            f.write(pandoc_process.stdout.decode())
        
        output_url = f"{BASE_URL}/static/{filename}"
        output = pandoc_process.stdout.decode()

        if not check_solution:
            # Generate TTS for the output
            try:
                audio_url = await text_to_speech(output)
            except HTTPException as e:
                print(f"TTS Service Unavailable: {e.detail}")
                audio_url = None
                return SubmissionResult(success=False, problem="TTS service unavailable.", output=output)
            return SubmissionResult(success=False, problem="Not Checking Solution", output=output, audio_url=audio_url)

        prompt = prompts[sublesson.lesson_type].format(
            task=sublesson.task,
            solution=sublesson.solution,
            submission=submission,
        )

        client = await get_client()
        async with client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": "Bearer " + OPENAI_API_KEY},
            json={
                "model": "gpt-4o-mini",  # Ensure this is a valid model
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful teaching assistant who is helping the professor grade student submissions.",
                    },
                    {"role": "user", "content": prompt},
                ],
            },
        ) as resp:
            if not resp.ok:
                error_detail = await resp.text()
                print(f"OpenAI API Error [{resp.status}]: {error_detail}")
                raise HTTPException(status_code=500, detail="OpenAI service unavailable.")
            
            resp_json = await resp.json()
            response = resp_json["choices"][0]["message"]["content"]
            matches = re.match(r".*[Pp]ass:\s*(.*)\s*[Hh]int:\s*([\s\S]*)", response)
            if matches is None:
                print("OpenAI response parsing failed.")
                raise HTTPException(status_code=500, detail="Invalid response from OpenAI.")
            success, hint = matches[1], matches[2]
            print(f"Grading Result - Success: {success}, Hint: {hint}")
            if success.strip().lower() in ["yes", "success"]:
                # Generate TTS for the success response
                try:
                    audio_url = await text_to_speech(hint)
                except HTTPException as e:
                    print(f"TTS Service Unavailable for Success Hint: {e.detail}")
                    audio_url = None
                return SubmissionResult(
                    success=True,
                    output=output,
                    hint=hint,
                    audio_url=audio_url
                )
            else:
                # Generate TTS for the failure response
                try:
                    audio_url = await text_to_speech(hint)
                except HTTPException as e:
                    print(f"TTS Service Unavailable for Failure Hint: {e.detail}")
                    audio_url = None
                return SubmissionResult(
                    success=False,
                    problem="Compiles but incorrect latex.",
                    hint=hint,
                    output=output,
                    audio_url=audio_url
                )
    except HTTPException as e:
        print(f"HTTP Exception in Submission: {e.detail}")
        raise e
    except Exception as e:
        print(f"Unexpected Error in Submission: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# Endpoint to handle chat interactions
@app.post("/lessons/{lesson_id}/{sublesson_id}/chat")
async def chat(
    lesson_id: int,
    sublesson_id: int,
    submission: str,
    messages: list[LessonChatMessage],
    db: Session = Depends(get_db)
) -> LessonChatMessage | None:
    try:
        lesson_plan = LessonPlan.get(db, lesson_id=lesson_id)
        if not lesson_plan or sublesson_id >= len(lesson_plan.sublessons):
            print(f"Sublesson {sublesson_id} not found in Lesson Plan {lesson_id}.")
            raise HTTPException(status_code=404, detail="Sublesson not found.")
        sublesson = lesson_plan.sublessons[sublesson_id]
        client = await get_client()
        prompt = prompts.get(f"{sublesson.lesson_type}_chat")
        if not prompt:
            print(f"No chat prompt found for lesson type {sublesson.lesson_type}.")
            raise HTTPException(status_code=500, detail="Chat prompt configuration missing.")
        prompt = prompt.format(
            task=sublesson.task,
            solution=sublesson.solution,
            submission=submission, 
        )
        print(f"Chat Prompt: {prompt}")
        async with client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": "Bearer " + OPENAI_API_KEY},
            json={
                "model": "gpt-4o-mini",  # Ensure this is a valid model
                "messages": [
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    *[message.dict() for message in messages]
                ],
            },
        ) as resp:
            if not resp.ok:
                error_detail = await resp.text()
                print(f"OpenAI API Error [{resp.status}]: {error_detail}")
                raise HTTPException(status_code=500, detail="OpenAI service unavailable.")
            resp_json = await resp.json()
            print(f"OpenAI Response: {resp_json['choices'][0]['message']}")
            return resp_json["choices"][0]["message"]
    except HTTPException as e:
        print(f"HTTP Exception in Chat: {e.detail}")
        raise e
    except Exception as e:
        print(f"Unexpected Error in Chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# Update the /tts endpoint
@app.post("/tts", response_model=AudioResponse)
async def convert_text_to_speech(tts_request: TTSRequest, db: Session = Depends(get_db)):
    try:
        audio_url = await text_to_speech(tts_request.text, tts_request.language, tts_request.voice_name)
        return AudioResponse(audio_url=audio_url)
    except HTTPException as e:
        print(f"TTS Service Error: {e.detail}")
        raise e
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# Function to create sample data
def create_sample_data(db: Session):
    # Check if sample data already exists
    if db.query(LessonPlanORM).count() == 0:
        print("Creating sample lesson plans...")
        db.add(LessonPlanORM(
            title="LaTeX 101",
            author="Vikriti Lokegaonkar",
            sublessons=[
                LessonORM(
                    title="Introduction to LaTeX",
                    lesson_text="<h1>Welcome to your first Markdown and LaTeX tutorial.</h1>\n\n<p>LaTeX (pronounced Lah-Tek) is a typesetting engine used to create beautifully formatted documents.</p>\n\n<p>In this lesson, we will cover the basics of setting up a LaTeX document structure and using Markdown-style headings for top-level and sub-level sections.</p>\n\n<p>A basic LaTeX document contains commands that define its structure. Start by setting the document class and adding content inside the <code>document</code> environment:</p>\n\n<pre>\n\\documentclass{article}\n\\begin{document}\nHello, LaTeX World!\n\\end{document}\n</pre>\n\n<p>Now create a document with the structure shown above and include two headings: a top-level heading and a level 3 heading. Use the syntax below:</p>\n\n<pre>\n# This is a top-level heading\n\n### This is a third level heading.\n</pre>",
                    solution="\\documentclass{article}\n\\begin{document}\n# Hello\n\n### World\n\\end{document}",
                    task="Refer to LaTeX documentation.",
                    lesson_type=LessonTypeEnum.latex,
                    solution_boilerplate="\\documentclass{article}"
                ),
                LessonORM(
                    title="Formatting Text and Creating Lists",
                    lesson_text="<h1>Text Formatting and Lists</h1>\n\n<p>LaTeX offers commands to format text with bold, italics, and underline:</p>\n\n<pre>\n\\textbf{bold text}\n\\textit{italic text}\n\\underline{underlined text}\n</pre>\n\n<p>It also supports creating unordered and ordered lists. Use <code>itemize</code> for an unordered list and <code>enumerate</code> for an ordered list:</p>\n\n<pre>\n\\begin{itemize}\n  \\item First item\n  \\item Second item\n\\end{itemize>\n\n\\begin{enumerate}\n  \\item First item\n  \\item Second item\n\\end{enumerate>\n</pre>\n\n<p>Create a document with one sentence that includes bold, italic, and underlined text, as well as both an unordered and ordered list.</p>",
                    solution="\\documentclass{article}\n\\begin{document}\nThis is \\textbf{bold}, \\textit{italic}, and \\underline{underlined} text.\n\n\\begin{itemize}\n  \\item First item\n  \\item Second item\n\\end{itemize}\n\n\\begin{enumerate}\n  \\item First item\n  \\item Second item\n\\end{enumerate}\n\\end{document}",
                    task="Refer to LaTeX documentation.",
                    lesson_type=LessonTypeEnum.latex,
                    solution_boilerplate="\\documentclass{article}"
                ),
                LessonORM(
                    title="Math Equations",
                    lesson_text="<h1>Inserting Math Equations</h1>\n\n<p>LaTeX is widely used for creating complex mathematical equations. You can include inline equations with <code>$...$</code> and displayed equations with <code>\\[ ... \\]</code>:</p>\n\n<pre>\nThis is an inline equation: $E = mc^2$.\n\n\\[\n\\int_{a}^{b} x^2 dx\n\\]</pre>\n\n<p>Create a document with an inline equation for the Pythagorean theorem and a displayed integral from 0 to infinity.</p>",
                    solution="\\documentclass{article}\n\\begin{document}\nThis is the inline equation for the Pythagorean theorem: $a^2 + b^2 = c^2$.\n\nDisplayed integral:\n\n\\[\n\\int_{0}^{\\infty} e^{-x} dx\n\\]\n\\end{document}",
                    task="Refer to LaTeX math documentation.",
                    lesson_type=LessonTypeEnum.latex,
                    solution_boilerplate="\\documentclass{article}"
                ),
                LessonORM(
                    title="Tables in LaTeX",
                    lesson_text="<h1>Creating Tables</h1>\n\n<p>LaTeX allows you to create tables using the <code>tabular</code> environment:</p>\n\n<pre>\n\\begin{tabular}{|c|c|}\n\\hline\nColumn 1 & Column 2 \\\\\n\\hline\nData 1 & Data 2 \\\\\n\\hline\n\\end{tabular}</pre>\n\n<p>Create a document with a 2x2 table where the headers are <em>Name</em> and <em>Age</em>, and add two rows with names and ages.</p>",
                    solution="\\documentclass{article}\n\\begin{document}\n\\begin{tabular}{|c|c|}\n\\hline\nName & Age \\\\\n\\hline\nAlice & 25 \\\\\nBob & 30 \\\\\n\\hline\n\\end{tabular}\n\\end{document}",
                    task="Refer to LaTeX table documentation.",
                    lesson_type=LessonTypeEnum.latex,
                    solution_boilerplate="\\documentclass{article}"
                ),
        ]))

        db.add(LessonPlanORM(
            title="Sample Linux Plan",
            author="Joe Smith",
            sublessons=[
                LessonORM(
                    title="Getting started with Linux",
                    lesson_text="Basic Linux Commands",
                    solution="Use commands like ls, cd, mkdir.",
                    task="Refer to Linux command manuals.",
                    lesson_type=LessonTypeEnum.linux,
                    solution_boilerplate="#!/bin/bash"
                )
        ]))

        db.commit()
        print("Added sample LessonPlans.")

# Event handler to create sample data on startup
@app.on_event("startup")
def startup_event():
    db = SessionLocal()
    create_sample_data(db)
    db.close()

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

