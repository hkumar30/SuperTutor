from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from typing import Literal, List
import uvicorn
import os
import subprocess
import random
import string
from dotenv import load_dotenv
import aiohttp
import re
from sqlalchemy import (
    create_engine, Column, Integer, String, ForeignKey, Enum
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, Session
import enum
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL")
if OPENAI_API_KEY is None:
    raise Exception("Yo buddy, OPENAI_API_KEY not defined. Check your .env file.")
if BASE_URL is None:
    BASE_URL = "http://localhost:8000"

prompts = {}

with open("prompts/latex_prompt", "r") as f:
    prompts["latex"] = f.read()
with open("prompts/latex_chat_prompt", "r") as f:
    prompts["latex_chat"] = f.read()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=["*"],
)

http_client = None
async def get_client():
    global http_client
    if http_client is None:
        http_client = aiohttp.ClientSession()
    return http_client

# Create /static directory if it does not exist.
# Then mount it to /static to be hosted
# This will store the output images
os.makedirs("database", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

DATABASE_URL = "sqlite:///./database/lesson_plans.db"

# Create the SQLAlchemy engine
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Base class for declarative class definitions
Base = declarative_base()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

class LessonTypeEnum(enum.Enum):
    latex = "latex"
    linux = "linux"

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
    title = Column(String, nullable=False)
    author = Column(String, nullable=False)
    
    # Establish relationship with LessonORM
    sublessons = relationship("LessonORM", backref="lesson_plan", cascade="all, delete-orphan")

# Create all tables in the database
Base.metadata.create_all(bind=engine)

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
    def get(cls, db, lesson_id: int):
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

# Schema (POST for creating lessons)
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

@app.get("/lessons/")
def get_lesson_plans(db: Session = Depends(get_db)):
    lesson_plans = db.query(LessonPlanORM).all()
    return [{"id": lp.id, "title": lp.title, "author": lp.author} for lp in lesson_plans]

@app.post("/lessons/")
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
        return {"message": "Lesson created successfully!", "lesson_plan_id": new_lesson_plan.id}
    except ValidationError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Validation error: {e.errors()}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/lessons/{lesson_id}")
def lesson(lesson_id: int, db: Session = Depends(get_db)):
    lesson_plan = LessonPlan.get(db, lesson_id)
    if not lesson_plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"LessonPlan with id {lesson_id} not found."
        )
    return lesson_plan

@app.post("/lessons/{lesson_id}/{sublesson_id}/submit")
async def check_submission(
    lesson_id: int, sublesson_id: int, submission: str, check_solution: bool, db: Session = Depends(get_db)
) -> SubmissionResult | None:
    lesson_plan = LessonPlan.get(db, lesson_id=lesson_id)
    sublesson = lesson_plan.sublessons[sublesson_id]

    filename = "".join(random.choices(string.ascii_letters, k=8)) + ".html"
    pandoc_process = subprocess.run(
        #["pandoc", "-V", 'geometry:papersize={5in,2.7in},margin=0.1cm', "-o", f"static/{filename}"], input=submission.encode()
        ["pandoc", "--mathml"], input=submission.encode(),
        capture_output=True
    )
    # stdoutdata, stderrdata = pandoc_process.communicate(submission)
    if pandoc_process.returncode != 0:
        return SubmissionResult(success=False, problem="Doesn't compile")

    output_url =f"{BASE_URL}/static/{filename}"
    output = pandoc_process.stdout

    if not check_solution:
        return SubmissionResult(success=False, problem="Not Checking Solution", output=output)

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
            "model": "gpt-4o-mini",
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
            print(await resp.json())
            return None
            # raise Exception("Well Damn... OpenAI request failed.")

        resp_json = await resp.json()
        response = resp_json["choices"][0]["message"]["content"]
        matches = re.match(r".*[Pp]ass:\s*(.*)\s*[Hh]int:\s*([\s\S]*)", response)
        if matches is None:
            return None
        success, hint = matches[1], matches[2]
        print(success, hint)
        if success.strip().lower() in ["yes", "success"]:
            return SubmissionResult(
                success=True, output_url=output_url, hint=hint
            )
        else:
            return SubmissionResult(
                success=False,
                problem="Compiles but incorrect latex.",
                hint=hint,
                output_url=output_url,
            )

@app.post("/lessons/{lesson_id}/{sublesson_id}/chat")
async def chat(
    lesson_id: int, sublesson_id: int, submission: str, messages: list[LessonChatMessage], db: Session = Depends(get_db) 
) -> LessonChatMessage | None:
    lesson_plan = LessonPlan.get(db, lesson_id=lesson_id)
    sublesson = lesson_plan.sublessons[sublesson_id]
    client = await get_client()
    prompt = prompts[sublesson.lesson_type + "_chat"].format(
                               task=sublesson.task,
        solution=sublesson.solution,
        submission=submission, 
                    )
    print(prompt)
    async with client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": "Bearer " + OPENAI_API_KEY},
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": prompt,
                },
                *[message.model_dump() for message in messages]
            ],
        },
    ) as resp:
        if not resp.ok:
            print(await resp.json())
            return None
        resp_json = await resp.json()
        print(resp_json["choices"][0]["message"])
        return resp_json["choices"][0]["message"]

def create_sample_data(db: Session):
    # Check if sample data already exists
    if db.query(LessonPlanORM).count() == 0:
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
                    solution_boilerplate="\\documentclass{article}" #Was not sure what to put here, we can change it later.
                ),
                LessonORM(
                    title="Formatting Text and Creating Lists",
                    lesson_text="<h1>Text Formatting and Lists</h1>\n\n<p>LaTeX offers commands to format text with bold, italics, and underline:</p>\n\n<pre>\n\\textbf{bold text}\n\\textit{italic text}\n\\underline{underlined text}\n</pre>\n\n<p>It also supports creating unordered and ordered lists. Use <code>itemize</code> for an unordered list and <code>enumerate</code> for an ordered list:</p>\n\n<pre>\n\\begin{itemize}\n  \\item First item\n  \\item Second item\n\\end{itemize>\n\n\\begin{enumerate}\n  \\item First item\n  \\item Second item\n\\end{enumerate>\n</pre>\n\n<p>Create a document with one sentence that includes bold, italic, and underlined text, as well as both an unordered and ordered list.</p>",
                    solution="\\documentclass{article}\n\\begin{document}\nThis is \\textbf{bold}, \\textit{italic}, and \\underline{underlined} text.\n\n\\begin{itemize}\n  \\item First item\n  \\item Second item\n\\end{itemize}\n\n\\begin{enumerate}\n  \\item First item\n  \\item Second item\n\\end{enumerate}\n\\end{document}",
                    task="Refer to LaTeX documentation.",
                    lesson_type=LessonTypeEnum.latex,
                    solution_boilerplate="\\documentclass{article}" #Was not sure what to put here, we can change it later.
                ),
                LessonORM(
                    title="Math Equations",
                    lesson_text="<h1>Inserting Math Equations</h1>\n\n<p>LaTeX is widely used for creating complex mathematical equations. You can include inline equations with <code>$...$</code> and displayed equations with <code>\\[ ... \\]</code>:</p>\n\n<pre>\nThis is an inline equation: $E = mc^2$.\n\n\\[\n\\int_{a}^{b} x^2 dx\n\\]</pre>\n\n<p>Create a document with an inline equation for the Pythagorean theorem and a displayed integral from 0 to infinity.</p>",
                    solution="\\documentclass{article}\n\\begin{document}\nThis is the inline equation for the Pythagorean theorem: $a^2 + b^2 = c^2$.\n\nDisplayed integral:\n\n\\[\n\\int_{0}^{\\infty} e^{-x} dx\n\\]\n\\end{document}",
                    solution_information="Refer to LaTeX math documentation.",
                    lesson_type=LessonTypeEnum.latex,
                    solution_boilerplate="\\documentclass{article}"
                ),
                LessonORM(
                    title="Tables in LaTeX",
                    lesson_text="<h1>Creating Tables</h1>\n\n<p>LaTeX allows you to create tables using the <code>tabular</code> environment:</p>\n\n<pre>\n\\begin{tabular}{|c|c|}\n\\hline\nColumn 1 & Column 2 \\\\\n\\hline\nData 1 & Data 2 \\\\\n\\hline\n\\end{tabular}</pre>\n\n<p>Create a document with a 2x2 table where the headers are <em>Name</em> and <em>Age</em>, and add two rows with names and ages.</p>",
                    solution="\\documentclass{article}\n\\begin{document}\n\\begin{tabular}{|c|c|}\n\\hline\nName & Age \\\\\n\\hline\nAlice & 25 \\\\\nBob & 30 \\\\\n\\hline\n\\end{tabular}\n\\end{document}",
                    solution_information="Refer to LaTeX table documentation.",
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
        # db.refresh()
        print(f"Added LessonPlan with ID: {id}")

@app.on_event("startup")
def startup_event():
    db = SessionLocal()
    create_sample_data(db)
    db.close()

if __name__ == "__main__":
    # Start FastAPI app.
    uvicorn.run(app, host="0.0.0.0", port=8000)


