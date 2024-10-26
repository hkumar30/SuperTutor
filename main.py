from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Literal
import uvicorn
import os
import subprocess
import random
import string
from sqlalchemy import (
    create_engine, Column, Integer, String, ForeignKey, Enum
)
from sqlalchemy.orm import sessionmaker, relationship, declarative_base, Session
import enum

app = FastAPI()

# Create /static directory if it doesnt exist.
# Then mount it to /static to be hosted
# This will store the output images
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
        yield db
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
    solution_information = Column(String, nullable=False)
    lesson_type = Column(Enum(LessonTypeEnum), nullable=False)
    solution_boilerplate = Column(String, nullable=False)

class LessonPlanORM(Base):
    __tablename__ = 'lesson_plans'
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    author = Column(String, nullable=False)
    
    # Establish relationship with LessonORM
    lessons = relationship("LessonORM", backref="lesson_plan", cascade="all, delete-orphan")

# Create all tables in the database
Base.metadata.create_all(bind=engine)

class Lesson(BaseModel):
    title: str
    lesson_text: str
    solution: str
    solution_information: str
    lesson_type: Literal["latex", "linux"]
    solution_boilerplate: str

class LessonPlan(BaseModel):
    id: int
    title: str
    author: str
    lessons: list[Lesson]

    @classmethod
    def get(cls, db, lesson_id: int):
        lesson_plan_orm = db.query(LessonPlanORM).filter(LessonPlanORM.id == lesson_id).first()
        
        if not lesson_plan_orm:
            return None

        return cls(
            title=lesson_plan_orm.title,
            author=lesson_plan_orm.author,
            lessons=[
                Lesson(
                    title=lesson.title,
                    lesson_text=lesson.lesson_text,
                    solution=lesson.solution,
                    solution_information=lesson.solution_information,
                    lesson_type=lesson.lesson_type.value,
                    solution_boilerplate=lesson.solution_boilerplate
                )
                for lesson in lesson_plan_orm.lessons
            ]
        )

class SubmissionResult(BaseModel):
    success: bool
    output_url: str | None = None

@app.get("/lesson_plan")
def lesson(lesson_id: int, db: Session = Depends(get_db)):
    lesson_plan = LessonPlan.get(db, lesson_id)
    if not lesson_plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"LessonPlan with id {lesson_id} not found."
        )
    return lesson_plan


@app.post("/check_submission")
def check_submission(lesson_id: int, sublesson_id: int, submission: str) -> SubmissionResult:
    filename = "".join(random.choices(string.ascii_letters, k=8)) + ".pdf"
    pandoc_process = subprocess.run(['pandoc', '-o', f'static/{filename}'], input=submission.encode())
    #stdoutdata, stderrdata = pandoc_process.communicate(submission)
    if pandoc_process.returncode == 0:
        return SubmissionResult(success=True, output_url="http://localhost:8000/static/" + filename)
    else:
        return SubmissionResult(success=False)

def create_sample_data(db: Session):
    # Check if sample data already exists
    if db.query(LessonPlanORM).count() == 0:
        db.add(LessonPlanORM(
            title="Sample LaTeX Plan #1",
            author="John Smith",
            lessons=[
                LessonORM(
                    title="Lesson 1",
                    lesson_text="Introduction to LaTeX",
                    solution="Use LaTeX to format documents.",
                    solution_information="Refer to LaTeX documentation.",
                    lesson_type=LessonTypeEnum.latex,
                    solution_boilerplate="\\documentclass{article}" #Was not sure what to put here, we can change it later.
                )
        ]))

        db.add(LessonPlanORM(
            title="Sample Linux Plan",
            author="Joe Smith",
            lessons=[
                LessonORM(
                    title="Lesson 2",
                    lesson_text="Basic Linux Commands",
                    solution="Use commands like ls, cd, mkdir.",
                    solution_information="Refer to Linux command manuals.",
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