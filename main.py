from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Literal
import uvicorn
import os
import subprocess
import random
import string

app = FastAPI()

# Create /static directory if it doesnt exist.
# Then mount it to /static to be hosted
# This will store the output images
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

db = None # Init DB Here

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
        # DATABASE CODE HERE

        return cls(
            id=0,
            title="Test Lesson",
            author="John Smith",
            lessons=[
                Lesson(title="Lesson 1", lesson_text="Ah", solution="AHH", lesson_type="latex", solution_information="HAHA",solution_boilerplate="Meow")
            ]
        )

class SubmissionResult(BaseModel):
    success: bool
    output_url: str | None = None

@app.get("/lesson_plan")
def lesson(lesson_id: int) -> LessonPlan:
    return LessonPlan.get(db, lesson_id)


@app.post("/check_submission")
def check_submission(lesson_id: int, sublesson_id: int, submission: str) -> SubmissionResult:
    filename = "".join(random.choices(string.ascii_letters, k=8)) + ".pdf"
    pandoc_process = subprocess.run(['pandoc', '-o', f'static/{filename}'], input=submission.encode())
    #stdoutdata, stderrdata = pandoc_process.communicate(submission)
    if pandoc_process.returncode == 0:
        return SubmissionResult(success=True, output_url="http://localhost:8000/static/" + filename)
    else:
        return SubmissionResult(success=False)

if __name__ == "__main__":

    # Start FastAPI app.
    uvicorn.run(app, host="0.0.0.0", port=8000)