from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import uvicorn

app = FastAPI()

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
    pass

@app.get("/lesson_plan")
def lesson(lesson_id: int) -> LessonPlan:
    return LessonPlan.get(db, lesson_id)


@app.post("/check_submission")
def check_submission(lesson_id: int, sublesson_id: int, submission: str) -> SubmissionResult:
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)