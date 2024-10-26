from fastapi import FastAPI
from typing import List
import uvicorn

app = FastAPI()

class Lesson:
    pass

class SubmissionResult:
    pass

@app.get("/lesson")
def lessons() -> Lesson:
    pass


@app.post("/check_submission")
def check_submission(lesson_id: int, sublesson_id: int, submission: str) -> SubmissionResult:
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)