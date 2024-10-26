from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Literal
import uvicorn
import os
import subprocess
import random
import string
from dotenv import load_dotenv
import aiohttp
import re

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise Exception("OPENAI_API_KEY not defined. Check your .env file, buddy.")

prompts = {}

with open("prompts/latex_prompt", "r") as f:
    prompts["latex"] = f.read()

app = FastAPI()

http_client = None
async def get_client():
    global http_client
    if http_client is None:
        http_client = aiohttp.ClientSession()
    return http_client

# Create /static directory if it doesnt exist.
# Then mount it to /static to be hosted
# This will store the output images
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

db = None  # Init DB Here


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
                Lesson(
                    title="Lesson 1",
                    lesson_text="Write the Quadratic Formula using \frac{}",
                    solution="$\frac{-b \pm \sqrt{b^2 -4ac}}{2a}$",
                    lesson_type="latex",
                    solution_information="Write the quadratic formula ",
                    solution_boilerplate="$\text{Put your solution here}$",
                )
            ],
        )


class SubmissionResult(BaseModel):
    success: bool
    problem: str | None = None
    hint: str | None = None
    output_url: str | None = None


@app.get("/lesson_plan")
def lesson(lesson_id: int) -> LessonPlan:
    return LessonPlan.get(db, lesson_id)


@app.post("/check_submission")
async def check_submission(
    lesson_id: int, sublesson_id: int, submission: str
) -> SubmissionResult | None:

    lesson_plan = LessonPlan.get(db, lesson_id=lesson_id)
    sublesson = lesson_plan.lessons[sublesson_id]

    filename = "".join(random.choices(string.ascii_letters, k=8)) + ".pdf"
    pandoc_process = subprocess.run(
        ["pandoc", "-o", f"static/{filename}"], input=submission.encode()
    )
    # stdoutdata, stderrdata = pandoc_process.communicate(submission)
    if pandoc_process.returncode != 0:
        return SubmissionResult(success=False, problem="Doesn't compile")

    prompt = prompts[sublesson.lesson_type].format(
        solution_information=sublesson.solution_information,
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
                    "content": "You are a helpful teaching assistant who is helping the professor grade student submissionms.",
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
                success=True, output_url="http://localhost:8000/static/" + filename, hint=hint
            )
        else:
            return SubmissionResult(
                success=False,
                problem="Comples but incorrect latex.",
                hint=hint,
                output_url="http://localhost:8000/static/" + filename,
            )


if __name__ == "__main__":

    # Start FastAPI app.
    uvicorn.run(app, host="0.0.0.0", port=8000)
