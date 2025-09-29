from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
import random
import os
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite's default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
# Set your API key as environment variable: export GEMINI_API_KEY="your-api-key"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set")


client = genai.Client()


class TopicResponse(BaseModel):
    topic: str
    bullet_points: list[str]
    prep_time: int
    speaking_time: int


@app.get("/generate-topic/{prep_time}/{speaking_time}", response_model=TopicResponse)
async def generate_topic(prep_time: int, speaking_time: int):
    """
    Generate a speech topic with bullet points based on preparation and speaking time.

    Args:
        prep_time: Time available for preparation in seconds
        speaking_time: Time available for speaking in seconds

    Returns:
        TopicResponse with topic and bullet points
    """

    if prep_time < 0 or speaking_time < 0:
        raise HTTPException(status_code=400, detail="Time values must be positive")

    # Determine number of bullet points based on speaking time
    if speaking_time <= 90:
        num_bullets = "3-4"
    else:
        num_bullets = "5-6"

    random_seed = random.randint(1, 1000000)
    categories = [
        "technology",
        "psychology",
        "nature",
        "society",
        "science",
        "arts",
        "philosophy",
        "history",
        "culture",
        "sports",
        "health",
        "education",
    ]
    random_category = random.choice(categories)

    # Create prompt for Gemini
    prompt = f"""
[ID: {random_seed}] Generate a UNIQUE and CREATIVE speech topic suitable for someone with {prep_time} seconds to prepare and {speaking_time} seconds to speak.
Be creative and generate something different each time. Vary the topics across different categories in {random_category}. However, the topic should be such that
the speaker should be able to easily talk about it for {speaking_time}. It shouldn't require niche general knowledge, we are testing speaker's impromptu ability
over actual facts and history.

The topic should be:
- Appropriate for the given time constraints
- Interesting and engaging
- Not too complex if prep time is short
- More in-depth if prep time is longer

Provide the response in the following format:
TOPIC: [Your topic here]
BULLETS:
- [Bullet point 1]
- [Bullet point 2]
- [Bullet point 3]
{"- [Bullet point 4]" if speaking_time > 90 else ""}
{"- [Bullet point 5]" if speaking_time > 90 else ""}
{"- [Bullet point 6]" if speaking_time > 120 else ""}

Generate exactly {num_bullets} bullet points that are:
- Key points to cover in the speech
- Concise and clear
- Helpful for structuring the speech within {speaking_time} seconds
- Do not use "**..**" like bold, italics or anything. Return text with no formatting at all always.
"""

    try:
        # Generate content using Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=2,
                top_p=0.95,
                top_k=40,
            ),
        )
        response_text = response.text

        # Parse the response
        lines = response_text.strip().split("\n")
        topic = ""
        bullet_points = []

        for line in lines:
            line = line.strip()
            if line.startswith("TOPIC:"):
                topic = line.replace("TOPIC:", "").strip()
            elif line.startswith("-") or line.startswith("•"):
                bullet = line.lstrip("-•").strip()
                if bullet:
                    bullet_points.append(bullet)

        # Fallback parsing if structured format not followed
        if not topic:
            # Try to extract topic from first substantial line
            for line in lines:
                if line and not line.startswith("-") and not line.startswith("•"):
                    topic = line.strip()
                    break

        if not bullet_points:
            # Extract any lines that look like bullet points
            bullet_points = [
                line.lstrip("-•").strip()
                for line in lines
                if line.strip().startswith(("-", "•"))
            ]

        if not topic:
            topic = "Public Speaking Topic"

        if not bullet_points:
            raise ValueError("Failed to extract bullet points from response")

        return TopicResponse(
            topic=topic,
            bullet_points=bullet_points,
            prep_time=prep_time,
            speaking_time=speaking_time,
        )

    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=f"Error generating topic: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "Speech Topic Generator API",
        "usage": "GET /generate-topic/{prep_time}/{speaking_time}",
        "example": "/generate-topic/60/120",
    }
