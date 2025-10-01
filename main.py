from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
import random
import os
from fastapi.middleware.cors import CORSMiddleware
import re
import math
from typing import List, Dict, Any
from functools import lru_cache
import spacy
import language_tool_python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat

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


# Request/Response models
class AnalyzeRequest(BaseModel):
    transcript: str
    duration: float  # total audio duration in seconds (optional but passed)
    speakTime: float  # speaking time (seconds) - front-end already sends this


class GrammarIssue(BaseModel):
    message: str
    offset: int
    length: int
    replacements: List[str]
    ruleId: str
    context: str


class AnalyzeResponse(BaseModel):
    grammar_issues: List[Dict[str, Any]]
    keywords: List[str]
    keyword_scores: Dict[str, float]
    topic_string: str
    semantic_similarity: float
    vocabulary_richness: float
    sentence_metrics: Dict[str, Any]
    sentiment: Dict[str, float]
    extras: Dict[str, Any] = {}


# --- Lazy loads / caching for heavy models --- #
@lru_cache(maxsize=1)
def load_spacy():
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        # If model not installed, raise an informative error
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. Install it with:\n"
            "python -m pip install 'spacy' && python -m spacy download en_core_web_sm"
        )
    return nlp


@lru_cache(maxsize=1)
def load_langtool():
    # language_tool_python runs a local Java process. Ensure Java is installed.
    # Set a persistent directory
    os.environ["LANGTOOL_HOME"] = "/home/raj/.languagetool"

    tool = language_tool_python.LanguageTool("en-US")
    return tool


@lru_cache(maxsize=1)
def load_sentence_transformer():
    # small but good multipurpose model. You can change to a different model name via
    # env var SENT_TRANSFORMER if you prefer.
    model_name = os.getenv("SENT_TRANSFORMER", "sentence-transformers/all-MiniLM-L6-v2")
    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def load_sentiment():
    return SentimentIntensityAnalyzer()


# --- Utility functions --- #
def simple_preprocess(text: str) -> str:
    """Normalize whitespace and remove weird control chars."""
    return re.sub(r"\s+", " ", text).strip()


def restore_sentence_boundaries(
    transcript: str, nlp, max_tokens_per_sentence: int = 20
) -> List[str]:
    """
    Transcript may be unpunctuated. spaCy sentence segmentation needs punctuation to be reliable.
    Strategy:
      1. Try spaCy sentence splitting (it helps if there is some punctuation).
      2. If spaCy returns 1 giant sentence (very likely for unpunctuated text), heuristically break
         into pseudo-sentences of ~max_tokens_per_sentence tokens.
    This gives approximate sentence-level metrics that are still useful for averages.
    """
    doc = nlp(transcript)
    sents = list(doc.sents)
    if len(sents) > 1:
        return [simple_preprocess(s.text) for s in sents]

    # fallback: split into chunks of N tokens
    tokens = [t.text for t in doc]
    if len(tokens) == 0:
        return []
    chunks = []
    for i in range(0, len(tokens), max_tokens_per_sentence):
        chunk = " ".join(tokens[i: i + max_tokens_per_sentence])
        chunks.append(simple_preprocess(chunk))
    return chunks


def compute_vocab_richness(transcript: str) -> float:
    tokens = re.findall(r"\b\w+\b", transcript.lower())
    if not tokens:
        return 0.0
    unique = set(tokens)
    return len(unique) / len(tokens)


def avg_sentence_length(sentences: List[str]) -> float:
    if not sentences:
        return 0.0
    lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    return float(sum(lengths)) / len(lengths)


def sentence_complexity_metric(sentences: List[str]) -> Dict[str, float]:
    """
    Approximate complexity:
      - Flesch Reading Ease and Flesch-Kincaid Grade from textstat (suitable for English).
      - average syllables per word (approx).
    """
    joined = " ".join(sentences)
    if not joined.strip():
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "avg_syllables_per_word": 0.0,
        }
    try:
        fre = textstat.flesch_reading_ease(joined)
        fk = textstat.flesch_kincaid_grade(joined)
    except Exception:
        fre = 0.0
        fk = 0.0
    # avg syllables per word
    words = re.findall(r"\b\w+\b", joined)
    if len(words) == 0:
        avg_syll = 0.0
    else:
        avg_syll = sum(textstat.syllable_count(w) for w in words) / len(words)
    return {
        "flesch_reading_ease": fre,
        "flesch_kincaid_grade": fk,
        "avg_syllables_per_word": avg_syll,
    }


def extract_keywords_tfidf(
    corpus_text: str, top_n: int = 8
) -> (List[str], Dict[str, float]):
    """
    Use TF-IDF to pick top unigrams/bigrams that describe the transcript.
    For single-document case, we use TF-IDF over the document itself but with stop words removed;
    we then choose terms with highest TF (TF-IDF degenerates, but it still surfaces frequent non-stop tokens).
    """
    # small preprocessor to keep words and common bigrams
    vectorizer = TfidfVectorizer(
        max_df=1.0,
        min_df=1,
        stop_words="english",
        ngram_range=(1, 2),
        token_pattern=r"\b\w+\b",
        max_features=5000,
    )
    X = vectorizer.fit_transform([corpus_text])
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_scores = X.toarray()[0]
    top_indices = tfidf_scores.argsort()[::-1][:top_n]
    keywords = feature_array[top_indices].tolist()
    scores = {k: float(tfidf_scores[i]) for k, i in zip(keywords, top_indices)}
    return keywords, scores


def calc_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1 is None or vec2 is None:
        return 0.0
    return float(util.cos_sim(vec1, vec2).item())


# --- API endpoint --- #
@app.post("/api/analyze-speech", response_model=AnalyzeResponse)
def analyze_speech(req: AnalyzeRequest):
    transcript = req.transcript or ""
    duration = req.duration or 0.0
    speakTime = req.speakTime or 0.0
    print("Analyze speech hit")

    # basic validation
    if not isinstance(transcript, str) or len(transcript.strip()) == 0:
        raise HTTPException(
            status_code=400, detail="Transcript must be a non-empty string"
        )

    transcript = simple_preprocess(transcript)

    # load models
    nlp = load_spacy()
    langtool = load_langtool()
    sbert = load_sentence_transformer()
    sentiment_analyzer = load_sentiment()

    # ---- Grammar check (LanguageTool) ----
    # Note: LanguageTool returns matches with suggested replacements and offsets.
    # Because the transcript is unpunctuated, LanguageTool may find fewer sentence-level grammar matches,
    # but it still catches many common grammar issues (agreement, missing articles, repeated words, etc).
    try:
        lt_matches = langtool.check(transcript)
    except Exception as e:
        # In some environments the language tool Java server might fail; return graceful fallback.
        lt_matches = []
        print("LanguageTool error:", e)

    grammar_issues = []
    for m in lt_matches:
        grammar_issues.append(
            {
                "message": m.message,
                "offset": m.offset,
                "length": m.errorLength,
                "replacements": m.replacements,
                "ruleId": m.ruleId,
                "context": m.context,
            }
        )

    # ---- Sentence-level handling for metrics ----
    sentences = restore_sentence_boundaries(transcript, nlp, max_tokens_per_sentence=18)

    # ---- Vocabulary richness ----
    vocab_richness = compute_vocab_richness(transcript)

    # ---- Sentence-level metrics ----
    avg_len = avg_sentence_length(sentences)
    complexity = sentence_complexity_metric(sentences)
    sentence_metrics = {
        "num_sentences": len(sentences),
        "avg_sentence_length_words": avg_len,
        "complexity": complexity,
    }

    # ---- Keyword extraction (TF-IDF) ----
    keywords, keyword_scores = extract_keywords_tfidf(transcript, top_n=8)
    topic_string = " ".join(keywords) if keywords else ""

    # ---- Embeddings + semantic similarity ----
    # We'll compute embedding for the transcript and for the topic_string (keywords joined).
    # If you want to compare against an external 'expected topic' string, pass it in the request,
    # or change the code to compute embeddings against a topic set.
    try:
        emb_transcript = sbert.encode(transcript, convert_to_tensor=True)
        emb_topic = sbert.encode(
            topic_string if topic_string else transcript, convert_to_tensor=True
        )
        semantic_similarity = calc_cosine_similarity(emb_transcript, emb_topic)
    except Exception as e:
        print("Embedding error:", e)
        semantic_similarity = 0.0

    # ---- Sentiment (VADER) ----
    sent_scores = sentiment_analyzer.polarity_scores(transcript)

    # ---- Extras: pacing estimate, words per minute ----
    words = re.findall(r"\b\w+\b", transcript)
    num_words = len(words)
    wpm = (num_words / speakTime * 60) if speakTime > 0 else None
    extras = {
        "num_words": num_words,
        "words_per_minute_est": wpm,
        "duration_reported": duration,
        "speakTime_reported": speakTime,
    }

    return {
        "grammar_issues": grammar_issues,
        "keywords": keywords,
        "keyword_scores": keyword_scores,
        "topic_string": topic_string,
        "semantic_similarity": semantic_similarity,
        "vocabulary_richness": vocab_richness,
        "sentence_metrics": sentence_metrics,
        "sentiment": sent_scores,
        "extras": extras,
    }
