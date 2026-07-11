"""
Adaptive Tutor Agent
---------------------
Replaces the old rule-based decide_next_step() + random question sampling
with an LLM-driven agent that reasons over a student's history and chooses
its own next action.

Entry point: run_tutor_agent(user_id, lesson_id, score, quiz_type, topic_accuracy)
Called from quiz_service.submit_quiz() in place of decide_next_step().
"""

import os
import json
from datetime import datetime

import google.generativeai as genai

from db import profiles_col, question_bank_col, agent_logs_col

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY missing in .env")

genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.0-flash"

VALID_DECISIONS = {"NEXT_LESSON", "GO_SIMPLIFIED_QUIZ", "TARGETED_REMEDIATION", "SHOW_SUPPORT_OPTIONS"}


# ---------------------------------------------------------------------------
# Tools the agent can call. Each one wraps existing collections/queries you
# already have — nothing new to stand up.
# ---------------------------------------------------------------------------

def get_learner_profile(user_id: str) -> dict:
    """Tool: returns the student's weak topics, completed lessons, and recent history."""
    profile = profiles_col.find_one({"user_id": user_id}) or {}
    return {
        "weak_topics": profile.get("weak_topics", {}),
        "completed_lessons": profile.get("completed_lessons", []),
        "recent_history": profile.get("history", [])[-5:],  # last 5 attempts is enough context
    }


def fetch_targeted_questions(topic: str, difficulty: str, count: int = 4) -> dict:
    """Tool: pulls questions for one specific topic/difficulty, instead of a random mixed sample."""
    questions = list(question_bank_col.aggregate([
        {"$match": {"topic": topic, "difficulty": difficulty}},
        {"$sample": {"size": count}}
    ]))
    return {
        "topic": topic,
        "difficulty": difficulty,
        "question_ids": [str(q["_id"]) for q in questions],
        "count_found": len(questions),
    }


def generate_hint(question_text: str, wrong_answer: str, topic: str) -> dict:
    """Tool: asks Gemini for a short, student-friendly explanation of the mistake."""
    prompt = (
        f"A student learning English grammar (topic: {topic}) answered "
        f"'{wrong_answer}' to this question: \"{question_text}\". "
        f"In 1-2 short sentences, explain the mistake and the correct rule. "
        f"Keep it simple and encouraging, no jargon."
    )
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return {"hint": response.text.strip()}


def log_decision(user_id: str, lesson_id: str, decision: str, reasoning: str, tools_used: list) -> dict:
    """Tool: writes the agent's decision + reasoning trace so it's visible on the teacher dashboard."""
    agent_logs_col.insert_one({
        "user_id": user_id,
        "lesson_id": lesson_id,
        "decision": decision,
        "reasoning": reasoning,
        "tools_used": tools_used,
        "created_at": datetime.utcnow(),
    })
    return {"logged": True}


# ---------------------------------------------------------------------------
# Tool schema Gemini needs to know what it can call and with what arguments.
# ---------------------------------------------------------------------------

TOOL_DECLARATIONS = [
    {
        "name": "get_learner_profile",
        "description": "Get the student's weak topics, completed lessons, and recent attempt history.",
        "parameters": {
            "type": "object",
            "properties": {"user_id": {"type": "string"}},
            "required": ["user_id"],
        },
    },
    {
        "name": "fetch_targeted_questions",
        "description": "Fetch questions for one specific topic and difficulty, for targeted remediation.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"},
                "difficulty": {"type": "string", "enum": ["basic", "moderate", "advanced"]},
                "count": {"type": "integer"},
            },
            "required": ["topic", "difficulty"],
        },
    },
    {
        "name": "generate_hint",
        "description": "Generate a short personalised explanation for a wrong answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "question_text": {"type": "string"},
                "wrong_answer": {"type": "string"},
                "topic": {"type": "string"},
            },
            "required": ["question_text", "wrong_answer", "topic"],
        },
    },
]

TOOL_FUNCTIONS = {
    "get_learner_profile": get_learner_profile,
    "fetch_targeted_questions": fetch_targeted_questions,
    "generate_hint": generate_hint,
}


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an adaptive tutor agent for an English grammar learning platform.

After each quiz attempt, you decide what the student should do next. You have tools to look up
the student's history and pull targeted content — use them before deciding, don't guess.

Reasoning process:
1. Call get_learner_profile to see the student's pattern across past attempts, not just this one score.
2. If a topic has been weak repeatedly (not just this attempt), prefer TARGETED_REMEDIATION over a
   generic simplified quiz, and call fetch_targeted_questions for that specific topic.
3. If the student is doing well and this is a one-off dip, GO_SIMPLIFIED_QUIZ is enough.
4. If score is strong (>=70) with no recurring weak topic, decide NEXT_LESSON.
5. If the student seems stuck even after remediation attempts, decide SHOW_SUPPORT_OPTIONS.

When you have enough information, respond with ONLY a JSON object (no markdown, no backticks):
{
  "decision": "NEXT_LESSON" | "GO_SIMPLIFIED_QUIZ" | "TARGETED_REMEDIATION" | "SHOW_SUPPORT_OPTIONS",
  "reasoning": "1-3 sentences explaining why, referencing the specific pattern you saw",
  "target_topic": "topic name if decision is TARGETED_REMEDIATION, else null"
}
"""


def run_tutor_agent(user_id: str, lesson_id: str, score: int, quiz_type: str, topic_accuracy: dict) -> dict:
    """
    Main entry point. Call this instead of decide_next_step() from quiz_service.submit_quiz().

    Returns:
        {
            "decision": str,          # same values your frontend already expects, plus TARGETED_REMEDIATION
            "reasoning": str,
            "target_topic": str | None,
            "remediation_question_ids": list[str]   # populated only for TARGETED_REMEDIATION
        }
    """
    model = genai.GenerativeModel(
        MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
        tools=[{"function_declarations": TOOL_DECLARATIONS}],
    )

    chat = model.start_chat()
    tools_used = []

    initial_message = (
        f"user_id: {user_id}\n"
        f"lesson_id: {lesson_id}\n"
        f"this attempt's score: {score}%\n"
        f"quiz_type: {quiz_type}\n"
        f"this attempt's topic_accuracy: {json.dumps(topic_accuracy)}\n\n"
        f"Decide the next step for this student."
    )

    response = chat.send_message(initial_message)

    # Agent tool-calling loop — max 5 hops so a confused model can't loop forever
    for _ in range(5):
        function_call = _extract_function_call(response)
        if not function_call:
            break

        tool_name = function_call.name
        tool_args = dict(function_call.args)
        tools_used.append(tool_name)

        tool_fn = TOOL_FUNCTIONS.get(tool_name)
        if not tool_fn:
            tool_result = {"error": f"unknown tool {tool_name}"}
        else:
            # user_id/lesson_id aren't always passed back by the model reliably —
            # patch them in from the known context rather than trusting the call.
            if tool_name == "get_learner_profile":
                tool_args["user_id"] = user_id
            tool_result = tool_fn(**tool_args)

        response = chat.send_message(
            genai.protos.Content(
                parts=[genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response={"result": tool_result},
                    )
                )]
            )
        )

    parsed = _parse_final_decision(response.text)

    remediation_question_ids = []
    if parsed["decision"] == "TARGETED_REMEDIATION" and parsed.get("target_topic"):
        result = fetch_targeted_questions(parsed["target_topic"], "basic", count=4)
        remediation_question_ids = result["question_ids"]
        tools_used.append("fetch_targeted_questions")

    log_decision(
        user_id=user_id,
        lesson_id=lesson_id,
        decision=parsed["decision"],
        reasoning=parsed["reasoning"],
        tools_used=tools_used,
    )

    return {
        "decision": parsed["decision"],
        "reasoning": parsed["reasoning"],
        "target_topic": parsed.get("target_topic"),
        "remediation_question_ids": remediation_question_ids,
    }


def _extract_function_call(response):
    try:
        part = response.candidates[0].content.parts[0]
        return part.function_call if part.function_call.name else None
    except (IndexError, AttributeError):
        return None


def _parse_final_decision(text: str) -> dict:
    """Parses the agent's final JSON reply, with a safe fallback if it goes off-script."""
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(cleaned)
        if parsed.get("decision") in VALID_DECISIONS:
            return parsed
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: never let a malformed agent response break the quiz flow.
    return {
        "decision": "GO_SIMPLIFIED_QUIZ",
        "reasoning": "Agent response could not be parsed; defaulted to a safe fallback.",
        "target_topic": None,
    }
