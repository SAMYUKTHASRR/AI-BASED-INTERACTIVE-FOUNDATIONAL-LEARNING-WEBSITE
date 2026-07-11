from datetime import datetime
from bson import ObjectId
from db import quizzes_col, attempts_col, question_bank_col
from services.profile_service import update_profile_after_attempt
from services.tutor_agent import run_tutor_agent

WEIGHTS = {
    "basic": 1,
    "moderate": 2,
    "advanced": 3
}


def _get_attempt_no(user_id: str, lesson_id: str) -> int:
    count = attempts_col.count_documents({
        "user_id": user_id,
        "lesson_id": lesson_id
    })
    return count + 1


def _pick_questions(lesson_id: str, quiz_mode: str):
    """
    mixed      -> 2 basic + 2 moderate + 2 advanced
    simplified -> 4 basic + 2 moderate
    """

    if quiz_mode == "simplified":
        plan = [("basic", 4), ("moderate", 2)]
        quiz_type = "simplified"
    else:
        plan = [("basic", 2), ("moderate", 2), ("advanced", 2)]
        quiz_type = "mixed"

    picked = []
    difficulty_breakdown = {
        "basic": 0,
        "moderate": 0,
        "advanced": 0
    }

    for difficulty, count in plan:
        questions = list(question_bank_col.aggregate([
            {
                "$match": {
                    "lesson_id": str(lesson_id),
                    "difficulty": difficulty
                }
            },
            {"$sample": {"size": count}}
        ]))

        picked.extend(questions)
        difficulty_breakdown[difficulty] += len(questions)

    return picked, quiz_type, difficulty_breakdown


def start_quiz(user_id: str, lesson_id: str, quiz_mode: str = "mixed"):
    lesson_id = str(lesson_id)

    attempt_no = _get_attempt_no(user_id, lesson_id)

    questions, quiz_type, difficulty_breakdown = _pick_questions(lesson_id, quiz_mode)

    if not questions:
        raise ValueError("No questions found for this lesson.")

    question_ids = []
    question_payload = []

    for q in questions:
        qid = str(q["_id"])
        question_ids.append(qid)

        question_payload.append({
            "question_id": qid,
            "question": q["question"],
            "options": q["options"],
            "difficulty": q.get("difficulty", "basic"),
            "topic": q.get("topic", "unknown")
        })

    quiz_doc = {
        "user_id": user_id,
        "lesson_id": lesson_id,
        "attempt_no": attempt_no,
        "quiz_type": quiz_type,
        "difficulty_breakdown": difficulty_breakdown,
        "question_ids": question_ids,
        "status": "active",
        "created_at": datetime.utcnow()
    }

    res = quizzes_col.insert_one(quiz_doc)

    return {
        "quiz_id": str(res.inserted_id),
        "lesson_id": lesson_id,
        "attempt_no": attempt_no,
        "quiz_type": quiz_type,
        "difficulty_breakdown": difficulty_breakdown,
        "questions": question_payload
    }


def submit_quiz(user_id: str, quiz_id: str, submitted_answers, time_taken_sec: int = 0):
    quiz = quizzes_col.find_one({"_id": ObjectId(quiz_id)})

    if not quiz:
        raise ValueError("Quiz not found.")

    if quiz["user_id"] != user_id:
        raise PermissionError("This quiz does not belong to you.")

    if quiz.get("status") == "submitted":
        raise PermissionError("Quiz already submitted.")

    lesson_id = quiz["lesson_id"]
    attempt_no = quiz.get("attempt_no", 1)
    quiz_type = quiz.get("quiz_type", "mixed")
    difficulty_breakdown = quiz.get("difficulty_breakdown", {
        "basic": 0,
        "moderate": 0,
        "advanced": 0
    })

    question_ids = [ObjectId(qid) for qid in quiz["question_ids"]]
    questions = list(question_bank_col.find({"_id": {"$in": question_ids}}))

    answers_map = {}

    if isinstance(submitted_answers, list):
        for ans in submitted_answers:
            answers_map[ans["question_id"]] = ans["selected_index"]
    elif isinstance(submitted_answers, dict):
        answers_map = submitted_answers
    else:
        raise ValueError("submitted_answers must be a list or dict")

    total_weight = 0
    earned_weight = 0
    topic_stats = {}

    for q in questions:
        qid = str(q["_id"])
        difficulty = q.get("difficulty", "basic")
        weight = WEIGHTS.get(difficulty, 1)

        total_weight += weight

        selected_index = answers_map.get(qid)
        correct_index = q["correct_index"]
        topic = q.get("topic", "unknown")

        if topic not in topic_stats:
            topic_stats[topic] = {"correct": 0, "total": 0}

        topic_stats[topic]["total"] += 1

        if selected_index is not None and int(selected_index) == int(correct_index):
            earned_weight += weight
            topic_stats[topic]["correct"] += 1

    score = round((earned_weight / total_weight) * 100) if total_weight > 0 else 0

    topic_accuracy = {
        topic: round(stats["correct"] / stats["total"], 2) if stats["total"] > 0 else 0
        for topic, stats in topic_stats.items()
    }

    agent_result = run_tutor_agent(
        user_id=user_id,
        lesson_id=lesson_id,
        score=score,
        quiz_type=quiz_type,
        topic_accuracy=topic_accuracy,
    )
    decision = agent_result["decision"]

    attempt_doc = {
        "user_id": user_id,
        "lesson_id": lesson_id,
        "quiz_id": quiz_id,
        "attempt_no": attempt_no,
        "quiz_type": quiz_type,
        "difficulty_breakdown": difficulty_breakdown,
        "score": score,
        "time_taken_sec": time_taken_sec,
        "topic_accuracy": topic_accuracy,
        "decision": decision,
        "created_at": datetime.utcnow()
    }

    attempts_col.insert_one(attempt_doc)

    quizzes_col.update_one(
        {"_id": ObjectId(quiz_id)},
        {"$set": {"status": "submitted"}}
    )

    update_profile_after_attempt(
        user_id=user_id,
        lesson_id=lesson_id,
        score=score,
        topic_accuracy=topic_accuracy,
        decision=decision
    )

    return {
        "score": score,
        "topic_accuracy": topic_accuracy,
        "decision": decision,
        "quiz_type": quiz_type,
        "difficulty_breakdown": difficulty_breakdown
    }
