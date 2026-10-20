
import json
import random
import re
from pathlib import Path

from transformers import pipeline


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "outputs"
JSON_OUTPUT = OUTPUT_DIR / "generated_questions.json"
TEXT_OUTPUT = OUTPUT_DIR / "generated_questions.txt"

MODEL_NAME = "google/flan-t5-base"


def load_question_model():
    """Load the question-generation model."""
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Loading question model on {device}...")

    return pipeline(
        "text2text-generation",
        model=MODEL_NAME,
        device=device,
    )


def extract_candidate_facts(text, max_facts=40):
    """
    Extract candidate factual sentences from the uploaded document.

    These are candidate facts, not independently verified facts.
    """

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?m)^---\s*Page\s+\d+\s*---\s*$", " ", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    sentences = re.split(r"(?<=[.!?])\s+", text)

    facts = []
    seen = set()

    for sentence in sentences:
        sentence = sentence.strip(" \t\n-•")

        word_count = len(sentence.split())

        if not 8 <= word_count <= 45:
            continue

        # Skip sentences that are mostly citations or broken fragments.
        if sentence.count("(") > sentence.count(")") + 2:
            continue

        key = re.sub(r"\W+", " ", sentence.lower()).strip()

        if not key or key in seen:
            continue

        if not re.search(r"[A-Za-z]{3,}", sentence):
            continue

        seen.add(key)
        facts.append(sentence)

    # Spread selected facts throughout the document.
    if len(facts) > max_facts:
        positions = [
            round(i * (len(facts) - 1) / (max_facts - 1))
            for i in range(max_facts)
        ]
        facts = [facts[i] for i in positions]

    return facts


def generate_question(fact, question_model):
    """Generate a question answerable from one source fact."""

    prompt = (
        "Write one clear educational question whose answer is "
        "directly stated in the fact below. Ask about the specific "
        "information, not 'this statement'. Do not answer it. "
        "Return only one question.\n"
        f"Fact: {fact}\n"
        "Question:"
    )

    result = question_model(
        prompt,
        max_new_tokens=64,
        do_sample=False,
    )

    question = result[0]["generated_text"].strip().strip('"')
    question = re.sub(r"^(Question:)\s*", "", question, flags=re.I)

    if len(question.split()) < 4:
        return None

    if not question.endswith("?"):
        question = question.rstrip(". ") + "?"

    return question


def generate_mcqs_from_text(
    text,
    question_model,
    number_of_questions=10,
):
    """
    Generate MCQs from the supplied document text.

    Each correct answer is a source sentence. Other source sentences
    are used as distractors, so questions still require review.
    """

    facts = extract_candidate_facts(text)

    if len(facts) < 4:
        raise ValueError(
            "Not enough suitable sentences were found. "
            "Try a PDF with selectable text and fuller paragraphs."
        )

    count = min(number_of_questions, len(facts))
    rng = random.Random(42)

    # Sample across the document instead of using only its first pages.
    if count == len(facts):
        selected_facts = facts[:]
    else:
        indexes = [
            round(i * (len(facts) - 1) / (count - 1))
            for i in range(count)
        ] if count > 1 else [len(facts) // 2]

        selected_facts = [facts[i] for i in indexes]

    questions = []
    used_questions = set()

    for index, correct_answer in enumerate(selected_facts, start=1):
        print(f"Generating question {index}/{len(selected_facts)}...")

        question = generate_question(correct_answer, question_model)

        if not question or question.lower() in used_questions:
            continue

        used_questions.add(question.lower())

        # Pick distinct source sentences as distractors.
        distractor_pool = [
            fact for fact in facts
            if fact != correct_answer
        ]
        rng.shuffle(distractor_pool)

        distractors = distractor_pool[:3]

        if len(distractors) < 3:
            continue

        options = [correct_answer] + distractors
        rng.shuffle(options)

        correct_letter = "ABCD"[options.index(correct_answer)]

        questions.append({
            "question_number": len(questions) + 1,
            "question": question,
            "options": {
                letter: option
                for letter, option in zip("ABCD", options)
            },
            "correct_option": correct_letter,
            "correct_answer": correct_answer,
            "explanation": (
                "The uploaded document states: " + correct_answer
            ),
            "source_fact": correct_answer,
            "review_required": True,
        })

    if not questions:
        raise ValueError(
            "The model did not produce usable questions. "
            "Try another document or inspect the extracted text."
        )

    return questions


def save_results(questions):
    """Save the generated quiz as JSON and readable text."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    JSON_OUTPUT.write_text(
        json.dumps(questions, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = []

    for item in questions:
        lines.append(
            f"\nQ{item['question_number']}. {item['question']}"
        )

        for letter, option in item["options"].items():
            lines.append(f"  {letter}. {option}")

        lines.append(
            f"Correct answer: {item['correct_option']}. "
            f"{item['correct_answer']}"
        )
        lines.append(f"Explanation: {item['explanation']}")
        lines.append("Review required: Yes")

    TEXT_OUTPUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"Saved questions to: {JSON_OUTPUT}")
    print(f"Saved readable quiz to: {TEXT_OUTPUT}")


if __name__ == "__main__":
    print(
        "This module generates questions from supplied document text. "
        "Use the Streamlit application to upload a PDF and generate a quiz."
    )
