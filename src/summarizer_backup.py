
from pathlib import Path
import re

import torch
from transformers import pipeline


MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

OUTPUT_DIR = Path("outputs")
SUMMARY_FILE = OUTPUT_DIR / "summary.txt"
CHUNK_SUMMARIES_FILE = OUTPUT_DIR / "chunk_summaries.txt"


def load_summarization_model():
    """Load the pretrained summarization model."""

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_NAME}")

    summarizer = pipeline(
        task="summarization",
        model=MODEL_NAME,
        device=device,
    )

    print("Model loaded successfully.")
    return summarizer


def clean_text(text: str) -> str:
    """Clean whitespace and common PDF line-break artifacts."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Join words split by hyphenation at a line break.
    text = re.sub(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)", "", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    return text.strip()


def split_into_chunks(
    text: str,
    chunk_size: int = 2500,
    overlap: int = 300,
) -> list[str]:
    """Split text into overlapping character-based chunks."""

    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def summarize_text(
    text: str,
    summarizer,
    max_output_tokens: int = 130,
) -> str:
    """Summarize a single text chunk."""

    text = clean_text(text)

    if not text:
        return ""

    tokenizer = summarizer.tokenizer

    encoded = tokenizer(
        text,
        truncation=True,
        max_length=900,
        add_special_tokens=True,
    )

    input_tokens = len(encoded["input_ids"])

    if input_tokens < 50:
        max_length = max(15, int(input_tokens * 0.90))
        min_length = 5
    else:
        max_length = min(
            max_output_tokens,
            max(40, int(input_tokens * 0.45)),
        )
        min_length = min(25, max(10, int(max_length * 0.35)))

    max_length = max(max_length, min_length + 5)

    result = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        truncation=True,
    )

    return clean_text(result[0]["summary_text"])


def remove_duplicate_sentences(text: str) -> str:
    """Remove exact duplicate sentences while preserving order."""

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    unique_sentences = []
    seen = set()

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        key = re.sub(r"\W+", " ", sentence.lower()).strip()

        if key and key not in seen:
            seen.add(key)
            unique_sentences.append(sentence)

    return " ".join(unique_sentences)


def summarize_document(
    text: str,
    summarizer,
    group_size: int = 3,
) -> tuple[str, list[str]]:
    """
    Stage 1: Summarize source chunks.
    Stage 2: Consolidate groups of three chunk summaries.
    Stage 3: Combine the consolidated sections without
    another summarization pass.
    """

    text = clean_text(text)

    if not text:
        raise ValueError("The document contains no readable text.")

    chunks = split_into_chunks(text)

    if not chunks:
        raise ValueError("Could not create document chunks.")

    print(f"Created {len(chunks)} source chunks.")

    chunk_summaries = []

    for index, chunk in enumerate(chunks, start=1):
        print(f"Summarizing chunk {index}/{len(chunks)}...")

        summary = summarize_text(
            chunk,
            summarizer,
            max_output_tokens=130,
        )

        if summary:
            chunk_summaries.append(summary)

    if not chunk_summaries:
        raise RuntimeError("No chunk summaries were generated.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    CHUNK_SUMMARIES_FILE.write_text(
        "\n\n".join(chunk_summaries) + "\n",
        encoding="utf-8",
    )

    section_summaries = []

    total_groups = (
        len(chunk_summaries) + group_size - 1
    ) // group_size

    for start in range(0, len(chunk_summaries), group_size):
        group = chunk_summaries[start:start + group_size]
        group_text = "\n".join(group)

        group_number = start // group_size + 1

        print(
            f"Consolidating group {group_number}/{total_groups}..."
        )

        section_summary = summarize_text(
            group_text,
            summarizer,
            max_output_tokens=220,
        )

        if section_summary:
            section_summaries.append(section_summary)

    if not section_summaries:
        raise RuntimeError("No consolidated summaries were generated.")

    final_sections = []

    for index, section in enumerate(section_summaries, start=1):
        section = remove_duplicate_sentences(section)

        if section:
            final_sections.append(
                f"Section {index}\n{section}"
            )

    final_summary = clean_text("\n\n".join(final_sections))

    SUMMARY_FILE.write_text(
        final_summary + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 50)
    print("HIERARCHICAL SUMMARIZATION COMPLETE")
    print("=" * 50)
    print(f"Source chunks: {len(chunks)}")
    print(f"Chunk summaries: {len(chunk_summaries)}")
    print(f"Consolidated sections: {len(section_summaries)}")
    print(f"Final summary words: {len(final_summary.split())}")
    print(f"Saved summary to: {SUMMARY_FILE}")

    return final_summary, chunk_summaries


def main():
    """Run summarization on previously extracted document text."""

    input_file = Path("data/processed/cleaned_text.txt")

    if not input_file.exists():
        input_file = Path("data/processed/extracted_text.txt")

    if not input_file.exists():
        print("No extracted document text found.")
        print("Run PDF extraction first.")
        return

    text = input_file.read_text(encoding="utf-8")
    summarizer = load_summarization_model()

    final_summary, _ = summarize_document(text, summarizer)

    print("\nFINAL SUMMARY\n")
    print(final_summary)


if __name__ == "__main__":
    main()
