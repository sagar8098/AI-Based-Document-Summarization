
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """
    Clean extracted document text.

    - Normalize line endings
    - Remove unnecessary whitespace
    - Preserve paragraph boundaries
    """

    # Normalize Windows and old Mac line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove spaces and tabs at the beginning/end of lines
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Replace 3 or more consecutive newlines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace repeated spaces and tabs within lines
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def load_and_clean_text(file_path: str) -> str:
    """Load an extracted text file and clean its contents."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    text = path.read_text(encoding="utf-8")

    return clean_text(text)


if __name__ == "__main__":
    input_path = "data/processed/extracted_text.txt"

    try:
        cleaned_text = load_and_clean_text(input_path)

        print("Text preprocessing successful!")
        print(f"Characters after cleaning: {len(cleaned_text)}")
        print("\nPreview:\n")
        print(cleaned_text[:1500])

        output_path = Path("data/processed/cleaned_text.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(cleaned_text, encoding="utf-8")

        print(f"\nCleaned text saved to: {output_path}")

    except FileNotFoundError as error:
        print(f"Error: {error}")
