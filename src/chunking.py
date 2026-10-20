
from pathlib import Path


def split_into_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150
) -> list[str]:
    """
    Split text into overlapping chunks.

    chunk_size and overlap are measured in characters,
    not words or model tokens.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer breaking at a word boundary.
        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # Move forward while keeping some overlap.
        next_start = end - overlap

        # Ensure progress even when a boundary is encountered.
        start = max(start + 1, next_start)

    return chunks


if __name__ == "__main__":
    input_path = Path("data/processed/cleaned_text.txt")

    if not input_path.exists():
        print("Cleaned text not found. Run preprocessing.py first.")
    else:
        text = input_path.read_text(encoding="utf-8")

        chunks = split_into_chunks(
            text,
            chunk_size=1000,
            overlap=150
        )

        output_dir = Path("data/processed/chunks")
        output_dir.mkdir(parents=True, exist_ok=True)

        for old_file in output_dir.glob("chunk_*.txt"):
            old_file.unlink()

        for index, chunk in enumerate(chunks, start=1):
            output_file = output_dir / f"chunk_{index:03d}.txt"
            output_file.write_text(chunk, encoding="utf-8")

        print("Text chunking successful!")
        print(f"Total characters: {len(text)}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Chunk size: 1000 characters")
        print(f"Overlap: 150 characters")

        for index, chunk in enumerate(chunks, start=1):
            print(f"\n--- Chunk {index} ({len(chunk)} characters) ---")
            print(chunk[:250])

        print(f"\nChunks saved in: {output_dir}")
