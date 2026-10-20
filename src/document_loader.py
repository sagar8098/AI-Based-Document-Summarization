
import fitz
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF document.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text as a single string.
    """

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")

    if pdf_file.suffix.lower() != ".pdf":
        raise ValueError("Please provide a PDF file.")

    extracted_pages = []

    with fitz.open(pdf_file) as document:
        if len(document) == 0:
            raise ValueError("The PDF document is empty.")

        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text").strip()

            if page_text:
                extracted_pages.append(
                    f"\n--- Page {page_number} ---\n{page_text}"
                )

    full_text = "\n".join(extracted_pages)

    if not full_text.strip():
        raise ValueError(
            "No readable text found. The PDF may contain scanned images."
        )

    return full_text


if __name__ == "__main__":
    print("PDF Text Extractor")
    print("------------------")

    pdf_path = input("Enter the path to your PDF: ").strip()
    pdf_path = pdf_path.strip('"').strip("'")

    try:
        text = extract_text_from_pdf(pdf_path)

        print("\nExtraction successful!")
        print(f"Characters extracted: {len(text)}")
        print("\nFirst 2000 characters:\n")
        print(text[:2000])

        output_path = Path("data/processed/extracted_text.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

        print(f"\nFull extracted text saved to: {output_path}")

    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")
