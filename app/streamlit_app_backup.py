
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.document_loader import extract_text_from_pdf
from src.preprocessing import clean_text
from src.summarizer import load_summarization_model, summarize_document
from src.question_generator import (
    load_question_model,
    generate_mcqs_from_text,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="📄",
    layout="wide",
)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def get_summarizer():
    return load_summarization_model()


@st.cache_resource
def get_question_model():
    return load_question_model()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "document_name" not in st.session_state:
    st.session_state.document_name = ""

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "quiz" not in st.session_state:
    st.session_state.quiz = []

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False


# --------------------------------------------------
# APPLICATION UI
# --------------------------------------------------

st.title("📄 Document AI Assistant")
st.write(
    "Upload a PDF to generate an AI summary and a quiz "
    "based on the uploaded document."
)

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
)

summary_style = st.selectbox(
    "Choose summary length",
    ["Short", "Standard", "Detailed"],
    index=2,
)

summary_settings = {
    "Short": 5,
    "Standard": 3,
    "Detailed": 2,
}


# Reset previous results if a different document is uploaded.
if uploaded_file is not None:
    if st.session_state.document_name != uploaded_file.name:
        st.session_state.document_text = ""
        st.session_state.document_name = uploaded_file.name
        st.session_state.summary = ""
        st.session_state.quiz = []
        st.session_state.quiz_submitted = False


if uploaded_file is not None:
    st.info(f"Selected document: {uploaded_file.name}")

    col1, col2 = st.columns(2)

    # ----------------------------------------------
    # EXTRACT DOCUMENT TEXT
    # ----------------------------------------------

    with col1:
        if st.button("Extract PDF Text", use_container_width=True):
            try:
                with st.spinner("Extracting PDF text..."):
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf",
                        delete=True,
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file.flush()

                        extracted_text = extract_text_from_pdf(
                            temp_file.name
                        )

                cleaned_text = clean_text(extracted_text)

                if not cleaned_text:
                    st.error("No readable text was found in the PDF.")
                else:
                    st.session_state.document_text = cleaned_text
                    st.session_state.summary = ""
                    st.session_state.quiz = []
                    st.session_state.quiz_submitted = False
                    st.success("PDF text extracted successfully!")

            except Exception as error:
                st.error(f"PDF extraction failed: {error}")

    # ----------------------------------------------
    # GENERATE SUMMARY
    # ----------------------------------------------

    with col2:
        if st.button("Generate Summary", type="primary",
                     use_container_width=True):
            try:
                # Extract automatically if not already extracted.
                if not st.session_state.document_text:
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf",
                        delete=True,
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file.flush()

                        extracted_text = extract_text_from_pdf(
                            temp_file.name
                        )

                    st.session_state.document_text = clean_text(
                        extracted_text
                    )

                with st.spinner(
                    "Generating summary. Large PDFs may take several minutes..."
                ):
                    summarizer = get_summarizer()

                    summary, _ = summarize_document(
                        st.session_state.document_text,
                        summarizer,
                        group_size=summary_settings[summary_style],
                    )

                st.session_state.summary = summary
                st.success("Summary generated!")

            except Exception as error:
                st.error(f"Summarization failed: {error}")

    # ----------------------------------------------
    # DISPLAY EXTRACTED TEXT
    # ----------------------------------------------

    if st.session_state.document_text:
        with st.expander("View extracted text"):
            st.text_area(
                "Document content",
                value=st.session_state.document_text,
                height=250,
                disabled=True,
            )

    # ----------------------------------------------
    # DISPLAY SUMMARY
    # ----------------------------------------------

    if st.session_state.summary:
        summary = st.session_state.summary

        st.subheader(f"{summary_style} Document Summary")
        st.write(summary)

        original_words = len(
            st.session_state.document_text.split()
        )
        summary_words = len(summary.split())

        reduction = (
            (1 - summary_words / original_words) * 100
            if original_words else 0
        )

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric("Original words", f"{original_words:,}")
        metric2.metric("Summary words", f"{summary_words:,}")
        metric3.metric("Word reduction", f"{reduction:.1f}%")

        st.download_button(
            "Download Summary",
            data=summary,
            file_name="document_summary.txt",
            mime="text/plain",
        )

    st.divider()

    # ----------------------------------------------
    # GENERATE DOCUMENT-SPECIFIC QUIZ
    # ----------------------------------------------

    st.subheader("📝 Generate Quiz")

    number_of_questions = st.slider(
        "Number of questions",
        min_value=3,
        max_value=10,
        value=5,
    )

    if st.button("Generate Quiz", use_container_width=True):
        try:
            if not st.session_state.document_text:
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf",
                    delete=True,
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file.flush()

                    extracted_text = extract_text_from_pdf(
                        temp_file.name
                    )

                st.session_state.document_text = clean_text(
                    extracted_text
                )

            with st.spinner(
                "Generating questions from your document..."
            ):
                question_model = get_question_model()

                quiz = generate_mcqs_from_text(
                    st.session_state.document_text,
                    question_model,
                    number_of_questions=number_of_questions,
                )

            st.session_state.quiz = quiz
            st.session_state.quiz_submitted = False
            st.success(f"Generated {len(quiz)} questions!")

        except Exception as error:
            st.error(f"Quiz generation failed: {error}")

    # ----------------------------------------------
    # TAKE THE QUIZ
    # ----------------------------------------------

    if st.session_state.quiz:
        st.subheader("Test Your Understanding")

        with st.form("document_quiz"):
            selected_answers = {}

            for question in st.session_state.quiz:
                st.markdown(
                    f"**Q{question['question_number']}. "
                    f"{question['question']}**"
                )

                options = question["options"]

                selected_answers[question["question_number"]] = (
                    st.radio(
                        "Choose your answer:",
                        options=list(options.keys()),
                        format_func=lambda key, opts=options: (
                            f"{key}. {opts[key]}"
                        ),
                        key=f"answer_{question['question_number']}",
                        index=None,
                    )
                )

                st.write("")

            submitted = st.form_submit_button(
                "Submit Quiz",
                type="primary",
            )

            if submitted:
                st.session_state.quiz_answers = selected_answers
                st.session_state.quiz_submitted = True

        # ------------------------------------------
        # SCORE AND EXPLANATIONS
        # ------------------------------------------

        if st.session_state.quiz_submitted:
            quiz = st.session_state.quiz
            answers = st.session_state.quiz_answers

            score = 0

            for question in quiz:
                selected = answers.get(
                    question["question_number"]
                )

                if selected == question["correct_option"]:
                    score += 1

            st.subheader("Quiz Results")

            st.metric(
                "Your score",
                f"{score} / {len(quiz)}",
            )

            st.progress(score / len(quiz) if quiz else 0)

            for question in quiz:
                selected = answers.get(
                    question["question_number"]
                )
                correct = question["correct_option"]

                st.markdown(
                    f"**Q{question['question_number']}. "
                    f"{question['question']}**"
                )

                if selected == correct:
                    st.success("Correct!")
                else:
                    st.error(
                        f"Incorrect. Your answer: "
                        f"{selected if selected else 'Not answered'}"
                    )

                st.write(
                    f"**Correct answer:** {correct}. "
                    f"{question['options'][correct]}"
                )

                st.write(
                    f"**Explanation:** {question['explanation']}"
                )

                st.divider()


st.divider()
st.caption(
    "Document AI Assistant | PDF extraction, summarization, "
    "and document-specific quiz generation"
)
