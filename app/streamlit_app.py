
import sys
import html
import tempfile
from pathlib import Path

import streamlit as st

# =========================================================
# 1. PROJECT SETUP
# =========================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.document_loader import extract_text_from_pdf
from src.preprocessing import clean_text
from src.summarizer import load_summarization_model, summarize_document
from src.question_generator import (
    load_question_model,
    generate_mcqs_from_text,
)

# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="DocumentAI Studio",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 3. CUSTOM STYLING
# =========================================================
st.markdown(
    """
   
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #7657F6;
    --ink: #172033;
    --muted: #667085;
    --page: #F6F7FB;
    --border: #E4E7EF;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp,
[data-testid="stAppViewContainer"] {
    background: #F6F7FB !important;
    color: #172033 !important;
}

/* Main content text */
[data-testid="stMain"] p,
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] label,
[data-testid="stMain"] li,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] {
    color: #172033 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #151B2C !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] label * {
    color: #E5E7EB !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: #AAB3C5 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border-radius: 9px;
    padding: 8px 10px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #292343 !important;
}

/* Brand */
.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0 25px;
}

.brand-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: linear-gradient(135deg, #9279FF, #6946E8);
    font-size: 23px;
}

.brand-name {
    color: #FFFFFF !important;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 21px;
}

.brand-subtitle {
    color: #B6BED1 !important;
    font-size: 10px;
    letter-spacing: 1.5px;
}

/* Headings */
.eyebrow {
    color: #6244E8 !important;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.8px;
    margin-bottom: 8px;
}

.hero-title {
    color: #172033 !important;
    font-family: 'Manrope', sans-serif;
    font-size: clamp(28px, 4vw, 42px);
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 12px;
}

.hero-description,
.section-description {
    color: #667085 !important;
    line-height: 1.7;
}

.section-title {
    color: #172033 !important;
    font-family: 'Manrope', sans-serif;
    font-size: 21px;
    font-weight: 800;
    margin: 8px 0;
}

/* Cards */
.surface-card,
.metric-card {
    background: #FFFFFF !important;
    border: 1px solid #E4E7EF;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 14px;
}

.feature-title,
.metric-value {
    color: #172033 !important;
    font-weight: 800;
}

.feature-description,
.metric-label {
    color: #667085 !important;
    line-height: 1.7;
}

.feature-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 46px;
    height: 46px;
    border-radius: 14px;
    background: #F0EDFF;
    font-size: 22px;
    margin-bottom: 12px;
}

.metric-label {
    font-size: 12px;
    margin-bottom: 9px;
}

.metric-value {
    font-family: 'Manrope', sans-serif;
    font-size: 25px;
    overflow-wrap: anywhere;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    border-radius: 11px;
    min-height: 43px;
    font-weight: 700;
}

.stButton > button[kind="primary"] {
    background: #7657F6 !important;
    border-color: #7657F6 !important;
    color: #FFFFFF !important;
}

/* Uploader */
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1px dashed #B9ACFF !important;
    border-radius: 16px;
    padding: 12px;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] label *,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section *,
[data-testid="stFileUploader"] small {
    color: #374151 !important;
}

[data-testid="stFileUploader"] button {
    background: #F0EDFF !important;
    color: #6244E8 !important;
    border: 1px solid #D8D0FF !important;
}

/* Quiz answer labels: target radio widgets in main content */
[data-testid="stMain"] [data-testid="stRadio"] label,
[data-testid="stMain"] [data-testid="stRadio"] label *,
[data-testid="stMain"] [data-testid="stRadio"] p,
[data-testid="stMain"] [data-testid="stRadio"] span {
    color: #263247 !important;
    opacity: 1 !important;
}

[data-testid="stMain"] [data-testid="stRadio"] label {
    border-radius: 8px;
    padding: 8px 10px;
    line-height: 1.6;
}

[data-testid="stMain"] [data-testid="stRadio"] label:hover {
    background: #F0EDFF !important;
}

/* Summary reading card */
.summary-card {
    background: #FFFFFF !important;
    color: #172033 !important;
    border: 1px solid #E2E6EF;
    border-radius: 16px;
    padding: 28px;
    margin: 12px 0 20px;
    font-size: 16px;
    line-height: 1.9;
    overflow-wrap: anywhere;
    white-space: normal;
}

.summary-card,
.summary-card * {
    color: #172033 !important;
}

.summary-card-label {
    color: #6244E8 !important;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin-bottom: 16px;
}

.footer {
    color: #667085 !important;
    text-align: center;
    font-size: 12px;
    padding: 26px 0 10px;
}

@media (max-width: 700px) {
    .hero-title {
        font-size: 29px;
    }

    .surface-card,
    .metric-card {
        padding: 16px;
    }

    .summary-card {
        padding: 18px;
        font-size: 15px;
    }
}
</style>

    """,
    unsafe_allow_html=True,
)

# =========================================================
# 4. MODEL CACHING
# =========================================================
@st.cache_resource(show_spinner=False)
def get_summarizer():
    return load_summarization_model()


@st.cache_resource(show_spinner=False)
def get_question_model():
    return load_question_model()


# =========================================================
# 5. SESSION STATE
# =========================================================
DEFAULTS = {
    "document_text": "",
    "document_name": "",
    "document_id": "",
    "summary": "",
    "chunk_summaries": [],
    "quiz": [],
    "quiz_submitted": False,
    "quiz_answers": {},
    "page_navigation": "Dashboard",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_document_state():
    st.session_state.document_text = ""
    st.session_state.document_name = ""
    st.session_state.document_id = ""
    st.session_state.summary = ""
    st.session_state.chunk_summaries = []
    st.session_state.quiz = []
    st.session_state.quiz_submitted = False
    st.session_state.quiz_answers = {}


def navigate_to(page_name):
    st.session_state.page_navigation = page_name


def page_heading(kicker, title, description):
    st.markdown(
        f'<div class="eyebrow">{html.escape(kicker)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="hero-title">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="hero-description">{html.escape(description)}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{html.escape(str(label))}</div>
            <div class="metric-value">{html.escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 6. SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">✦</div>
            <div>
                <div class="brand-name">DocumentAI</div>
                <div class="brand-subtitle">INTELLIGENCE STUDIO</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("WORKSPACE")

    page = st.radio(
        "Navigation",
        ["Dashboard", "Summarize", "Generate Quiz", "About"],
        key="page_navigation",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("YOUR DOCUMENT")

    if st.session_state.document_name:
        st.markdown(f"📄 **{st.session_state.document_name}**")
        st.caption(
            f"{len(st.session_state.document_text.split()):,} extracted words"
        )

        if st.button("Clear current document", use_container_width=True):
            reset_document_state()
            st.rerun()
    else:
        st.caption("Upload a PDF to begin.")

    st.markdown("---")
    st.caption("Built with Python, Streamlit and Hugging Face Transformers.")


# =========================================================
# 7. PDF UPLOAD AND EXTRACTION
# =========================================================
def document_uploader():
    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Choose a text-based PDF. Scanned PDFs may require OCR.",
        key="pdf_upload",
    )

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}:{uploaded_file.size}"

        if file_id != st.session_state.document_id:
            st.info(f"Selected PDF: **{uploaded_file.name}**")

            if st.button(
                "Extract document text",
                type="primary",
                use_container_width=True,
                key="extract_pdf",
            ):
                temp_path = None

                try:
                    with st.spinner("Extracting text from your PDF..."):
                        with tempfile.NamedTemporaryFile(
                            suffix=".pdf",
                            delete=False,
                        ) as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_path = Path(temp_file.name)

                        extracted_text = extract_text_from_pdf(str(temp_path))
                        extracted_text = clean_text(extracted_text)

                    if not extracted_text.strip():
                        st.error(
                            "No readable text was found. This PDF may be scanned "
                            "and may require OCR."
                        )
                    else:
                        reset_document_state()
                        st.session_state.document_text = extracted_text
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.document_id = file_id
                        st.success("Text extracted successfully.")
                        st.rerun()

                except Exception as error:
                    st.error(f"PDF extraction failed: {error}")

                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)

    if st.session_state.document_text:
        st.success(
            f"Ready: {st.session_state.document_name} · "
            f"{len(st.session_state.document_text.split()):,} words extracted"
        )


# =========================================================
# 8. DASHBOARD
# =========================================================
if page == "Dashboard":
    page_heading(
        "YOUR AI WORKSPACE",
        "Turn documents into knowledge.",
        "Upload a PDF, generate an AI summary, and create a quiz to "
        "check your understanding — all in one workspace.",
    )

    left, right = st.columns([1.25, 0.75], gap="large")

    with left:
        st.markdown(
            '<div class="section-title">Start with a document</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-description">Upload your study material, '
            'report, notes or research paper.</div>',
            unsafe_allow_html=True,
        )

        document_uploader()

        if st.session_state.document_text:
            st.markdown("### Your document is ready")

            c1, c2 = st.columns(2)

            with c1:
                st.button(
                    "Open summarizer →",
                    type="primary",
                    use_container_width=True,
                    key="open_summarizer",
                    on_click=navigate_to,
                    args=("Summarize",),
                )

            with c2:
                st.button(
                    "Create a quiz →",
                    use_container_width=True,
                    key="open_quiz",
                    on_click=navigate_to,
                    args=("Generate Quiz",),
                )

    with right:
        st.markdown(
            """
            <div class="surface-card">
                <div class="feature-icon">✨</div>
                <div class="feature-title">AI-powered summarization</div>
                <div class="feature-description">
                    Turn long documents into shorter, easier-to-read summaries
                    using the DistilBART model.
                </div>
            </div>
            <div class="surface-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">Document-based quiz</div>
                <div class="feature-description">
                    Generate multiple-choice questions from extracted content
                    and review your answers.
                </div>
            </div>
            <div class="surface-card">
                <div class="feature-icon">📚</div>
                <div class="feature-title">One learning workspace</div>
                <div class="feature-description">
                    Extract, summarize, review and test your understanding
                    without switching between tools.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div class="section-title">Workspace overview</div>',
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)

    with a:
        metric_card(
            "DOCUMENT WORDS",
            f"{len(st.session_state.document_text.split()):,}"
            if st.session_state.document_text else "—",
        )

    with b:
        metric_card(
            "SUMMARY",
            "Ready" if st.session_state.summary else "Not generated",
        )

    with c:
        metric_card(
            "QUIZ QUESTIONS",
            str(len(st.session_state.quiz))
            if st.session_state.quiz else "—",
        )


# =========================================================
# 9. SUMMARIZATION
# =========================================================
elif page == "Summarize":
    page_heading(
        "DOCUMENT INTELLIGENCE",
        "Summarize your document.",
        "Create a concise version of your document using your existing "
        "AI summarization pipeline.",
    )

    document_uploader()

    if not st.session_state.document_text:
        st.info("Upload a PDF and extract its text to get started.")

    else:
        text = st.session_state.document_text
        word_count = len(text.split())

        a, b, c = st.columns(3)

        with a:
            metric_card("DOCUMENT WORDS", f"{word_count:,}")

        with b:
            metric_card("CHARACTERS", f"{len(text):,}")

        with c:
            metric_card("DOCUMENT", st.session_state.document_name)

        st.markdown(
            '<div class="section-title">Summary settings</div>',
            unsafe_allow_html=True,
        )

        settings_col, button_col = st.columns([2, 1])

        with settings_col:
            st.markdown(
                '<div style="color:#172033;font-weight:600;margin-bottom:6px;">'
                'Summary style</div>',
                unsafe_allow_html=True,
            )

            summary_length = st.selectbox(
                "Summary style",
                ["Short", "Standard", "Detailed"],
                index=1,
                label_visibility="collapsed",
                help=(
                    "Controls how chunk summaries are grouped for "
                    "consolidation; it does not guarantee an exact word count."
                ),
                key="summary_length",
            )

        group_size = {
            "Short": 5,
            "Standard": 3,
            "Detailed": 2,
        }[summary_length]

        with button_col:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

            generate_summary = st.button(
                "✨ Generate summary",
                type="primary",
                use_container_width=True,
                key="generate_summary",
            )

        if generate_summary:
            try:
                with st.spinner(
                    "Loading the summarization model and processing your document. "
                    "Long documents may take several minutes..."
                ):
                    summarizer = get_summarizer()

                    final_summary, chunk_summaries = summarize_document(
                        text,
                        summarizer,
                        group_size=group_size,
                    )

                if not final_summary or not final_summary.strip():
                    raise ValueError(
                        "The summarization pipeline returned an empty summary."
                    )

                st.session_state.summary = final_summary
                st.session_state.chunk_summaries = chunk_summaries

                st.success("Summary generated successfully.")

            except Exception as error:
                st.error(
                    "Summarization failed. Check the terminal for the full error."
                )
                st.exception(error)

        if st.session_state.summary:
            summary = st.session_state.summary
            summary_words = len(summary.split())

            reduction = (
                max(0, round((1 - summary_words / word_count) * 100))
                if word_count else 0
            )

            st.markdown("---")
            st.markdown(
                '<div class="section-title">Your summary</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-description">Review the generated '
                'summary against the original document.</div>',
                unsafe_allow_html=True,
            )

            x, y, z = st.columns(3)

            with x:
                metric_card("SUMMARY WORDS", f"{summary_words:,}")

            with y:
                metric_card("WORD REDUCTION", f"{reduction}%")

            with z:
                metric_card(
                    "CHUNK SUMMARIES",
                    str(len(st.session_state.chunk_summaries)),
                )

            # Escape generated text before placing it in a styled HTML card.
            safe_summary = html.escape(summary).replace("\n", "<br>")

            st.markdown(
                f"""
                <div class="summary-card">
                    <div class="summary-card-label">GENERATED SUMMARY</div>
                    <div style="color:#172033 !important;">
                        {safe_summary}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.download_button(
                "⬇ Download summary",
                data=summary,
                file_name="document_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )

            with st.expander("View extracted source text"):
                st.text_area(
                    "Extracted document text",
                    text,
                    height=260,
                    label_visibility="collapsed",
                )

            with st.expander("View individual chunk summaries"):
                for index, item in enumerate(
                    st.session_state.chunk_summaries,
                    start=1,
                ):
                    st.markdown(f"**Chunk {index}**")
                    st.write(item)


# =========================================================
# 10. QUESTION GENERATION AND QUIZ
# =========================================================
elif page == "Generate Quiz":
    page_heading(
        "ACTIVE RECALL",
        "Test what you've learned.",
        "Generate multiple-choice questions from your document and "
        "review the answers after submission.",
    )

    document_uploader()

    if not st.session_state.document_text:
        st.info("Upload a PDF and extract its text before generating a quiz.")

    else:
        st.markdown(f"**Source document:** {st.session_state.document_name}")

        number_of_questions = st.slider(
            "Number of questions",
            min_value=3,
            max_value=10,
            value=5,
            key="number_of_questions",
        )

        if st.button(
            "🧠 Generate quiz",
            type="primary",
            use_container_width=True,
            key="generate_quiz",
        ):
            try:
                with st.spinner(
                    "Loading the question model and generating questions..."
                ):
                    question_model = get_question_model()

                    questions = generate_mcqs_from_text(
                        st.session_state.document_text,
                        question_model,
                        number_of_questions=number_of_questions,
                    )

                if not questions:
                    raise ValueError(
                        "The model did not generate any usable questions."
                    )

                st.session_state.quiz = questions
                st.session_state.quiz_submitted = False
                st.session_state.quiz_answers = {}

                st.success(f"Generated {len(questions)} questions.")

            except Exception as error:
                st.error("Quiz generation failed. Check the terminal for details.")
                st.exception(error)

        questions = st.session_state.quiz

        if questions:
            st.markdown("---")
            st.markdown(
                f'<div class="section-title">Your quiz · {len(questions)} questions</div>',
                unsafe_allow_html=True,
            )

            st.caption(
                "Select one answer per question, then submit to review "
                "your score and explanations."
            )

            if not st.session_state.quiz_submitted:
                with st.form("document_quiz_form"):
                    answers = {}

                    for index, item in enumerate(questions):
                        st.markdown(f"**Question {index + 1}**")
                        st.write(item.get("question", "Question unavailable."))

                        options = item.get("options", {})
                        option_letters = list(options.keys())

                        if not option_letters:
                            st.warning(
                                "This question has no available options. "
                                "Please generate a new quiz."
                            )
                            continue

                        selected = st.radio(
                            f"Answer {index + 1}",
                            options=option_letters,
                            format_func=lambda letter, opts=options: (
                                f"{letter}. {opts[letter]}"
                            ),
                            index=None,
                            key=f"quiz_answer_{index}",
                            label_visibility="collapsed",
                        )

                        answers[index] = selected
                        st.divider()

                    submitted = st.form_submit_button(
                        "Submit answers",
                        type="primary",
                        use_container_width=True,
                    )

                if submitted:
                    st.session_state.quiz_answers = answers
                    st.session_state.quiz_submitted = True
                    st.rerun()

            else:
                answers = st.session_state.quiz_answers
                score = sum(
                    answers.get(index) == item.get("correct_option")
                    for index, item in enumerate(questions)
                )

                percentage = (
                    round(score / len(questions) * 100)
                    if questions else 0
                )

                score_col, accuracy_col = st.columns(2)

                with score_col:
                    metric_card("YOUR SCORE", f"{score} / {len(questions)}")

                with accuracy_col:
                    metric_card("ACCURACY", f"{percentage}%")

                st.progress(percentage / 100)

                if percentage >= 80:
                    st.success("Great work! Review your answers below.")
                elif percentage >= 50:
                    st.info(
                        "Good attempt. Review the explanations to strengthen "
                        "your understanding."
                    )
                else:
                    st.warning(
                        "Review the source facts and try again to reinforce "
                        "the key concepts."
                    )

                st.markdown("### Answer review")

                for index, item in enumerate(questions):
                    selected = answers.get(index)
                    correct_letter = item.get("correct_option")
                    is_correct = selected == correct_letter
                    options = item.get("options", {})

                    with st.expander(
                        f"{'✅' if is_correct else '❌'} Question "
                        f"{index + 1} — {'Correct' if is_correct else 'Review'}",
                        expanded=not is_correct,
                    ):
                        st.markdown(f"**{item.get('question', '')}**")

                        if selected:
                            st.write(
                                f"Your answer: {selected}. "
                                f"{options.get(selected, '')}"
                            )
                        else:
                            st.write("Your answer: Not answered")

                        st.write(
                            f"Correct answer: {correct_letter}. "
                            f"{options.get(correct_letter, item.get('correct_answer', ''))}"
                        )

                        st.write(
                            item.get("explanation")
                            or "Check the corresponding information in the source document."
                        )

                        if item.get("source_fact"):
                            st.caption(f"Source fact: {item['source_fact']}")

                results_text = "\n\n".join(
                    [
                        f"Question {index + 1}: {item.get('question', '')}\n"
                        f"Your answer: {answers.get(index) or 'Not answered'}\n"
                        f"Correct answer: {item.get('correct_option', '')}\n"
                        f"Explanation: {item.get('explanation', '')}"
                        for index, item in enumerate(questions)
                    ]
                    + [
                        f"Final score: {score}/{len(questions)} ({percentage}%)"
                    ]
                )

                st.download_button(
                    "⬇ Download quiz results",
                    data=results_text,
                    file_name="quiz_results.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

                if st.button(
                    "Generate another quiz",
                    use_container_width=True,
                    key="new_quiz",
                ):
                    st.session_state.quiz = []
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.rerun()


# =========================================================
# 11. ABOUT
# =========================================================
elif page == "About":
    page_heading(
        "ABOUT THE PROJECT",
        "AI-powered document learning.",
        "DocumentAI uses natural language processing to make long documents "
        "easier to review and study.",
    )

    st.markdown(
        """
        <div class="surface-card">
            <div class="feature-title">Project capabilities</div>
            <div class="feature-description">
                • Extract text from text-based PDF files.<br>
                • Clean and prepare extracted text.<br>
                • Summarize longer documents using a pretrained model.<br>
                • Generate document-based multiple-choice questions.<br>
                • Calculate quiz scores and display explanations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)

    with a:
        metric_card("FRONTEND", "Streamlit")

    with b:
        metric_card("SUMMARIZATION", "DistilBART")

    with c:
        metric_card("QUESTION GENERATION", "FLAN-T5")

    st.warning(
        "AI-generated summaries and questions can contain mistakes. "
        "Check important information against the original document. "
        "Quiz distractors may require manual review."
    )


# =========================================================
# 12. FOOTER
# =========================================================
st.markdown(
    '<div class="footer">DocumentAI Studio · AI-assisted document learning</div>',
    unsafe_allow_html=True,
)
