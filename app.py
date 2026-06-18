import io
import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader


APP_TITLE = "Compliance Module 1 Chatbot"
DEFAULT_PDF_CANDIDATES = [
    "/Users/dharmarajrathod/Downloads/Compliance - Revised Module 1 (1).pdf",
    os.path.join(os.path.dirname(__file__), "Compliance - Revised Module 1 (1).pdf"),
    "/Users/dharmarajrathod/Downloads/Module 1 - Revised.pdf",
    os.path.join(os.path.dirname(__file__), "Module 1 - Revised.pdf"),
    os.path.join(os.path.dirname(__file__), "Module 1 - Prior Authorization Foundations and CoverMyMeds Workflow.pdf"),
    os.path.join(os.path.dirname(__file__), "module1.pdf"),
]
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = "gemini-embedding-001"
UNAVAILABLE_MESSAGE = "This information is not available in the module."
PHI_WARNING = "Remove real personal, customer, or company information. Use fictional or deidentified data."
WELCOME_MESSAGE = "Ask a Module 1 question about KYC, onboarding, UBOs, OCR, LLMs, GDPR, or cross-border compliance."
GENERIC_API_ERROR = "The chatbot could not get a response right now."
PROMPT_TEMPLATE = """You are a course learning assistant.
Answer ONLY using the provided module content.
Do NOT use outside knowledge.
Keep answers brief and clear.
Maximum 5 lines.
If the answer is not in the module, say: 'This information is not available in the module.'
Context:
{context}

Question:
{question}"""
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
MAX_CHUNKS = 5
MAX_RESPONSE_LINES = 5
SIMILARITY_THRESHOLD = 0.32
LEXICAL_SIMILARITY_THRESHOLD = 0.12
GROUNDING_THRESHOLD = 0.25
SNIPPET_THRESHOLD = 0.18
MAX_SNIPPETS = 5
STOPWORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "do", "for",
    "from", "how", "i", "in", "is", "it", "me", "module", "of", "on", "or",
    "please", "tell", "that", "the", "this", "to", "what", "which", "with",
}


@dataclass
class PageContent:
    page_number: int
    section_title: str
    text: str


@dataclass
class Chunk:
    text: str
    tokens: set[str]
    index: int
    page_number: int
    section_title: str
    embedding: Optional[List[float]] = None


@dataclass
class Snippet:
    text: str
    page_number: int
    section_title: str
    score: float


def get_api_key() -> Optional[str]:
    session_key = st.session_state.get("gemini_api_key", "").strip()
    if session_key:
        return session_key

    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if env_key:
        return env_key

    try:
        return st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None


def get_gemini_client() -> genai.Client:
    return genai.Client(api_key=get_api_key())


def init_session_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("knowledge_chunks", [])
    st.session_state.setdefault("knowledge_label", None)
    st.session_state.setdefault("knowledge_text", "")
    st.session_state.setdefault("gemini_api_key", "")


def normalize_text(text: str) -> str:
    cleaned = text.replace("\x00", " ")
    cleaned = re.sub(
        r"Module 1 Revised:\s*Prior Authorization Foundations,\s*Failures and Public PA Resource Workflow Using AMA and CMS",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"Module\s+1\s+\|\s+Automated\s+KYC\s+and\s+Cross-Border\s+Digital\s+Onboarding",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def extract_section_title(raw_text: str, page_number: int) -> str:
    for raw_line in raw_text.splitlines():
        line = normalize_text(raw_line)
        if 4 <= len(line) <= 90 and re.search(r"[A-Za-z]", line):
            return line
    return f"Page {page_number}"


def extract_pages_from_reader(reader: PdfReader) -> List[PageContent]:
    pages: List[PageContent] = []
    for page_index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        normalized = normalize_text(raw_text)
        if not normalized:
            continue
        pages.append(
            PageContent(
                page_number=page_index,
                section_title=extract_section_title(raw_text, page_index),
                text=normalized,
            )
        )
    return pages


def extract_pages_from_pdf_bytes(pdf_bytes: bytes) -> List[PageContent]:
    return extract_pages_from_reader(PdfReader(io.BytesIO(pdf_bytes)))


def extract_pages_from_pdf_path(pdf_path: str) -> List[PageContent]:
    return extract_pages_from_reader(PdfReader(pdf_path))


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", text.lower()))


def meaningful_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS}


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def split_into_chunks(pages: Sequence[PageContent], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    chunks: List[Chunk] = []
    index = 0
    for page in pages:
        start = 0
        while start < len(page.text):
            end = min(len(page.text), start + chunk_size)
            chunk_text = page.text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        tokens=meaningful_tokens(chunk_text),
                        index=index,
                        page_number=page.page_number,
                        section_title=page.section_title,
                    )
                )
                index += 1
            if end >= len(page.text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def normalize_vector(values: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return list(values)
    return [value / norm for value in values]


def embed_texts(texts: Sequence[str], task_type: str) -> List[List[float]]:
    if not texts or not get_api_key():
        return []
    client = get_gemini_client()
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=list(texts),
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768,
        ),
    )
    return [normalize_vector(embedding.values) for embedding in response.embeddings]


def add_embeddings_to_chunks(chunks: Sequence[Chunk]) -> None:
    texts = [chunk.text for chunk in chunks]
    vectors = embed_texts(texts, "RETRIEVAL_DOCUMENT")
    if len(vectors) != len(chunks):
        return
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def embed_query(query: str) -> Optional[List[float]]:
    vectors = embed_texts([query], "RETRIEVAL_QUERY")
    return vectors[0] if vectors else None


def retrieve_relevant_chunks(query: str, chunks: Sequence[Chunk], limit: int = MAX_CHUNKS) -> tuple[List[Chunk], float]:
    query_tokens = meaningful_tokens(query)
    ordered_query_terms = [token for token in re.findall(r"[a-z0-9]{2,}", query.lower()) if token not in STOPWORDS]
    query_embedding = embed_query(query) if get_api_key() else None
    normalized_query = normalize_text(query).lower()
    asks_definition = normalized_query.startswith(("what is", "what are", "define", "explain"))
    scored = []

    for chunk in chunks:
        lexical_score = len(query_tokens & chunk.tokens) / max(len(query_tokens), 1) if query_tokens else 0.0
        semantic_score = cosine_similarity(query_embedding, chunk.embedding) if query_embedding and chunk.embedding else 0.0
        exact_bonus = 0.15 if query.lower() in chunk.text.lower() else 0.0
        chunk_text = chunk.text.lower()
        definition_bonus = 0.0
        if asks_definition and query_tokens:
            phrase = " ".join(token for token in ordered_query_terms if token in chunk_text)
            if phrase and (
                f"{phrase} is " in chunk_text
                or f"{phrase} are " in chunk_text
                or f"{phrase} means " in chunk_text
                or "best described as" in chunk_text
            ):
                definition_bonus = 0.35
            if phrase and f"{phrase} is a payer review process" in chunk_text:
                definition_bonus = 0.65
            if phrase and (f"what {phrase} is" in chunk_text or f"what {phrase} means" in chunk_text):
                definition_bonus = 0.0
        title_penalty = 0.10 if chunk.page_number == 1 and "course overview" not in chunk_text else 0.0
        score = (semantic_score * 0.75) + (lexical_score * 0.25) + exact_bonus + definition_bonus - title_penalty
        scored.append((score, lexical_score, chunk))

    scored.sort(key=lambda item: (item[0], item[1], -item[2].index), reverse=True)
    selected = [chunk for score, _, chunk in scored[:limit] if score > 0]
    top_score = scored[0][0] if scored else 0.0
    return selected, top_score


def build_support_snippets(question: str, chunks: Sequence[Chunk], limit: int = MAX_SNIPPETS) -> List[Snippet]:
    query_tokens = meaningful_tokens(question)
    lowered_question = question.lower()
    snippets: List[Snippet] = []
    seen = set()

    for chunk in chunks:
        for sentence in split_sentences(chunk.text):
            if sentence.endswith("?"):
                continue
            normalized = sentence.lower()
            if (
                "generative ai fo r prio r autho rizatio n" in normalized
                or "instructo r led versio n" in normalized
                or normalized.startswith("module title recommended duration")
                or normalized.startswith("course overview and module positioning")
                or normalized.startswith("and public pa resource workflow")
                or normalized.startswith("s in a pa learning workflow")
                or "learner reflection questions" in normalized
                or normalized.count(" minutes ") >= 2
            ):
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            sentence_tokens = meaningful_tokens(sentence)
            lexical_score = len(query_tokens & sentence_tokens) / max(len(query_tokens), 1) if query_tokens else 0.0
            phrase_bonus = 0.25 if lowered_question in normalized else 0.0
            token_bonus = 0.10 if any(token in normalized for token in query_tokens if len(token) > 4) else 0.0
            score = lexical_score + phrase_bonus + token_bonus
            if score >= SNIPPET_THRESHOLD:
                snippets.append(
                    Snippet(
                        text=sentence,
                        page_number=chunk.page_number,
                        section_title=chunk.section_title,
                        score=score,
                    )
                )

    snippets.sort(key=lambda snippet: snippet.score, reverse=True)
    return snippets[:limit]


def likely_contains_phi(text: str) -> bool:
    patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b(?:dob|date of birth)\b",
        r"\b(?:mrn|member id|patient id|policy id)\b",
        r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b\d{10}\b",
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
    ]
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def is_greeting_or_smalltalk(text: str) -> bool:
    normalized = re.sub(r"[^a-z\s]", " ", text.lower()).strip()
    return normalized in {
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
    }


def build_context(snippets: Sequence[Snippet]) -> str:
    return "\n\n".join(
        f"[Page {snippet.page_number} | Section: {snippet.section_title}]\n{snippet.text}"
        for snippet in snippets
    )


def build_prompt(question: str, snippets: Sequence[Snippet]) -> str:
    return PROMPT_TEMPLATE.format(context=build_context(snippets), question=question)


def trim_answer(answer: str) -> str:
    cleaned_lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not cleaned_lines:
        return UNAVAILABLE_MESSAGE
    return "\n".join(cleaned_lines[:MAX_RESPONSE_LINES])


def is_grounded_answer(answer: str, snippets: Sequence[Snippet]) -> bool:
    answer_tokens = meaningful_tokens(answer)
    if not answer_tokens:
        return True
    context_tokens = set()
    for snippet in snippets:
        context_tokens.update(meaningful_tokens(snippet.text))
    overlap = len(answer_tokens & context_tokens) / max(len(answer_tokens), 1)
    return overlap >= GROUNDING_THRESHOLD


def format_sources(snippets: Sequence[Snippet]) -> str:
    seen = []
    for snippet in snippets:
        label = f"Page {snippet.page_number} ({snippet.section_title})"
        if label not in seen:
            seen.append(label)
    return "Source: " + "; ".join(seen[:MAX_CHUNKS])


def build_extractive_fallback(snippets: Sequence[Snippet]) -> str:
    selected = [snippet.text for snippet in snippets[:3]]
    if not selected:
        return UNAVAILABLE_MESSAGE
    return "\n".join(selected)


def ask_pa_coach(question: str, snippets: Sequence[Snippet]) -> str:
    client = get_gemini_client()
    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=build_prompt(question, snippets),
        config=types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.1,
            top_k=1,
        ),
    )
    return trim_answer(response.text or "")


def build_knowledge_base(pages: Sequence[PageContent]) -> List[Chunk]:
    chunks = split_into_chunks(pages)
    try:
        add_embeddings_to_chunks(chunks)
    except Exception:
        pass
    return chunks


def reset_chat() -> None:
    st.session_state.messages = []


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Settings")
        st.text_input(
            "Gemini API key",
            type="password",
            key="gemini_api_key",
            help="Optional. You can also set GEMINI_API_KEY, GOOGLE_API_KEY, or Streamlit secrets.",
        )
        st.caption(f"Model: `{DEFAULT_MODEL}`")

        if st.button("Clear chat", use_container_width=True):
            reset_chat()
            st.rerun()

        st.divider()
        st.subheader("Knowledge PDF")
        uploaded_file = st.file_uploader("Upload compliance PDF", type=["pdf"])
        if uploaded_file is not None:
            try:
                load_knowledge_base_from_upload(uploaded_file)
                st.success(f"Loaded {uploaded_file.name}")
                st.rerun()
            except Exception as exc:
                st.error(f"Unable to load uploaded PDF: {exc}")

        st.caption("Answers are restricted to the loaded compliance module content.")


def load_knowledge_base_from_upload(uploaded_file) -> None:
    pdf_bytes = uploaded_file.getvalue()
    pages = extract_pages_from_pdf_bytes(pdf_bytes)
    if not pages:
        raise ValueError("No text could be extracted from the uploaded PDF.")
    full_text = " ".join(page.text for page in pages)
    st.session_state.knowledge_text = full_text
    st.session_state.knowledge_chunks = build_knowledge_base(pages)
    st.session_state.knowledge_label = uploaded_file.name
    st.session_state.messages = []


def load_knowledge_base_from_default_path(pdf_path: str) -> None:
    pages = extract_pages_from_pdf_path(pdf_path)
    if not pages:
        raise ValueError("No text could be extracted from the default PDF.")
    full_text = " ".join(page.text for page in pages)
    st.session_state.knowledge_text = full_text
    st.session_state.knowledge_chunks = build_knowledge_base(pages)
    st.session_state.knowledge_label = os.path.basename(pdf_path)
    st.session_state.messages = []


def ensure_default_pdf_loaded() -> None:
    if st.session_state.knowledge_chunks:
        return
    for pdf_path in DEFAULT_PDF_CANDIDATES:
        if os.path.exists(pdf_path):
            load_knowledge_base_from_default_path(pdf_path)
            return


def answer_question(question: str, chunks: Sequence[Chunk]) -> tuple[str, Optional[str]]:
    relevant_chunks, top_score = retrieve_relevant_chunks(question, chunks)
    has_embeddings = any(chunk.embedding for chunk in chunks)
    minimum_score = SIMILARITY_THRESHOLD if has_embeddings and get_api_key() else LEXICAL_SIMILARITY_THRESHOLD
    if not relevant_chunks or top_score < minimum_score:
        return UNAVAILABLE_MESSAGE, None
    snippets = build_support_snippets(question, relevant_chunks)
    if not snippets:
        return UNAVAILABLE_MESSAGE, None

    try:
        answer = ask_pa_coach(question, snippets) if get_api_key() else build_extractive_fallback(snippets)
    except Exception:
        answer = build_extractive_fallback(snippets)

    if answer == UNAVAILABLE_MESSAGE:
        return answer, None
    if not is_grounded_answer(answer, snippets):
        return UNAVAILABLE_MESSAGE, None
    return trim_answer(answer), format_sources(snippets)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_session_state()

    try:
        ensure_default_pdf_loaded()
    except Exception as exc:
        st.session_state.knowledge_chunks = []
        st.session_state.knowledge_label = None
        st.error(f"Default PDF load failed: {exc}")

    st.title("Compliance Coach")
    st.write(
        "Ask questions about automated KYC, cross-border digital onboarding, "
        "UBO checks, OCR, LLMs, GDPR, and compliance workflow. This assistant "
        "uses only the loaded compliance PDF."
    )
    st.caption(
        f"Knowledge source: `{st.session_state.knowledge_label or 'Not loaded'}` | "
        f"Active model: `{DEFAULT_MODEL}`"
    )

    if get_api_key():
        st.success("Gemini is configured. Answers will use retrieved compliance module context.")
    else:
        st.info("Gemini API key not configured. The chatbot is using extractive answers from the loaded PDF.")

    if st.button("Clear chat"):
        reset_chat()
        st.rerun()

    if not st.session_state.knowledge_chunks:
        st.info("Upload the compliance PDF to begin, or add the PDF file to the app repository.")
        uploaded_file = st.file_uploader("Upload compliance PDF", type=["pdf"])
        if uploaded_file is not None:
            try:
                load_knowledge_base_from_upload(uploaded_file)
                st.rerun()
            except Exception as exc:
                st.error(f"Unable to load uploaded PDF: {exc}")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask a compliance module question")
    if not user_query:
        return

    cleaned_query = user_query.strip()
    if not cleaned_query:
        st.warning("Enter a question to continue.")
        return

    st.session_state.messages.append({"role": "user", "content": cleaned_query})
    with st.chat_message("user"):
        st.markdown(cleaned_query)

    if likely_contains_phi(cleaned_query):
        assistant_reply = PHI_WARNING
    elif is_greeting_or_smalltalk(cleaned_query):
        assistant_reply = WELCOME_MESSAGE
    else:
        answer, source = answer_question(cleaned_query, st.session_state.knowledge_chunks)
        assistant_reply = answer if not source else f"{answer}\n\n{source}"

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)


if __name__ == "__main__":
    main()
