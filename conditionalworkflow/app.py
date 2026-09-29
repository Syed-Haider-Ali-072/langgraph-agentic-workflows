import os

# model pehle se download ho chuka hai, is liye internet check band (atakne se bachata hai)
# agar kabhi model dobara download karna ho to ye line hata do
os.environ["HF_HUB_OFFLINE"] = "1"

# Streamlit ke file watcher ko band kar rahe hain (import se pehle set karna zaroori hai).
# Warna wo `transformers` ke andar sainkron files scan karta hai jo `torchvision` maangti
# hain (jo install nahi hai), aur pehli load me bohot der lag jati hai.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("STREAMLIT_SERVER_RUN_ON_SAVE", "false")
os.environ.setdefault("STREAMLIT_CLIENT_SHOW_ERROR_DETAILS", "false")

import time
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st

# ============================================================
# PAGE CONFIG (must be the first Streamlit call)
# ============================================================

st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# YOUR ORIGINAL CODE (logic unchanged)
# ============================================================

from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph, START, END

from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CACHE_DIR = os.path.join(BASE_DIR, "faiss_cache")


# step 1 - Building the Rag Retrievers
# (models and indexes are cached so Streamlit does not reload them on every click)

@st.cache_resource(show_spinner=False)
def load_embeddings():

    print("Loading embedding model...")

    _t = time.time()

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"Embedding model loaded in {time.time() - _t:.1f}s\n")

    return emb


embeddings = None  # filled in by the loading block near the bottom


def load_pdf(pdf_path: str):

    """Loads a PDF page by page with progress. Uses PyMuPDF (fast) if installed."""

    try:
        import fitz  # noqa: F401  (comes from the pymupdf package)
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(pdf_path)
        print("  Using PyMuPDF loader (fast)")
    except ImportError:
        loader = PyPDFLoader(pdf_path)
        print("  Using PyPDF loader (slow) - run: pip install pymupdf")

    pages = []

    for i, page in enumerate(loader.lazy_load(), 1):
        pages.append(page)
        if i == 1 or i % 10 == 0:
            print(f"  ...loaded {i} pages")

    print(f"  Total pages loaded: {len(pages)}")

    # remove pages that have no text (blank pages or pure images)
    pages_with_text = [p for p in pages if p.page_content.strip()]

    if len(pages_with_text) < len(pages):
        print(f"  Skipped {len(pages) - len(pages_with_text)} pages with no text")

    return pages_with_text


def build_retriver(pdf_path: str):

    if not os.path.exists(pdf_path):
        folder = os.path.dirname(pdf_path)
        available = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\nPDFs in this folder: {available}"
        )

    name = os.path.splitext(os.path.basename(pdf_path))[0]

    stat = os.stat(pdf_path)

    # cache name changes automatically if the PDF is replaced or edited
    index_path = os.path.join(CACHE_DIR, f"{name}_{stat.st_size}_{int(stat.st_mtime)}")

    print(f"[{name}]")

    # reuse the saved index if it exists (instant start)
    if os.path.exists(index_path):
        print("  Found saved index, loading it (skipping PDF processing)")
        vectorstore = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )
        return vectorstore.as_retriever(search_kwargs={"k": 4})

    t = time.time()

    print("  Step 1/3: reading PDF...")

    document = load_pdf(pdf_path)

    print("  Step 2/3: splitting into chunks...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    chunks = splitter.split_documents(document)

    print(f"  Created {len(chunks)} chunks")

    if not chunks:
        raise ValueError(
            f"No extractable text found in '{pdf_path}'. "
            f"It may be a scanned/image-based PDF - try OCR'ing it first."
        )

    print("  Step 3/3: creating embeddings (this is the slow part on CPU)...")

    vectorstore = None

    BATCH = 64

    for i in range(0, len(chunks), BATCH):

        batch = chunks[i:i + BATCH]

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

        print(f"  ...embedded {min(i + BATCH, len(chunks))}/{len(chunks)} chunks")

    os.makedirs(CACHE_DIR, exist_ok=True)

    vectorstore.save_local(index_path)

    print(f"  Done in {time.time() - t:.1f}s (index saved for next time)\n")

    return vectorstore.as_retriever(search_kwargs={"k": 4})


@st.cache_resource(show_spinner=False)
def load_retrievers():

    academic = build_retriver(os.path.join(BASE_DIR, "studenthandbook1.pdf"))

    fee = build_retriver(os.path.join(BASE_DIR, "feestructure.pdf"))

    return academic, fee


@st.cache_resource(show_spinner=False)
def load_llm():

    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)


academic_retriever = None  # filled in by the loading block near the bottom

fee_retriever = None

llm = None


# Step 2 - State

class State(TypedDict):

    programme: str

    messages: Annotated[list, add_messages]

    query_type: str

    retrieved_context: str


# step 3 - Nodes generaation

def classifier_node(state: State) -> dict:

    """ look at the latest user message and decide which path to take. """

    last_message = state['messages'][-1].content

    prompt = (
        "classify the following student query into exactly one category: "
        " 'academic' , 'fee', or 'general'. \n\n"

        "use 'academic' for questions about attendance, exams, gradings, credits, "
        "promotion, course structure summer training or degree requirements.\n\n"

        "use 'fee' for questions about tution payment, refund, late charges, scholorships or "
        "or any money related topic \n\n"

        "use 'general' for greetings, casual talk or anything not related the college rules or "
        "fee.\n\n"

        f"query: {last_message}\n\n"

        "return only one word: academic, fee or general."
    )

    response = llm.invoke(prompt)

    category = response.content.strip().lower()

    if "academic" in category:

        category = "academics"

    elif "fee" in category:

        category = "fee"

    else:

        category = "general"

    return {"query_type": category}


def academic_rag_node(state: State) -> dict:

    """ Retrieves relevant chunks from the academics handbook. """

    query = state['messages'][-1].content

    docs = academic_retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    return {"retrieved_context": context}


def fee_rag_node(state: State) -> dict:

    """ Retrieves relevant chunks from the fee structure pdf."""

    query = state['messages'][-1].content

    docs = fee_retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    return {"retrieved_context": context}


def general_mode(state: State) -> dict:

    """Answers directly using the llm's own knowedge, no retrieval needed."""

    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:

    """generates the final answer, personalized using the student's programme."""

    query = state["messages"][-1].content

    programme = state.get("programme", "unknown")

    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":

        prompt = (
            f"you are a friendly college aassistant talkingto a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )

    else:

        prompt = (
            f"you are a college assistant helping a {programme} student. "
            f"use the following context from offical college documents to answer."
            f"the question accurately, if the context mentions specify figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly and precise answer."
        )

    response = llm.invoke(prompt)

    return {"messages": [("ai", response.content.strip())]}


# STEP 4 - ROUTER FUNCTION

def route_query(state: State):

    if state['query_type'] == 'academics':

        return "academic_rag"

    elif state['query_type'] == "fee":

        return "fee_rag"

    else:

        return "general"


# Step 5 - Building the Graph
# (wrapped in a cached function so it is built only once)

@st.cache_resource(show_spinner=False)
def build_app():

    graph = StateGraph(State)

    graph.add_node("classifier", classifier_node)

    graph.add_node("academic_rag", academic_rag_node)

    graph.add_node("fee_rag", fee_rag_node)

    graph.add_node("general", general_mode)

    graph.add_node("response", response_node)


    # Edges

    graph.add_edge(START, "classifier")

    graph.add_conditional_edges("classifier", route_query)

    graph.add_edge("academic_rag", "response")

    graph.add_edge("fee_rag", "response")

    graph.add_edge("general", "response")

    graph.add_edge("response", END)

    return graph.compile()


programme_map = {

    "1": "BBA",

    "2": "BS Computer Science",

    "3": "BS Software Engineering",

    "4": "BS Artificial Intelligence",

    "5": "BS Business Administration",

    "6": "BS Accounting and Finance",

    "7": "BS Media Sciences",

    "8": "BS Psychology"

}

# ============================================================
# UI CONFIG
# ============================================================

ROUTE_INFO = {
    "academics": ("📘", "Academic", "Student Handbook", "route-academic"),
    "fee": ("💳", "Fee", "Fee Structure", "route-fee"),
    "general": ("💬", "General", "General knowledge", "route-general"),
}

SUGGESTIONS = [
    ("📘 Attendance rules", "What is the attendance policy?"),
    ("💳 Fee refund", "What is the fee refund policy?"),
    ("📚 Promotion criteria", "What are the criteria for promotion to the next semester?"),
    ("👋 Say hello", "Hi! What can you help me with?"),
]

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.6rem; padding-bottom: 6rem; max-width: 820px; }

    .hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #4f46e5 55%, #9333ea 100%);
        padding: 1.8rem 1.8rem; border-radius: 18px; color: white;
        margin-bottom: 1.1rem; box-shadow: 0 10px 30px rgba(79, 70, 229, .28);
    }
    .hero h1 { margin: 0; font-size: 1.9rem; font-weight: 800; line-height: 1.2; }
    .hero p  { margin: .4rem 0 0 0; opacity: .92; font-size: 1rem; }
    .hero .who {
        display: inline-block; margin-top: .8rem; padding: 4px 14px; border-radius: 999px;
        background: rgba(255,255,255,.18); font-size: .85rem; font-weight: 600;
    }

    .route {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: .76rem; font-weight: 700; margin-bottom: .45rem;
    }
    .route-academic { background: #dbeafe; color: #1e40af; }
    .route-fee      { background: #dcfce7; color: #166534; }
    .route-general  { background: #f3e8ff; color: #6b21a8; }

    .welcome {
        border: 1px dashed rgba(128,128,128,.4); border-radius: 16px;
        padding: 1.2rem 1.3rem; margin-bottom: 1rem;
        background: rgba(128,128,128,.05);
    }
    .welcome h3 { margin: 0 0 .3rem 0; }
    .welcome p  { margin: 0; opacity: .75; }

    .stButton > button {
        border-radius: 12px; font-weight: 600; transition: all .2s ease;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(0,0,0,.15); }

    /* ---------- Mobile / small screens ---------- */
    @media (max-width: 640px) {
        .block-container { padding-left: .8rem; padding-right: .8rem; padding-top: 1rem; }
        .hero { padding: 1.2rem 1.1rem; border-radius: 14px; }
        .hero h1 { font-size: 1.4rem; }
        .hero p  { font-size: .92rem; }
        .welcome { padding: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD MODELS AND DOCUMENTS (cached, runs once)
# ============================================================

if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ `GROQ_API_KEY` not found. Add it to your `.env` file and restart the app.")
    st.stop()

try:
    with st.spinner("Loading AI models and college documents... the first run can take a few minutes."):
        embeddings = load_embeddings()
        academic_retriever, fee_retriever = load_retrievers()
        llm = load_llm()
        app = build_app()
except Exception as exc:
    st.error(f"⚠️ Setup failed: {exc}")
    st.info("Check that `studenthandbook1.pdf` and `feestructure.pdf` are in the same folder as this file.")
    st.stop()

# ============================================================
# SESSION STATE
# ============================================================


def init_state():
    if "chat" not in st.session_state:
        st.session_state.chat = []   # [{role, content, query_type, context}]
    if "programme_key" not in st.session_state:
        st.session_state.programme_key = "2"
    if "pending" not in st.session_state:
        st.session_state.pending = None


init_state()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🎓 College Assistant")
    st.caption("Ask about academics, fees or anything else.")
    st.divider()

    st.markdown("### 👤 Your programme")
    st.selectbox(
        "Select your programme",
        options=list(programme_map.keys()),
        format_func=lambda k: programme_map[k],
        key="programme_key",
        label_visibility="collapsed",
    )
    student_programme = programme_map[st.session_state.programme_key]
    st.success(f"Set as **{student_programme}** student")

    st.divider()

    asked = [m for m in st.session_state.chat if m["role"] == "user"]
    routes = [m.get("query_type") for m in st.session_state.chat if m["role"] == "assistant"]

    st.markdown("### 📊 This session")
    c1, c2 = st.columns(2)
    c1.metric("Questions", len(asked))
    c2.metric("From documents", sum(1 for r in routes if r in ("academics", "fee")))

    st.divider()

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()

    with st.expander("ℹ️ How it works"):
        st.markdown(
            "Each question is **classified** and sent to the right place:\n\n"
            "- 📘 **Academic** → searches the Student Handbook\n"
            "- 💳 **Fee** → searches the Fee Structure\n"
            "- 💬 **General** → answered directly\n\n"
            "Answers for document questions use only the passages found in the college documents."
        )

# ============================================================
# MAIN
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <h1>🎓 College Assistant</h1>
        <p>Instant answers from your student handbook and fee structure.</p>
        <span class="who">👤 {student_programme}</span>
    </div>
    """,
    unsafe_allow_html=True,
)


def render_assistant_extras(msg: dict):
    """Route badge above the answer, source passages below it."""
    info = ROUTE_INFO.get(msg.get("query_type"))
    if info:
        icon, label, source, css = info
        st.markdown(
            f'<span class="route {css}">{icon} {label} · {source}</span>',
            unsafe_allow_html=True,
        )


def render_sources(msg: dict):
    ctx = msg.get("context", "")
    if ctx and ctx != "NO_RETRIEVAL_NEEDED":
        with st.expander("📄 Source passages used"):
            st.text(ctx)


# ---------- Welcome + suggestions ----------
if not st.session_state.chat:
    st.markdown(
        """
        <div class="welcome">
            <h3>👋 Welcome!</h3>
            <p>Ask me about attendance, exams, promotion, fees or refunds — or tap a suggestion below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, (label, question) in enumerate(SUGGESTIONS):
        if cols[i % 2].button(label, key=f"sugg_{i}", use_container_width=True):
            st.session_state.pending = question
            st.rerun()

# ---------- History ----------
for msg in st.session_state.chat:
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant":
            render_assistant_extras(msg)
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg)

# ---------- Input ----------
typed = st.chat_input("Ask about attendance, exams, fees...")

user_query = st.session_state.pending or typed
st.session_state.pending = None

if user_query:
    st.session_state.chat.append({"role": "user", "content": user_query})

    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Thinking..."):
            try:
                result = app.invoke({

                    "programme": student_programme,

                    'messages': [("human", user_query)]

                })

                reply = {
                    "role": "assistant",
                    "content": result['messages'][-1].content,
                    "query_type": result.get("query_type", "general"),
                    "context": result.get("retrieved_context", ""),
                }

                render_assistant_extras(reply)
                st.markdown(reply["content"])
                render_sources(reply)

                st.session_state.chat.append(reply)

            except Exception as exc:
                msg = f"Sorry, something went wrong: {exc}"
                st.error(msg)
                st.session_state.chat.append(
                    {"role": "assistant", "content": msg, "query_type": None, "context": ""}
                )

    st.rerun()