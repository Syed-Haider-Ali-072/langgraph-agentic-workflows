import os

# model pehle se download ho chuka hai, is liye internet check band (atakne se bachata hai)
# agar kabhi model dobara download karna ho to ye line hata do
os.environ["HF_HUB_OFFLINE"] = "1"

import time
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

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

print("Loading embedding model...")

_t = time.time()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print(f"Embedding model loaded in {time.time() - _t:.1f}s\n")


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


academic_retriever = build_retriver(os.path.join(BASE_DIR, "studenthandbook1.pdf"))

fee_retriever = build_retriver(os.path.join(BASE_DIR, "feestructure.pdf"))


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)

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

app = graph.compile()


# STEP 6 - RUNNING THE CODE

print("Welcome to the college assistant \n\n")

print("Which programme are you in ")

print("1. BBA")

print("2. BS Computer Science")

print("3. BS Software Engineering")

print("4. BS Artificial Intelligence")

print("5. BS Business Administration")

print("6. BS Accounting and Finance")

print("7. BS Media Sciences")

print("8. BS Psychology")


choice = input("\nEnter 1,2,3,4,5,6,7 or 8.....").strip()

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

while choice not in programme_map:

    choice = input("Invalid choice. Please enter a number from 1 to 8: ").strip()

student_programme = programme_map[choice]

print(f"\nGreat! You're set as a {student_programme} student.")

while True:

    user_query = input("You: ")

    if user_query.lower() in ["exit", 'quit']:

        break

    result = app.invoke({

        "programme": student_programme,

        'messages': [("human", user_query)]

    })

    print(f"Assistant : {result['messages'][-1].content} ")