import os
import time

from typing import TypedDict, Annotated

import streamlit as st

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

PROGRAMME_MAP = {
    "1": "BBA",
    "2": "BS Computer Science",
    "3": "BS Software Engineering",
    "4": "BS Artificial Intelligence",
    "5": "BS Business Administration",
    "6": "BS Accounting and Finance",
    "7": "BS Media Sciences",
    "8": "BS Psychology",
}

ACADEMIC_PDF = "StudentHandBook.pdf"
FEE_PDF = "feestructur.pdf"

QUERY_TYPE_LABELS = {
    "academics": "📘 Academic",
    "fee": "💳 Fee",
    "general": "💬 General",
}

# ============================================================
# STEP 1 - RAG RETRIEVERS (same logic as original script)
# ============================================================


@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_retriver(pdf_path: str, _embeddings):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(document)
    vectorstore = FAISS.from_documents(chunks, _embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


@st.cache_resource(show_spinner=False)
def load_retrievers():
    embeddings = load_embeddings()
    academic_retriever = build_retriver(ACADEMIC_PDF, embeddings)
    fee_retriever = build_retriver(FEE_PDF, embeddings)
    return academic_retriever, fee_retriever


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0.4)


# ============================================================
# STEP 2 - STATE
# ============================================================


class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


# ============================================================
# STEP 3 - NODES (same logic as original script)
# ============================================================


def make_classifier_node(llm):
    def classifier_node(state: State) -> dict:
        """look at the latest user message and decide which path to take."""
        last_message = state["messages"][-1].content

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

    return classifier_node


def make_academic_rag_node(academic_retriever):
    def academic_rag_node(state: State) -> dict:
        """Retrieves relevant chunks from the academics handbook."""
        query = state["messages"][-1].content
        docs = academic_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    return academic_rag_node


def make_fee_rag_node(fee_retriever):
    def fee_rag_node(state: State) -> dict:
        """Retrieves relevant chunks from the fee structure pdf."""
        query = state["messages"][-1].content
        docs = fee_retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return {"retrieved_context": context}

    return fee_rag_node


def general_mode(state: State) -> dict:
    """Answers directly using the llm's own knowedge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def make_response_node(llm):
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
                f"you are a college student helping a {programme} student. "
                f"use the following context from offical college documents to answer."
                f"the question accurately, if the context mentions specify figures for "
                f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Give a clear, friendly and precise answer."
            )

        response = llm.invoke(prompt)
        return {"messages": [("ai", response.content.strip())]}

    return response_node


# ============================================================
# STEP 4 - ROUTER FUNCTION
# ============================================================


def route_query(state: State):
    if state["query_type"] == "academics":
        return "academic_rag"
    elif state["query_type"] == "fee":
        return "fee_rag"
    else:
        return "general"


# ============================================================
# STEP 5 - BUILD THE GRAPH
# ============================================================


@st.cache_resource(show_spinner=False)
def build_app():
    academic_retriever, fee_retriever = load_retrievers()
    llm = load_llm()

    graph = StateGraph(State)

    graph.add_node("classifier", make_classifier_node(llm))
    graph.add_node("academic_rag", make_academic_rag_node(academic_retriever))
    graph.add_node("fee_rag", make_fee_rag_node(fee_retriever))
    graph.add_node("general", general_mode)
    graph.add_node("response", make_response_node(llm))

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", route_query)
    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")
    graph.add_edge("response", END)

    return graph.compile()


# ============================================================
# STEP 6 - STREAMLIT UI
# ============================================================


def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of dicts: role, content, query_type
    if "programme" not in st.session_state:
        st.session_state.programme = None
    if "app_ready" not in st.session_state:
        st.session_state.app_ready = False
    if "load_error" not in st.session_state:
        st.session_state.load_error = None


def sidebar_ui():
    with st.sidebar:
        st.markdown("## 🎓 College Assistant")
        st.caption("Ask about academics, fees, or anything else.")
        st.divider()

        st.markdown("### Your Programme")
        choice_label = st.selectbox(
            "Select your programme",
            options=list(PROGRAMME_MAP.keys()),
            format_func=lambda k: f"{k}. {PROGRAMME_MAP[k]}",
            index=list(PROGRAMME_MAP.keys()).index(
                [k for k, v in PROGRAMME_MAP.items() if v == st.session_state.programme][0]
            )
            if st.session_state.programme in PROGRAMME_MAP.values()
            else 1,
            label_visibility="collapsed",
        )
        new_programme = PROGRAMME_MAP[choice_label]

        if new_programme != st.session_state.programme:
            st.session_state.programme = new_programme

        st.success(f"Set as **{st.session_state.programme}** student")

        st.divider()
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.divider()
        with st.expander("ℹ️ How it works"):
            st.write(
                "Your question is classified as **academic**, **fee**, or "
                "**general**, then answered using the relevant college "
                "document (or general knowledge for casual queries)."
            )


def render_chat_history():
    for msg in st.session_state.chat_history:
        avatar = "🧑‍🎓" if msg["role"] == "user" else "🎓"
        with st.chat_message(msg["role"], avatar=avatar):
            if msg["role"] == "assistant" and msg.get("query_type"):
                st.caption(QUERY_TYPE_LABELS.get(msg["query_type"], ""))
            st.markdown(msg["content"])


def main():
    init_session_state()
    if st.session_state.programme is None:
        st.session_state.programme = PROGRAMME_MAP["2"]

    sidebar_ui()

    st.title("🎓 College Assistant")
    st.caption(f"Chatting as a **{st.session_state.programme}** student")

    # Load the graph app (retrievers + LLM), once, with clear feedback.
    if not st.session_state.app_ready and st.session_state.load_error is None:
        with st.spinner("Loading knowledge base and models... this may take a moment on first run."):
            try:
                for path, label in [(ACADEMIC_PDF, "Student Handbook"), (FEE_PDF, "Fee Structure")]:
                    if not os.path.exists(path):
                        raise FileNotFoundError(
                            f"Could not find '{path}' ({label}). "
                            f"Place it in the app's working directory and restart."
                        )
                st.session_state.compiled_app = build_app()
                st.session_state.app_ready = True
            except Exception as exc:
                st.session_state.load_error = str(exc)

    if st.session_state.load_error:
        st.error(f"⚠️ Setup error: {st.session_state.load_error}")
        if st.button("🔄 Retry"):
            st.session_state.load_error = None
            st.rerun()
        st.stop()

    render_chat_history()

    user_query = st.chat_input("Ask about attendance, fees, exams, or anything else...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(user_query)

        with st.chat_message("assistant", avatar="🎓"):
            placeholder = st.empty()
            placeholder.markdown("Thinking... ▌")
            try:
                result = st.session_state.compiled_app.invoke(
                    {
                        "programme": st.session_state.programme,
                        "messages": [("human", user_query)],
                    }
                )
                answer = result["messages"][-1].content
                query_type = result.get("query_type", "general")

                placeholder.empty()
                st.caption(QUERY_TYPE_LABELS.get(query_type, ""))
                st.markdown(answer)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer, "query_type": query_type}
                )
            except Exception as exc:
                placeholder.empty()
                error_msg = f"Sorry, something went wrong: {exc}"
                st.error(error_msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg, "query_type": None}
                )


if __name__ == "__main__":
    main()