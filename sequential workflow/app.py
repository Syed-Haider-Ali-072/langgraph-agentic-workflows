import os
from typing import TypedDict

import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Content Pipeline Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# YOUR ORIGINAL CODE (unchanged)
# ============================================================

# lets create the state

class pipelinestate(TypedDict):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

def editor_node(state: pipelinestate) -> dict:

    """stage 1: Clean up grammar. removes typos and refine the tone."""

    prompt = (
        "you are an expert copyeditor. clean up the following raw text. "
        "fix any grammatical errors, spelling mistakes and smooth out the transition flow "
        "while keeping the core message intact. Return only the edited text. \n\n"
        f"Text:\n{state['raw_input']}"
    )

    response = llm.invoke(prompt)

    return {"edited_text": response.content.strip()}


def scriptwriter_node(state: pipelinestate) -> dict:

    """stage 2: formats the clean text into an engaging video script style."""

    print("\n__ [stage 2] Executing scriptwriter node---")

    prompt = (
        "you are a charismatic YouTube content creator. take this edited text and transform "
        "it into a highly engaging punchy conversational video script hook. Make it sound like a real "
        "person speaking passionately. Return only the script content.\n\n"
        f"Edited Text:\n{state['edited_text']}"
    )

    response = llm.invoke(prompt)

    return {"script_text": response.content.strip()}


def translator_node(state: pipelinestate) -> dict:

    """stage 3: Translate the script into natural flowing Hinglish."""

    print("\n*--*- [Stage 3] Executing Hinglish Translator Node *--*-")

    prompt = (
        "You are an expert content localizer for a Pakistani audience. Take the following script "
        "and convert it into natural, flowing Hinglish. Do not simply translate it sentence by sentence "
        "or repeat the information. Adapt the language naturally, alternating comfortably between Hindi "
        "and English, the way an intellectual tech educator would naturally speak during a live stream. "
        "Keep the original meaning, context, and key technical terms intact. "
        "Return only the final Hinglish text.\n\n"
        f"Script:\n{state['script_text']}"
    )

    response = llm.invoke(prompt)

    return {"final_output": response.content.strip()}


# nodes and state  are ready now its time to create the graph and connecting the edges

from langgraph.graph import StateGraph, START, END

# Create the graph
graph = StateGraph(pipelinestate)

# adding nodes in in our graph

graph.add_node("editor", editor_node)
graph.add_node("scriptwriter", scriptwriter_node)
graph.add_node("translator", translator_node)

# adding edges sequential , one after another

graph.add_edge(START, "editor")
graph.add_edge('editor', "scriptwriter")
graph.add_edge('scriptwriter', "translator")
graph.add_edge('translator', END)

#compile the graph
app = graph.compile()

# ============================================================
# UI STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    }
    .hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; }
    .hero p  { margin: .4rem 0 0 0; opacity: .9; font-size: 1.05rem; }

    .stage-card {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        background: rgba(128,128,128,.05);
    }
    .stage-title { font-weight: 700; font-size: 1.05rem; margin-bottom: .5rem; }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 999px;
        font-size: .75rem; font-weight: 600; margin-left: 8px;
    }
    .badge-done    { background: #dcfce7; color: #166534; }
    .badge-pending { background: #e5e7eb; color: #4b5563; }

    .stButton > button {
        border-radius: 12px; font-weight: 700; padding: .65rem 1.2rem;
        border: none; transition: all .2s ease;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(0,0,0,.18); }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_TEXT = (
    "Ai agents are future of the tech. They can think, plan and act on their "
    "own. Langgraph helps you build these agents with proper control and memory."
)

STAGES = [
    ("editor", "✍️ Stage 1 — Editor", "edited_text"),
    ("scriptwriter", "🎬 Stage 2 — Scriptwriter", "script_text"),
    ("translator", "🌐 Stage 3 — Hinglish Translator", "final_output"),
]


def init_state():
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = DEFAULT_TEXT
    if "error" not in st.session_state:
        st.session_state.error = None


def render_stage(title, key, done):
    badge = (
        '<span class="badge badge-done">Done</span>'
        if done
        else '<span class="badge badge-pending">Waiting</span>'
    )
    st.markdown(
        f'<div class="stage-title">{title}{badge}</div>', unsafe_allow_html=True
    )


def main():
    init_state()

    # ---------- Hero ----------
    st.markdown(
        """
        <div class="hero">
            <h1>🎬 Content Pipeline Studio</h1>
            <p>Turn raw text into a polished video script and natural Hinglish — in three automated steps.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- API key check ----------
    if not os.getenv("GROQ_API_KEY"):
        st.warning("⚠️ `GROQ_API_KEY` not found. Add it to your `.env` file and restart the app.")

    left, right = st.columns([1, 1.15], gap="large")

    # ---------- Left: Input ----------
    with left:
        st.subheader("📝 Your Raw Text")
        raw_text = st.text_area(
            "Paste or type your text",
            value=st.session_state.raw_text,
            height=260,
            label_visibility="collapsed",
            placeholder="Type or paste the text you want to transform...",
        )
        st.session_state.raw_text = raw_text

        st.caption(f"{len(raw_text.split())} words · {len(raw_text)} characters")

        c1, c2 = st.columns(2)
        run = c1.button("🚀 Run Pipeline", type="primary", use_container_width=True)
        reset = c2.button("♻️ Reset", use_container_width=True)

        if reset:
            st.session_state.results = {}
            st.session_state.raw_text = DEFAULT_TEXT
            st.session_state.error = None
            st.rerun()

        with st.expander("ℹ️ How it works"):
            st.markdown(
                "1. **Editor** fixes grammar and spelling.\n"
                "2. **Scriptwriter** turns it into an engaging video script.\n"
                "3. **Translator** converts it into natural Hinglish."
            )

    # ---------- Right: Output ----------
    with right:
        st.subheader("✨ Results")
        slots = {}
        for node_name, title, key in STAGES:
            with st.container(border=True):
                done = key in st.session_state.results
                render_stage(title, key, done)
                slots[key] = st.empty()
                if done:
                    slots[key].markdown(st.session_state.results[key])
                else:
                    slots[key].caption("Output will appear here once this stage finishes.")

    # ---------- Run ----------
    if run:
        if not raw_text.strip():
            st.error("Please enter some text first.")
            st.stop()

        st.session_state.results = {}
        st.session_state.error = None
        progress = st.progress(0, text="Starting pipeline...")

        try:
            # same compiled graph, streamed so each stage shows as soon as it finishes
            for i, step in enumerate(app.stream({"raw_input": raw_text})):
                for node_name, update in step.items():
                    for k, v in update.items():
                        st.session_state.results[k] = v
                        slots[k].markdown(v)
                progress.progress(
                    (i + 1) / len(STAGES),
                    text=f"Completed stage {i + 1} of {len(STAGES)}",
                )
            progress.progress(1.0, text="✅ Pipeline complete")
            st.toast("Pipeline finished!", icon="🎉")
        except Exception as exc:
            progress.empty()
            st.session_state.error = str(exc)
            st.error(f"Something went wrong: {exc}")

    # ---------- Download ----------
    if "final_output" in st.session_state.results:
        st.divider()
        full_report = (
            "=== EDITED TEXT ===\n" + st.session_state.results.get("edited_text", "") + "\n\n"
            "=== VIDEO SCRIPT ===\n" + st.session_state.results.get("script_text", "") + "\n\n"
            "=== HINGLISH VERSION ===\n" + st.session_state.results.get("final_output", "")
        )
        st.download_button(
            "⬇️ Download all results (.txt)",
            data=full_report,
            file_name="pipeline_output.txt",
            mime="text/plain",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()