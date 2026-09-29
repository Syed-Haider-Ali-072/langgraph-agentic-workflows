import os
import re
import uuid

from typing import TypedDict, Annotated, Literal

import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="LinkedIn Post Studio",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# YOUR ORIGINAL CODE (unchanged)
# ============================================================

from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph, START, END


from langgraph.prebuilt import ToolNode


from langchain_tavily import TavilySearch

from langchain_groq import ChatGroq

from langgraph.types import interrupt, Command

from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()


# tools

search_tool = TavilySearch(max_results=3)

tools = [search_tool]


# llms

writer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

writer_llm_with_tools = writer_llm.bind_tools(tools)


# state building

class State(TypedDict):

    topic: str

    messages: Annotated[list, add_messages]

    draft: str

    review_feedback: str

    is_approved: bool

    attempt: int


# nodes

WRITER_SYSTEM_PROMPT = """

You are an expert LinkedIn content writer. Your job is to write an engaging,

professional LinkedIn post about the given topic.

If the topic requires up-to-date information, statistics, or current trends,

use the web search tool to gather fresh context before writing.

If you have already received feedback from a human reviewer on a previous draft,

carefully address every point from the feedback in the new draft.

Rules for good LinkedIn posts:

- Start with a strong hook in the first line.

- Provide one clear and valuable takeaway.

- Make the post easy to skim using short paragraphs.

- Keep it around 150-200 words.

- End with a question or call to action to invite engagement.

- Use a professional and engaging tone.

- Do not use hashtags.

- Return only the LinkedIn post without explaining your writing process.

"""


def writer_node(state: State) -> dict:

    """Writes the LinkedIn post. Can call Tavily search first."""

    attempt = state.get("attempt", 0) + 1

    topic = state["topic"]

    previous_feedback = state["review_feedback"]

    if attempt == 1:

        user_message = (

            f"Write a LinkedIn post on the topic: {topic}\n\n"

            f"If you need current information, search on the web first."

        )

    else:

        user_message = (

            f"Your previous draft on '{topic}' was rejected by the human reviewer.\n\n"

            f"Here is the reviewer's feedback:\n\n{previous_feedback}\n\n"

            f"Write a new improved draft that fixes every issue mentioned.\n"

            f"Do not repeat the same mistakes."

        )

    messages = [("system", WRITER_SYSTEM_PROMPT)]

    if state["messages"]:

        messages.extend(state["messages"])

    messages.append(("human", user_message))

    response = writer_llm_with_tools.invoke(messages)

    return {

        "messages": [("human", user_message), response],

        "attempt": attempt

    }


tool_node = ToolNode(tools)


def extract_draft_node(state: State):

    """After the writer finishes tool calls, pulls the final text out as the draft."""

    last_message = state["messages"][-1]

    draft = last_message.content

    print(f"\n\nGenerated post\n{draft}\n")

    return {"draft": draft}


HUMAN_REVIEW_PROMPT = (

    "You are the human reviewer in a LinkedIn content workflow.\n\n"

    "Review the generated post against these criteria:\n\n"

    "1. Strong hook in the first line.\n"

    "2. One clear and valuable takeaway.\n"

    "3. Easy to skim - uses short paragraphs.\n"

    "4. Around 150-200 words.\n"

    "5. Ends with a question or call to action.\n"

    "6. Professional and engaging tone.\n"

    "7. No hashtags.\n\n"

    "Type APPROVE if the post is publish-ready.\n"

    "Type REJECT: followed by your feedback if the post needs changes."

)


def human_review_node(state: State) -> dict:

    """Pauses the workflow and waits for human approval or feedback."""

    draft = state["draft"]

    review_request = (

        f"{HUMAN_REVIEW_PROMPT}\n\n"

        f"Generated LinkedIn Post:\n\n"

        f"{draft}\n\n"

        f"Your decision:"

    )

    human_response = interrupt(review_request)

    human_response = str(human_response).strip()

    is_approved = human_response.lower().startswith("approve")

    if is_approved:

        feedback = "Human reviewer approved the post."

    else:

        if ":" in human_response:

            feedback = human_response.split(":", 1)[1].strip()

        else:

            feedback = human_response

    verdict = "APPROVED" if is_approved else "REJECTED"

    print(f"[Verdict: {verdict}]")

    print(f"[Feedback: {feedback}]")

    return {

        "review_feedback": feedback,

        "is_approved": is_approved,

    }


# router function

def should_use_tool(state: State):

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):

        return "tools"

    return "extract_draft"


def should_stop_looping(state: State):

    if state["is_approved"]:

        print("Post has been approved")

        return END

    if state["attempt"] >= 3:

        print("Reached max attempts")

        return END

    return "writer"


# build the graph
# (wrapped in a cached function so the MemorySaver survives Streamlit reruns,
#  otherwise the paused workflow would be lost after every click)

@st.cache_resource(show_spinner=False)
def build_app():

    graph = StateGraph(State)

    graph.add_node("writer", writer_node)

    graph.add_node("tools", tool_node)

    graph.add_node("extract_draft", extract_draft_node)

    graph.add_node("human_review", human_review_node)

    graph.add_edge(START, "writer")

    graph.add_conditional_edges(

        "writer", should_use_tool,

    )

    graph.add_edge("tools", "writer")

    graph.add_edge("extract_draft", "human_review")

    graph.add_conditional_edges(

        "human_review", should_stop_looping

    )

    checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)


app = build_app()

MAX_ATTEMPTS = 3

# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1150px; }

    .hero {
        background: linear-gradient(135deg, #0a66c2 0%, #0e7490 55%, #14b8a6 100%);
        padding: 2.2rem 2rem; border-radius: 18px; color: white;
        margin-bottom: 1.5rem; box-shadow: 0 10px 30px rgba(10, 102, 194, .28);
    }
    .hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; }
    .hero p  { margin: .4rem 0 0 0; opacity: .92; font-size: 1.05rem; }

    .steps { display:flex; gap:.6rem; margin-bottom:1.2rem; flex-wrap:wrap; }
    .step {
        flex:1; min-width:150px; text-align:center; padding:.6rem .8rem;
        border-radius:12px; font-weight:600; font-size:.9rem;
        background: rgba(128,128,128,.12); color: rgba(128,128,128,.9);
    }
    .step.active { background:#0a66c2; color:white; box-shadow:0 4px 14px rgba(10,102,194,.35); }
    .step.done   { background:#dcfce7; color:#166534; }

    .post-card {
        border:1px solid rgba(128,128,128,.28); border-radius:16px;
        padding:1.3rem 1.5rem; background: rgba(128,128,128,.05);
        white-space: pre-wrap; line-height:1.6; font-size:1.02rem;
    }
    .chip {
        display:inline-block; padding:3px 12px; border-radius:999px;
        font-size:.78rem; font-weight:700; margin-right:6px;
    }
    .chip-ok   { background:#dcfce7; color:#166534; }
    .chip-warn { background:#fef3c7; color:#92400e; }
    .chip-bad  { background:#fee2e2; color:#991b1b; }
    .chip-info { background:#dbeafe; color:#1e40af; }

    .stButton > button {
        border-radius: 12px; font-weight: 700; padding: .65rem 1.2rem;
        border: none; transition: all .2s ease;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0a66c2, #14b8a6); color: white;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(0,0,0,.18); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================


def init_state():
    defaults = {
        "phase": "idle",          # idle -> review -> done
        "thread_id": str(uuid.uuid4()),
        "topic": "",
        "draft": "",
        "attempt": 0,
        "is_approved": False,
        "history": [],            # [{attempt, draft, verdict, feedback}]
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_all():
    st.session_state.phase = "idle"
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.topic = ""
    st.session_state.draft = ""
    st.session_state.attempt = 0
    st.session_state.is_approved = False
    st.session_state.history = []


def get_config():
    return {"configurable": {"thread_id": st.session_state.thread_id}}


def handle_result(result):
    """Reads the graph result and moves the UI to the right phase."""
    st.session_state.draft = result.get("draft", st.session_state.draft)
    st.session_state.attempt = result.get("attempt", st.session_state.attempt)
    st.session_state.is_approved = result.get("is_approved", False)

    if "__interrupt__" in result:
        st.session_state.phase = "review"
    else:
        st.session_state.phase = "done"


def start_generation(topic: str):
    initial_state = {
        "topic": topic,
        "messages": [],
        "draft": "",
        "review_feedback": "",
        "is_approved": False,
        "attempt": 0
    }
    result = app.invoke(initial_state, config=get_config())
    handle_result(result)


def submit_decision(decision: str, verdict: str, feedback: str):
    st.session_state.history.append(
        {
            "attempt": st.session_state.attempt,
            "draft": st.session_state.draft,
            "verdict": verdict,
            "feedback": feedback,
        }
    )
    result = app.invoke(Command(resume=decision), config=get_config())
    handle_result(result)


def analyze_post(text: str):
    """Quick automatic checks that mirror the reviewer criteria."""
    words = len(text.split())
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    has_hashtag = bool(re.search(r"#\w+", text))
    ends_with_q = text.strip().endswith("?") or "?" in text.strip().split("\n")[-1]
    return {
        "words": words,
        "word_ok": 130 <= words <= 230,
        "hashtag": has_hashtag,
        "paragraphs": len(paragraphs),
        "skim_ok": len(paragraphs) >= 3,
        "cta": ends_with_q,
    }


def chips_html(a: dict) -> str:
    def chip(ok, text):
        return f'<span class="chip {"chip-ok" if ok else "chip-warn"}">{"✓" if ok else "!"} {text}</span>'

    return (
        chip(a["word_ok"], f'{a["words"]} words')
        + chip(a["skim_ok"], f'{a["paragraphs"]} paragraphs')
        + chip(not a["hashtag"], "No hashtags" if not a["hashtag"] else "Hashtags found")
        + chip(a["cta"], "Ends with question" if a["cta"] else "No question / CTA")
    )


def steps_html(phase: str) -> str:
    order = ["idle", "review", "done"]
    labels = ["1 · Topic", "2 · Review draft", "3 · Final post"]
    out = '<div class="steps">'
    for i, (key, label) in enumerate(zip(order, labels)):
        cls = "active" if key == phase else ("done" if order.index(phase) > i else "")
        out += f'<div class="step {cls}">{label}</div>'
    return out + "</div>"


# ============================================================
# MAIN
# ============================================================


def main():
    init_state()
    phase = st.session_state.phase

    st.markdown(
        """
        <div class="hero">
            <h1>💼 LinkedIn Post Studio</h1>
            <p>AI drafts your post, you stay in control — approve it or send feedback and get a better version.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    missing = [k for k in ("GROQ_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
    if missing:
        st.warning(f"⚠️ Missing in your `.env`: {', '.join(missing)}. Add them and restart the app.")

    st.markdown(steps_html(phase), unsafe_allow_html=True)

    left, right = st.columns([1.6, 1], gap="large")

    # ------------------------------------------------------------
    # RIGHT: status + history
    # ------------------------------------------------------------
    with right:
        st.subheader("📌 Status")
        if phase == "idle":
            st.info("Enter a topic to begin.")
        else:
            st.markdown(
                f'<span class="chip chip-info">Attempt {st.session_state.attempt} of {MAX_ATTEMPTS}</span>',
                unsafe_allow_html=True,
            )
            st.progress(min(st.session_state.attempt / MAX_ATTEMPTS, 1.0))
            st.caption(f"**Topic:** {st.session_state.topic}")

        if st.session_state.history:
            st.subheader("🕘 Review history")
            for h in reversed(st.session_state.history):
                icon = "✅" if h["verdict"] == "APPROVED" else "❌"
                with st.expander(f'{icon} Draft {h["attempt"]} — {h["verdict"].title()}'):
                    if h["verdict"] == "REJECTED":
                        st.markdown(f"**Your feedback:** {h['feedback']}")
                        st.divider()
                    st.markdown(h["draft"])

        if phase != "idle":
            st.divider()
            if st.button("♻️ Start a new post", use_container_width=True):
                reset_all()
                st.rerun()

    # ------------------------------------------------------------
    # LEFT: main workflow area
    # ------------------------------------------------------------
    with left:

        # ---------- PHASE 1: topic ----------
        if phase == "idle":
            st.subheader("✏️ What should the post be about?")
            topic = st.text_area(
                "Topic",
                value=st.session_state.topic,
                height=130,
                label_visibility="collapsed",
                placeholder="e.g. How AI agents are changing software development in 2026",
            )

            st.caption("💡 Tip: be specific — the more detail you give, the better the first draft.")

            if st.button("🚀 Generate post", type="primary", use_container_width=True):
                if not topic.strip():
                    st.error("Please enter a topic first.")
                else:
                    st.session_state.topic = topic.strip()
                    with st.spinner("Researching and writing your post..."):
                        try:
                            start_generation(st.session_state.topic)
                        except Exception as exc:
                            st.error(f"Something went wrong: {exc}")
                            st.stop()
                    st.rerun()

        # ---------- PHASE 2: human review ----------
        elif phase == "review":
            st.subheader(f"👀 Review draft {st.session_state.attempt}")

            draft = st.session_state.draft
            st.markdown(chips_html(analyze_post(draft)), unsafe_allow_html=True)
            st.markdown(f'<div class="post-card">{draft}</div>', unsafe_allow_html=True)

            with st.expander("📋 Review criteria"):
                st.markdown(
                    "1. Strong hook in the first line\n"
                    "2. One clear and valuable takeaway\n"
                    "3. Easy to skim — short paragraphs\n"
                    "4. Around 150–200 words\n"
                    "5. Ends with a question or call to action\n"
                    "6. Professional and engaging tone\n"
                    "7. No hashtags"
                )

            st.markdown("### Your decision")
            feedback = st.text_area(
                "Feedback (needed only if you reject)",
                height=110,
                placeholder="e.g. Make the hook stronger and shorten paragraph 2.",
                key=f"feedback_{st.session_state.attempt}",
            )

            c1, c2 = st.columns(2)
            approve = c1.button("✅ Approve", type="primary", use_container_width=True)
            reject = c2.button("❌ Reject & improve", use_container_width=True)

            if approve:
                with st.spinner("Finalizing..."):
                    try:
                        submit_decision("APPROVE", "APPROVED", "Human reviewer approved the post.")
                    except Exception as exc:
                        st.error(f"Something went wrong: {exc}")
                        st.stop()
                st.rerun()

            if reject:
                if not feedback.strip():
                    st.error("Please write feedback so the AI knows what to fix.")
                else:
                    with st.spinner("Rewriting based on your feedback..."):
                        try:
                            submit_decision(f"REJECT: {feedback.strip()}", "REJECTED", feedback.strip())
                        except Exception as exc:
                            st.error(f"Something went wrong: {exc}")
                            st.stop()
                    st.rerun()

        # ---------- PHASE 3: final ----------
        else:
            if st.session_state.is_approved:
                st.success("🎉 Post approved and ready to publish!")
                st.subheader("✨ Final LinkedIn post")
            else:
                st.warning(
                    f"Reached the maximum of {MAX_ATTEMPTS} attempts. "
                    "Here is the latest draft — you can still copy and edit it manually."
                )
                st.subheader("📝 Latest draft")

            final_post = st.session_state.draft
            st.markdown(chips_html(analyze_post(final_post)), unsafe_allow_html=True)
            st.markdown(f'<div class="post-card">{final_post}</div>', unsafe_allow_html=True)

            st.markdown("#### 📎 Copy-ready text")
            st.code(final_post, language=None, wrap_lines=True)

            st.download_button(
                "⬇️ Download as .txt",
                data=final_post,
                file_name="linkedin_post.txt",
                mime="text/plain",
                use_container_width=True,
            )

            c1, c2 = st.columns(2)
            c1.metric("Total attempts", st.session_state.attempt)
            c2.metric("Approved", "Yes" if st.session_state.is_approved else "No")


if __name__ == "__main__":
    main()