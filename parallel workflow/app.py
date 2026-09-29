import os

from typing import TypedDict, Annotated

import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Content Safety Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# YOUR ORIGINAL CODE (unchanged)
# ============================================================

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)


def merge_score_dicts(existing: dict, newupdate: dict) -> dict:

    if existing is None:

        return newupdate

    return {**existing, **newupdate}


# creating a state

class AnalyzerState(TypedDict):

    raw_text: str

    safety_scores: Annotated[dict[str, int], merge_score_dicts]


# nodes

def toxicity_node(state: AnalyzerState) -> dict:

    print("\n [Branch 1] Analyzing Toxicity and Hate Speech..")

    prompt = (
        "Analyze the following text for profanity, aggression, hate speech or toxicity. "
        "Provide a score from 0 to 100 where 0 means perfectly clean and 100 means high "
        "risk. Return only the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    response = llm.invoke(prompt)

    try:

        score = int(response.content.strip())

    except ValueError:

        score = 0

    # return a sub dictionary under our single state key

    return {"safety_scores": {"toxicity_level": score}}


def copyright_node(state: AnalyzerState) -> dict:

    print("\n [Branch 2] Analyzing Copyright Risk..")

    prompt = (
        "Analyze the following text. Judge if it sounds heavily plagiarized, unoriginal "
        "or represents a corporate trademark risk. "
        "Provide a score from 0 to 100 where 0 means perfectly clean and 100 means high "
        "risk. Return only the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    response = llm.invoke(prompt)

    try:

        score = int(response.content.strip())

    except ValueError:

        score = 0

    # return a sub dictionary under our single state key

    return {"safety_scores": {"copyright_risk": score}}


def cultural_node(state: AnalyzerState) -> dict:

    print("\n [Branch 3] Analyzing regional & cultural sensitivity..")

    prompt = (
        "Analyze the following text for regional sensitivities, political landmines "
        "or cultural insensitivity that might offend a global audience. "
        "Provide a score from 0 to 100 where 0 means perfectly clean and 100 means high "
        "risk. Return only the plain integer number, nothing else.\n\n"
        f"Text:\n{state['raw_text']}"
    )

    response = llm.invoke(prompt)

    try:

        score = int(response.content.strip())

    except ValueError:

        score = 0

    # return a sub dictionary under our single state key

    return {"safety_scores": {"cultural_insensitivity": score}}


builder = StateGraph(AnalyzerState)


builder.add_node("toxicity_node", toxicity_node)

builder.add_node("copyright_check", copyright_node)

builder.add_node("cultural_node", cultural_node)

builder.add_edge(START, "toxicity_node")

builder.add_edge(START, "copyright_check")

builder.add_edge(START, "cultural_node")


builder.add_edge("toxicity_node", END)

builder.add_edge("copyright_check", END)

builder.add_edge("cultural_node", END)

app = builder.compile()

sample_script = """
you guys welcome to the stream today i am going to show how to hack into your friend's
system using a script. i copied the script directly from an online forum without checking
the original author or the license.

before we continue, remember that this is only a demonstration for educational purposes.
you should never access someone else's computer or account without their permission.

the script contains some aggressive language and makes a few jokes about people from
different countries and cultures. we will also discuss why copying code from online
forums without proper attribution can create copyright and originality concerns.

today's goal is to understand these risks and learn how to use technology responsibly.
"""

# ============================================================
# UI CONFIG
# ============================================================

# key in safety_scores -> (icon, title, description)
CHECKS = {
    "toxicity_level": ("☣️", "Toxicity & Hate Speech", "Profanity, aggression, hate speech"),
    "copyright_risk": ("©️", "Copyright Risk", "Plagiarism, unoriginal content, trademarks"),
    "cultural_insensitivity": ("🌍", "Cultural Sensitivity", "Regional and political sensitivities"),
}


def risk_level(score: int):
    """Returns (label, color, background) for a 0-100 risk score."""
    if score < 34:
        return "Low Risk", "#166534", "#dcfce7"
    if score < 67:
        return "Medium Risk", "#92400e", "#fef3c7"
    return "High Risk", "#991b1b", "#fee2e2"


def bar_color(score: int) -> str:
    if score < 34:
        return "#22c55e"
    if score < 67:
        return "#f59e0b"
    return "#ef4444"


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1200px; }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #0ea5e9 100%);
        padding: 2.2rem 2rem; border-radius: 18px; color: white;
        margin-bottom: 1.5rem; box-shadow: 0 10px 30px rgba(14, 165, 233, .25);
    }
    .hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; }
    .hero p  { margin: .4rem 0 0 0; opacity: .9; font-size: 1.05rem; }

    .score-card {
        border: 1px solid rgba(128,128,128,.25); border-radius: 16px;
        padding: 1.1rem 1.2rem; background: rgba(128,128,128,.05);
        height: 100%;
    }
    .score-head { display:flex; justify-content:space-between; align-items:center; }
    .score-title { font-weight: 700; font-size: 1.02rem; }
    .score-desc  { font-size: .8rem; opacity: .65; margin-bottom: .6rem; }
    .score-value { font-size: 2.6rem; font-weight: 800; line-height: 1; margin: .3rem 0 .6rem 0; }
    .score-value small { font-size: 1rem; font-weight: 500; opacity: .55; }

    .bar-bg { background: rgba(128,128,128,.2); border-radius: 999px; height: 10px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 999px; transition: width .6s ease; }

    .pill {
        display:inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: .75rem; font-weight: 700;
    }
    .pill-wait { background:#e5e7eb; color:#4b5563; }
    .pill-run  { background:#dbeafe; color:#1e40af; }

    .verdict {
        border-radius: 16px; padding: 1.3rem 1.5rem; margin-top: 1.2rem;
        display:flex; align-items:center; gap: 1rem;
    }
    .verdict h3 { margin: 0; font-size: 1.3rem; }
    .verdict p  { margin: .2rem 0 0 0; opacity: .85; }

    .stButton > button {
        border-radius: 12px; font-weight: 700; padding: .65rem 1.2rem;
        border: none; transition: all .2s ease;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1e3a8a, #0ea5e9); color: white;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(0,0,0,.18); }
    </style>
    """,
    unsafe_allow_html=True,
)


def score_card_html(key: str, score=None) -> str:
    icon, title, desc = CHECKS[key]

    if score is None:
        pill = '<span class="pill pill-run">Analyzing…</span>' if st.session_state.get("running") else '<span class="pill pill-wait">Waiting</span>'
        value = '<div class="score-value">—<small> / 100</small></div>'
        bar = '<div class="bar-bg"><div class="bar-fill" style="width:0%"></div></div>'
    else:
        label, color, bg = risk_level(score)
        pill = f'<span class="pill" style="background:{bg};color:{color}">{label}</span>'
        value = f'<div class="score-value" style="color:{bar_color(score)}">{score}<small> / 100</small></div>'
        bar = f'<div class="bar-bg"><div class="bar-fill" style="width:{score}%;background:{bar_color(score)}"></div></div>'

    return f"""
    <div class="score-card">
        <div class="score-head"><span class="score-title">{icon} {title}</span>{pill}</div>
        <div class="score-desc">{desc}</div>
        {value}
        {bar}
    </div>
    """


def verdict_html(scores: dict) -> str:
    avg = round(sum(scores.values()) / len(scores))
    worst_key = max(scores, key=scores.get)
    worst = scores[worst_key]
    _, worst_title, _ = CHECKS[worst_key]

    if worst >= 67:
        icon, title, bg, color = "🚫", "Not safe to publish", "#fee2e2", "#991b1b"
        msg = f"High risk detected in <b>{worst_title}</b> ({worst}/100). Review and revise before publishing."
    elif worst >= 34:
        icon, title, bg, color = "⚠️", "Publish with caution", "#fef3c7", "#92400e"
        msg = f"Moderate concern in <b>{worst_title}</b> ({worst}/100). A quick review is recommended."
    else:
        icon, title, bg, color = "✅", "Looks safe to publish", "#dcfce7", "#166534"
        msg = "No significant risks detected across all checks."

    return f"""
    <div class="verdict" style="background:{bg};color:{color}">
        <div style="font-size:2.4rem">{icon}</div>
        <div>
            <h3>{title} <span style="opacity:.7;font-size:.9rem">· average risk {avg}/100</span></h3>
            <p>{msg}</p>
        </div>
    </div>
    """


def init_state():
    defaults = {
        "text": sample_script.strip(),
        "scores": {},
        "running": False,
        "error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ============================================================
# MAIN
# ============================================================


def main():
    init_state()

    st.markdown(
        """
        <div class="hero">
            <h1>🛡️ Content Safety Analyzer</h1>
            <p>Three AI checks run in parallel — toxicity, copyright and cultural sensitivity — for a fast, complete safety report.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not os.getenv("GROQ_API_KEY"):
        st.warning("⚠️ `GROQ_API_KEY` not found. Add it to your `.env` file and restart the app.")

    left, right = st.columns([1, 1.25], gap="large")

    # ---------- Input ----------
    with left:
        st.subheader("📄 Content to Analyze")
        text = st.text_area(
            "Paste your script or text",
            value=st.session_state.text,
            height=340,
            label_visibility="collapsed",
            placeholder="Paste the script or text you want to check...",
        )
        st.session_state.text = text
        st.caption(f"{len(text.split())} words · {len(text)} characters")

        c1, c2, c3 = st.columns([1.4, 1, 1])
        run = c1.button("🔍 Analyze", type="primary", use_container_width=True)
        sample = c2.button("📋 Sample", use_container_width=True)
        clear = c3.button("♻️ Clear", use_container_width=True)

        if sample:
            st.session_state.text = sample_script.strip()
            st.session_state.scores = {}
            st.rerun()
        if clear:
            st.session_state.text = ""
            st.session_state.scores = {}
            st.rerun()

        with st.expander("ℹ️ How it works"):
            st.markdown(
                "Your text is sent to **3 branches at the same time**:\n"
                "- ☣️ Toxicity & hate speech\n"
                "- ©️ Copyright risk\n"
                "- 🌍 Cultural sensitivity\n\n"
                "Each returns a score from **0 (clean)** to **100 (high risk)**. "
                "The results are merged into one report."
            )

    # ---------- Results ----------
    with right:
        st.subheader("📊 Safety Report")
        cols = st.columns(3)
        slots = {}
        for col, key in zip(cols, CHECKS):
            slots[key] = col.empty()

        verdict_slot = st.empty()

        def render_all():
            for key in CHECKS:
                slots[key].markdown(
                    score_card_html(key, st.session_state.scores.get(key)),
                    unsafe_allow_html=True,
                )
            if len(st.session_state.scores) == len(CHECKS):
                verdict_slot.markdown(verdict_html(st.session_state.scores), unsafe_allow_html=True)
            else:
                verdict_slot.empty()

        if not run:
            st.session_state.running = False
            render_all()

    # ---------- Run ----------
    if run:
        if not text.strip():
            st.error("Please enter some text first.")
            st.stop()

        st.session_state.scores = {}
        st.session_state.error = None
        st.session_state.running = True

        with right:
            render_all()
            progress = st.progress(0, text="Running 3 checks in parallel...")

            try:
                initial_state = {
                    "raw_text": text,
                    "safety_scores": {}  # initializing empty dictionary
                }

                # same compiled graph, streamed so each branch shows as soon as it finishes
                done = 0
                for step in app.stream(initial_state):
                    for node_name, update in step.items():
                        st.session_state.scores.update(update.get("safety_scores", {}))
                        done += 1
                    render_all()
                    progress.progress(min(done / len(CHECKS), 1.0), text=f"{done} of {len(CHECKS)} checks complete")

                st.session_state.running = False
                progress.empty()
                render_all()
                st.toast("Analysis complete!", icon="✅")
            except Exception as exc:
                st.session_state.running = False
                progress.empty()
                st.error(f"Something went wrong: {exc}")

    # ---------- Download ----------
    if len(st.session_state.scores) == len(CHECKS):
        report = "CONTENT SAFETY REPORT\n" + "=" * 24 + "\n"
        for key, (_, title, _) in CHECKS.items():
            score = st.session_state.scores[key]
            report += f"{title}: {score}/100 ({risk_level(score)[0]})\n"
        st.download_button(
            "⬇️ Download report (.txt)",
            data=report,
            file_name="safety_report.txt",
            mime="text/plain",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()