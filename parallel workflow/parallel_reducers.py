
import os

from typing import TypedDict, Annotated

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

initial_state = {

    "raw_text": sample_script,

    "safety_scores": {}  # initializing empty dictionary

}

final_state = app.invoke(initial_state)

print(final_state["safety_scores"])



# .\venv\Scripts\Activate.ps1
