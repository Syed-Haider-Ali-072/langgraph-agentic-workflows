import os

from typing import TypedDict, Annotated, Literal

from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph, START, END

from langgraph.prebuilt import ToolNode

from langchain_groq import ChatGroq

from langchain_tavily import TavilySearch

# from langchain_community.document_loaders import PyPDFLoader

# from langchain_text_splitters import RecursiveCharacterTextSplitter

# from langchain_huggingface import HuggingFaceEmbeddings

# from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

# tools

search_tool = TavilySearch(max_results=3)

tools = [search_tool]


# llms

writer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)

writer_llm_with_tools = writer_llm.bind_tools(tools)

# reviewer

reviewer_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2)


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

If you have already received feedback on a previous draft, carefully address

every point from the feedback in the new draft.

Rules for good LinkedIn posts:

- Start with a strong hook in the first line.

- Provide a clear takeaway.

- Make the post easy to skim.

- Keep it around 150-200 words.

- End with a question or call to action to invite engagement.

- Do not use hashtags.

"""


def writer_node(state: State) -> dict:

    """writes the linked in post. can call tavily search first."""

    # if the last message is a tool result, this is the same attempt continuing
    is_tool_followup = bool(state["messages"]) and getattr(state["messages"][-1], "type", "") == "tool"

    attempt = state.get("attempt", 0) + (0 if is_tool_followup else 1)

    topic = state["topic"]

    previous_feedback = state["review_feedback"]

    if attempt == 1:

        user_message = (

            f"write a linkedin post on the topic {topic}"

            f"if you need current info search on the web"

        )

    else:

        user_message = (

            f"your previous draft on '{topic}' was rejected"

            f"Here is the reviewer's feedback \n\n {previous_feedback}\n\n"

            f"write a new improved draft that fixes every issue mentioned"

            f"do not repeat the same mistake"

        )

    messages = [("system", WRITER_SYSTEM_PROMPT)]

    if state["messages"]:

        messages.extend(state["messages"])

    if not is_tool_followup:

        messages.append(("human", user_message))

    response = writer_llm_with_tools.invoke(messages)

    if is_tool_followup:

        return {

            "messages": [response],

            "attempt": attempt

        }

    return {

        "messages": [("human", user_message), response],

        "attempt": attempt

    }


tool_node = ToolNode(tools)


def extract_draft_node(state: State):

    """After the writer finishes tool calls, pulls the final text out as the draft."""

    last_message = state['messages'][-1]

    draft = last_message.content

    print(f"\n\nGenerated post \n {draft} \n ")

    return {"draft": draft}


reviewer_system_prompt = (

    "You are a strict LinkedIn content reviewer. You judge whether a post is "

    "publish-ready. Evaluate the post against these criteria:\n\n"

    "1. Strong hook in the first line.\n"

    "2. One clear and valuable takeaway.\n"

    "3. Easy to skim - uses short paragraphs.\n"

    "4. Around 150-200 words.\n"

    "5. Ends with a question or call to action.\n"

    "6. Professional and engaging tone.\n"

    "7. No hashtags.\n\n"

    "Respond in exactly this format:\n\n"

    "Verdict: approve or rejected\n"

    "Feedback: <one short paragraph explaining why>\n\n"

    "Be strict but fair. Approve only if the post genuinely meets all "

    "criteria. Reject if even one criterion is clearly missing."

)


def reviewer_node(state: State) -> dict:

    """Reviews the draft and decides: approve or reject with feedback."""

    draft = state['draft']

    prompt = (

        f"review this linkedin post draft: \n"

        f"{draft}\n"

        f"give your review"

    )

    response = reviewer_llm.invoke(

        [("system", reviewer_system_prompt), ("human", prompt)]

    )

    review_text = response.content.strip()

    is_approved = "APPROVE" in review_text.upper().split("FEEDBACK")[0]

    if "FEEDBACK:" in review_text.upper():

        feedback = review_text[review_text.upper().index("FEEDBACK:") + len("FEEDBACK:"):].strip()

    else:

        feedback = review_text

    verdict = "APPROVED" if is_approved else "REJECTED"

    print(f"[Verdict: {verdict}]")

    print(f"[Feedback: {feedback}]")

    return {

        "review_feedback": feedback,

        "is_approved": is_approved,

    }


# router function

def should_use_tool(state: State):

    last_message = state['messages'][-1]

    if getattr(last_message, 'tool_calls', None):

        return "tools"

    return "extract_draft"


def should_stop_looping(state: State):

    if state['is_approved']:

        print("Post has been approved")

        return END

    if state['attempt'] >= 3:

        print("Reached max attempts")

        return END

    return "writer"


# build the graph

graph = StateGraph(State)

graph.add_node("writer", writer_node)

graph.add_node("tools", tool_node)

graph.add_node("extract_draft", extract_draft_node)

graph.add_node("reviewer", reviewer_node)

graph.add_edge(START, "writer")

graph.add_conditional_edges(

    "writer", should_use_tool,

)

graph.add_edge("tools", "writer")

graph.add_edge("extract_draft", "reviewer")

graph.add_conditional_edges(

    "reviewer", should_stop_looping

)

app = graph.compile()


print("=" * 55)

print("Welcome to the LinkedIn post generator")

print("=" * 55)

print("This tool will draft a LinkedIn post for you, review it")

print("itself and iterate until it's publish-ready")

print("=" * 55)


topic = input("\nWhat do you want a LinkedIn post about? ").strip()

if not topic:

    print("No topic given. Exiting")

else:

    print("Starting generation......\n")

    initial_state = {

        "topic": topic,

        "messages": [],

        "draft": "",

        "review_feedback": "",

        "is_approved": False,

        "attempt": 0

    }

    final_state = app.invoke(initial_state, config={"recursion_limit": 25})

    print("\n" + "=" * 55)

    print("Final LinkedIn post")

    print("=" * 55)

    print(final_state["draft"])

    print("=" * 55)

    print(f"Total attempts: {final_state['attempt']}")

    print(f"Approved: {final_state['is_approved']}")