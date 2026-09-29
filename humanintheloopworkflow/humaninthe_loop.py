import os

from typing import TypedDict, Annotated, Literal

from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph, START, END


from langgraph.prebuilt import ToolNode


from langchain_tavily import TavilySearch

from langchain_groq import ChatGroq

from langgraph.types import interrupt, Command

from langgraph.checkpoint.memory import MemorySaver

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

app = graph.compile(checkpointer=checkpointer)


print("=" * 55)

print("Welcome to the LinkedIn Post Generator")

print("=" * 55)

print("This tool will draft a LinkedIn post, pause for your review,")

print("and improve the post when you reject it.")

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

    config = {

        "configurable": {

            "thread_id": "linkedin-human-review"

        }

    }

    final_state = app.invoke(initial_state, config=config)

    while "__interrupt__" in final_state:

        interrupt_data = final_state["__interrupt__"][0].value

        print("\n" + "=" * 55)

        print("HUMAN REVIEW REQUIRED")

        print("=" * 55)

        print(interrupt_data)

        print("=" * 55)

        print("Example: APPROVE")

        print("Example: REJECT: Make the hook stronger and shorten paragraph 2.")

        human_response = input("\nYour decision: ").strip()

        final_state = app.invoke(

            Command(resume=human_response),

            config=config

        )


    print("\n" + "=" * 55)

    print("Final LinkedIn Post")

    print("=" * 55)

    print(final_state["draft"])

    print("=" * 55)

    print(f"Total attempts: {final_state['attempt']}")

    print(f"Approved: {final_state['is_approved']}")