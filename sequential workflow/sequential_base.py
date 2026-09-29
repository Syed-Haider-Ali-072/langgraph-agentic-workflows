import os
from typing import TypedDict

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

result = app.invoke({
    "raw_input": "Ai agents are future of the tech. They can think, plan and act on their "
                  "own. Langgraph helps you build these agents with proper control and memory."

})

# output
print("your result are : -\n\n")
print(result['final_output'])

# 1:28