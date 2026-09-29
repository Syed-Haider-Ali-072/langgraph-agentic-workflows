# 🔗 Sequential Workflow: Raw Text to Video Script to Hinglish

An **agentic AI workflow built with LangGraph** that turns rough text into a polished, ready-to-record video script and then into natural **Hinglish**, using a **three-stage pipeline** where each stage builds on the output of the one before it.

---

## ✨ Features

- 🔗 **Sequential pipeline**: editor → scriptwriter → translator, one after another
- ✍️ **Stage 1 (Editor)**: fixes grammar and spelling, smooths the flow, keeps the message
- 🎬 **Stage 2 (Scriptwriter)**: rewrites the clean text as an engaging, conversational video script
- 🌐 **Stage 3 (Translator)**: adapts the script into natural, flowing Hinglish for a Pakistani audience
- 🧩 **Typed shared state** that carries every intermediate result
- ⚡ **Fast inference** using Groq (`openai/gpt-oss-120b`)
- 🖥️ **Two ways to use it**: terminal or Streamlit web app

---

## 🧠 How It Works

```mermaid
flowchart LR
    A([START]) --> B[Editor]
    B --> C[Scriptwriter]
    C --> D[Translator]
    D --> E([END])
```

| Stage | Node | Reads from state | Writes to state | Job |
|-------|------|------------------|-----------------|-----|
| 1 | `editor` | `raw_input` | `edited_text` | Clean up grammar, typos and tone |
| 2 | `scriptwriter` | `edited_text` | `script_text` | Turn the text into a punchy video script |
| 3 | `translator` | `script_text` | `final_output` | Convert the script to natural Hinglish |

### 🗂️ Shared State

Every node reads from and writes to one typed dictionary, so data flows down the pipeline:

```python
class pipelinestate(TypedDict):
    raw_input: str
    edited_text: str
    script_text: str
    final_output: str
```

### 🔌 Graph Wiring

Edges connect the nodes in a straight line:

```python
graph.add_edge(START, "editor")
graph.add_edge("editor", "scriptwriter")
graph.add_edge("scriptwriter", "translator")
graph.add_edge("translator", END)
```

### 📝 Example Input

```
Ai agents are future of the tech. They can think, plan and act on their
own. Langgraph helps you build these agents with proper control and memory.
```

The pipeline returns an edited version, a video script, and the final Hinglish text.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Workflow orchestration |
| [LangChain](https://github.com/langchain-ai/langchain) | LLM abstractions |
| [Groq](https://groq.com/) (`langchain-groq`) | LLM inference for all three stages |
| [Streamlit](https://streamlit.io/) | Web user interface |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Loading API keys from `.env` |

---

## 📁 Project Structure

```
.
├── sequential_workflow.py    # CLI version (rename to match your file)
├── pipeline_ui.py            # Streamlit web UI
├── requirements.txt          # Python dependencies
├── .env                      # API key (NOT committed to Git)
├── .gitignore
├── sc/                       # Screenshots used in this README
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create and activate a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```txt
langgraph
langchain-core
langchain-groq
streamlit
python-dotenv
```

### 4. Add your API key

Create a file named `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your key here: https://console.groq.com/keys

> ⚠️ **Never push your `.env` file to GitHub.** Add it to `.gitignore` (see below).

---

## ▶️ Usage

### Option A: Terminal (CLI)

```bash
python sequential_workflow.py
```

The script runs the built-in sample text through all three stages and prints the final Hinglish output. To test your own text, change the value of `raw_input` in the `app.invoke(...)` call.

### Option B: Streamlit Web UI

```bash
streamlit run pipeline_ui.py
```

The web app opens in your browser at `http://localhost:8501`.

**UI highlights**
- Paste any text and click **Run Pipeline**
- Three stage cards (Editor, Scriptwriter, Translator) that switch from *Waiting* to *Done*
- Each stage result appears as soon as that stage finishes
- Progress bar, word counter, Reset button
- Download all three outputs as one `.txt` file

---

## 📸 Demo

### 💻 Code

![Code screenshot 1](sc/Screenshot%202026-09-28%20135151.png)

![Code screenshot 2](sc/Screenshot%202026-09-28%20135219.png)

![Code screenshot 3](sc/Screenshot%202026-09-28%20135308.png)

![Code screenshot 4](sc/Screenshot%202026-09-28%20135417.png)

![Code screenshot 5](sc/Screenshot%202026-09-28%20135436.png)

### 🖥️ Terminal (CLI)

![Terminal screenshot 1](sc/Screenshot%202026-09-28%20135614.png)

![Terminal screenshot 2](sc/Screenshot%202026-09-28%20135630.png)

![Terminal screenshot 3](sc/Screenshot%202026-09-28%20135651.png)

### 🌐 Streamlit UI

![Streamlit UI screenshot 1](sc/Screenshot%202026-09-28%20135725.png)

![Streamlit UI screenshot 2](sc/Screenshot%202026-09-28%20135725%20copy.png)

![Streamlit UI screenshot 3](sc/Screenshot%202026-09-28%20135839.png)

---

## ⚙️ Configuration

| Setting | Where | Default |
|---------|-------|---------|
| Model | `ChatGroq(model=...)` | `openai/gpt-oss-120b` |
| Temperature | `ChatGroq(..., temperature=0.7)` | `0.7` (creative, so results vary between runs) |
| Input text | `raw_input` (CLI) or text box (UI) | Built-in sample |

---

## 🧩 Extending the Pipeline

Adding a fourth stage (for example a *title and hashtag generator*) takes four steps:

1. Add a new key to `pipelinestate`, such as `title_text`.
2. Write a node that reads `final_output` and returns `{"title_text": ...}`.
3. Register it with `graph.add_node(...)`.
4. Move the last edge: connect `translator` to the new node, and the new node to `END`.

---

## ⚠️ Notes

- **Each stage depends on the one before it.** A weak result early in the pipeline carries through to the end.
- **Results vary between runs** because the temperature is `0.7`. Lower it for more consistent output.
- **No retry logic.** If an API call fails (for example a rate limit), the run stops with an error.

---

## 🧯 Troubleshooting

| Problem | Fix |
|---------|-----|
| `429 Rate limit exceeded` | Wait a minute and try again, or check your plan and usage in the Groq console. |
| `GROQ_API_KEY` not found | Make sure `.env` is in the folder you run the script from, and the key name matches exactly. |
| Images not showing on GitHub | Keep screenshots in the `sc/` folder and commit them together with the README. Use `%20` for spaces in file names (or rename files without spaces). |

---

## 🔒 Recommended `.gitignore`

```gitignore
# Environment and secrets
.env

# Python
venv/
__pycache__/
*.pyc
```

---

## 🗺️ Roadmap

- [ ] Choose the output language (Hinglish, Urdu, English) from the UI
- [ ] Adjustable tone for the scriptwriter (formal, funny, energetic)
- [ ] Retry logic for failed API calls
- [ ] Save results to a file or database

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Syed Haider Ali Shah**
GitHub: [@Syed-Haider-Ali-072](https://github.com/Syed-Haider-Ali-072)


⭐ If you found this project useful, consider giving it a star!