# langchain-tool-agents

Borrow & Build is a neighborhood tool library. Members check out ladders, saws, and garden gear for a few days at a time. We keep a checkout log in `data/checkouts.csv` and the same rows in `data/library.sqlite`.

A plain LLM call can guess from a prompt. An agent can reason, pick a tool, and read the log. These scripts start with small custom tools, then chains, then agents over the CSV and the database.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Put your OpenAI API key in `.env`.

## Let's bind custom tools and print the model's tool call

```powershell
python src/custom_tools.py
```

## Let's pipe a prompt through an LCEL chain

```powershell
python src/lcel_chain.py
```

## Let's validate the tool call, then run it ourselves

```powershell
python src/manual_tool_call.py
```
