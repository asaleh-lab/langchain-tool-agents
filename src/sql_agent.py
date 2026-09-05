"""Ask natural language questions over the same checkouts in SQLite."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

load_dotenv()

root = Path(__file__).resolve().parents[1]
db_path = root / "data" / "library.sqlite"

# Same twelve rows as checkouts.csv, now behind SQL
db = SQLDatabase.from_uri(f"sqlite:///{db_path.as_posix()}")

model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=model, temperature=0)

agent = create_sql_agent(
    llm,
    db=db,
    agent_type="tool-calling",
    verbose=True,
)

# Same ask as dataframe_agent.py; only the store changes
question = (
    "Sum late_fee_usd by category. Which category has the highest total? "
    "Answer with the category name and the total dollars."
)
print(f"Question: {question}\n")

result = agent.invoke({"input": question})
print(f"\nAnswer:\n{result['output']}")
