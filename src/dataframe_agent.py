"""Ask natural language questions over the checkout CSV with a DataFrame agent."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

load_dotenv()

root = Path(__file__).resolve().parents[1]
csv_path = root / "data" / "checkouts.csv"
out_dir = root / "output"
out_dir.mkdir(exist_ok=True)
plot_path = out_dir / "late_fees_by_category.png"

df = pd.read_csv(csv_path)

model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=model, temperature=0)

# The agent writes and runs pandas/matplotlib code against df.
# allow_dangerous_code is required for that local Python REPL.
agent = create_pandas_dataframe_agent(
    llm,
    df,
    agent_type="tool-calling",
    verbose=True,
    allow_dangerous_code=True,
)

question = (
    "Sum late_fee_usd by category. Which category has the highest total? "
    f"Save a bar chart of totals to {plot_path.as_posix()} and answer with "
    "the category name and the total dollars."
)
print(f"Question: {question}\n")

result = agent.invoke({"input": question})
print(f"\nAnswer:\n{result['output']}")
if plot_path.exists():
    print(f"\nSaved {plot_path}")
else:
    print(f"\nPlot not found at {plot_path}")
