"""Pipe a checkout summary through an LCEL prompt | model | parser chain."""

import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

root = Path(__file__).resolve().parents[1]
csv_path = root / "data" / "checkouts.csv"

with csv_path.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))

# local: one member's rows stuffed into the prompt as plain text
member = "Amina Lee"
lines = [
    f"{row['tool_name']} ({row['category']}): {row['days_out']} days, "
    f"${row['late_fee_usd']} late"
    for row in rows
    if row["member"] == member
]
checkout_text = "\n".join(lines)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer from the checkout lines only. One short sentence.",
        ),
        (
            "human",
            "Member: {member}\n\nCheckouts:\n{checkouts}\n\nQuestion: {question}",
        ),
    ]
)

model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=model, temperature=0)

# LCEL: prompt fills, model replies, parser returns a string
chain = prompt | llm | StrOutputParser()

question = "How much does this member owe in late fees?"
print(f"Member: {member}")
print(f"Question: {question}\n")
print(chain.invoke({"member": member, "checkouts": checkout_text, "question": question}))
