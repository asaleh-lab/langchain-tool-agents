"""Bind checkout lookup tools to a chat model and print its tool call."""

import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

root = Path(__file__).resolve().parents[1]
csv_path = root / "data" / "checkouts.csv"


def _read_checkouts() -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@tool
def member_late_fees(member: str) -> str:
    """Return total late fees in USD for a library member (partial name match)."""
    needle = member.lower()
    total = 0.0
    hits = []
    for row in _read_checkouts():
        if needle in row["member"].lower():
            fee = float(row["late_fee_usd"])
            total += fee
            hits.append(f"{row['tool_name']}: ${fee:.2f}")
    if not hits:
        return f"No checkouts found for {member!r}."
    lines = [f"Late fees for {member}: ${total:.2f}"] + hits
    return "\n".join(lines)


@tool
def longest_checkout_for_tool(tool_name: str) -> str:
    """Return who kept a tool out the longest (partial tool name match)."""
    needle = tool_name.lower()
    matches = [row for row in _read_checkouts() if needle in row["tool_name"].lower()]
    if not matches:
        return f"No checkouts found for tool {tool_name!r}."
    longest = max(matches, key=lambda row: int(row["days_out"]))
    return (
        f"{longest['member']} kept {longest['tool_name']} for "
        f"{longest['days_out']} days (checkout {longest['checkout_id']})."
    )


@tool
def checkouts_over_days(days: int) -> str:
    """List checkouts still out longer than the given number of days."""
    overdue = [row for row in _read_checkouts() if int(row["days_out"]) > days]
    if not overdue:
        return f"No checkouts over {days} days."
    lines = [f"Checkouts over {days} days:"]
    for row in sorted(overdue, key=lambda r: int(r["days_out"]), reverse=True):
        lines.append(
            f"  {row['member']} — {row['tool_name']} — {row['days_out']} days"
        )
    return "\n".join(lines)


tools = [member_late_fees, longest_checkout_for_tool, checkouts_over_days]
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=model, temperature=0)
llm_with_tools = llm.bind_tools(tools)

question = "Who kept the circular saw out the longest?"
print(f"Question: {question}\n")

response = llm_with_tools.invoke(question)

if response.tool_calls:
    for call in response.tool_calls:
        print(f"Tool: {call['name']}")
        print(f"Args: {call['args']}")
else:
    print("No tool call returned.")
    print(response.content)
