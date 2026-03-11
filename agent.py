import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()  # Load OPENAI_API_KEY from .env

# Define tools (functions yang agent bisa call)
@tool
def check_stock(item: str) -> str:
    """Cek stok item di e-commerce store."""
    # Mock: Asumsi stok ada
    return f"Stok untuk {item} tersedia: 10 unit."

@tool
def process_payment(amount: float) -> str:
    """Proses pembayaran mock untuk transaksi."""
    # Mock: Simulasi blockchain tx
    return f"Pembayaran {amount} USD berhasil via mock blockchain tx."

@tool
def confirm_order(item: str, amount: float) -> str:
    """Konfirmasi order setelah pembayaran."""
    return f"Order untuk {item} seharga {amount} USD dikonfirmasi."

# Setup LLM (gunakan OpenAI atau model lain)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Tools list
tools = [check_stock, process_payment, confirm_order]

# Prompt untuk agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "Kamu adalah AI agent untuk Agentic Commerce: handle transaksi otonom. Gunakan tools untuk cek stok, bayar, dan konfirmasi."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Buat agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run agent dengan input
input_query = "Beli 1 unit laptop seharga 1000 USD."
result = agent_executor.invoke({"input": input_query})
print(result['output'])
