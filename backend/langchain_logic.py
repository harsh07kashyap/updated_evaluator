# backend/langchain_logic.py


from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv(override=True)
import os
import re
import requests

print("DEBUG GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))


# -----------------------------
# Custom Embeddings
# -----------------------------
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False)

    def embed_query(self, text):
        return self.model.encode([text])[0]

# -----------------------------
# Prompt Template
# -----------------------------
template = """
You are an HR performance evaluator.
Given the following employee logs, evaluate the performance of the employee {employee_id}.

Criteria:
1. Productivity – tasks completed
2. Efficiency – balance of activities
3. Quality – fewer bugs/errors

Return a structured report with:
- Employee ID
- Summary
- Strengths
- Weaknesses
- Performance Rating (Excellent, Good, Needs Improvement)
- Suggestions
"Generate the report without extra blank lines. Use exactly one newline between paragraphs and no trailing spaces keeping it concise (max 200 words).

Logs:
{context}
"""

prompt = PromptTemplate(template=template, input_variables=["employee_id", "context"])

# -----------------------------
# LLM (Google Generative AI)
# -----------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=os.getenv("GOOGLE_API_KEY"))

# -----------------------------
# Main pipeline function
# -----------------------------
def evaluate_employee(employee_id: str, logs: str) -> str:
    """Generate performance evaluation for a given employee from logs."""

    # 1. Split logs into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=100)
    chunks = splitter.create_documents([logs])

    # 2. Build vector store
    embedding_function = SentenceTransformerEmbeddings()
    vector_store = FAISS.from_documents(chunks, embedding_function)

    # 3. Retrieve relevant chunks
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    query = f"Give all the logs for employee {employee_id}"
    retrieved_docs = retriever.invoke(query)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 4. Format final prompt
    final_prompt = prompt.invoke({"employee_id": employee_id, "context": context_text})

    # 5. Run LLM
    answer = llm.invoke(final_prompt)

    return answer.content


def clean_summary(text: str) -> str:
    # Remove multiple blank lines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    # Strip leading/trailing whitespace
    return text.strip()