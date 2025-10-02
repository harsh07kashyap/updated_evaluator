import streamlit as st
import json
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time
# from langchain_core.embeddings import Embeddings
# from sentence_transformers import SentenceTransformer
# from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv(override=True)
from pinecone import Pinecone, ServerlessSpec
import os
import re
import requests



# API_URL = "http://localhost:8000"
# api_endpoint_for_summary_generation = f"{API_URL}/generate_report"

# class SentenceTransformerEmbeddings(Embeddings):
#     def __init__(self, model_name="all-MiniLM-L6-v2"):
#         self.model = SentenceTransformer(model_name)

#     def embed_documents(self, texts):
#         return self.model.encode(texts, show_progress_bar=False)

#     def embed_query(self, text):
#         return self.model.encode([text])[0]

# @st.cache_resource  # Cache heavy models/resources once
# def load_embedding_model():
#     return SentenceTransformerEmbeddings("all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=os.getenv("GOOGLE_API_KEY"))
    
# Pinecone Init
# -----------------------------
pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name="automated-performance-evaluator-logs"
index = pc.Index(index_name)
  
    
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
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=os.getenv("GOOGLE_API_KEY"))

# -----------------------------
# Main pipeline function
# -----------------------------

def store_logs(employee_id: str, logs: str):
    """Store logs in Pinecone using MaaS (llama-text-embed-v2)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=100)
    chunks = splitter.split_text(logs)
    print(chunks)
    records = []
    for i, chunk in enumerate(chunks):
        records.append({
            "id": f"vec{i+1}",
            "text": chunk  # ✅ Must be "text", since Pinecone will embed this
        })
    
    if records:
        namespace = f"employee-{employee_id}"
        index.upsert_records(namespace=namespace, records=records)


def retrieve_logs(employee_id: str, query: str, top_k: int = 2) -> str:
    """Retrieve most relevant logs using Pinecone MaaS query."""
    results = index.search(
        namespace=f"employee-{employee_id}",
        # text=query,  # ✅ pass query text directly
        # top_k=top_k,
        # filter={"employee_id": {"$eq": employee_id}},
        # include_metadata=True
        query={
            "inputs": {"text": query},
            "top_k": top_k,
        }
    )
    print("Result is:", results)
    return "\n".join([m["metadata"].get("chunk_text", "") for m in results.get("matches", [])])


def evaluate_employee(employee_id: str, logs: str) -> str:
    """Generate performance evaluation for a given employee from logs."""

    # 1. Split logs into chunks
    # splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=100)
    # chunks = splitter.create_documents([logs])

    # # 2. Build vector store
    # embedding_function = load_embedding_model()
    # vector_store = FAISS.from_documents(chunks, embedding_function)

    # # 3. Retrieve relevant chunks
    # retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    # query = f"Give all the logs for employee {employee_id}"
    # retrieved_docs = retriever.invoke(query)
    # context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    store_logs(employee_id, logs)
    
    print("store_logs ran successfully.")

    # 2. Retrieve relevant chunks for employee
    query = f"Give all logs for employee {employee_id}"
    context_text = retrieve_logs(employee_id, query)

    # 4. Format final prompt
    final_prompt = prompt.invoke({"employee_id": employee_id, "context": context_text})

    # 5. Run LLM
    llm = load_llm()
    answer = llm.invoke(final_prompt)

    return answer.content

def clean_summary(text: str) -> str:
    # Remove multiple blank lines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    # Strip leading/trailing whitespace
    return text.strip()

def generate_report(req: dict) -> dict:
    """Generate employee report using LangChain pipeline"""
    raw_report = evaluate_employee(req["employee_name"], req["logs"])
    report = clean_summary(raw_report)
    return { "report": report}




# Page config
st.set_page_config(page_title="AI Performance Insights", page_icon="💡", layout="wide")

# Custom CSS for a classy look and orange button
st.markdown("""
    <style>
    /* Main container background */
    .main {
        background-color: #f0f2f6; /* Light gray background for a clean look */
    }
    
    /* Center and style the title */
    .stApp > header {
        background-color: #f0f2f6;
    }
    .main-header {
        text-align: left;
        color: #333333;
        font-weight: 600;
        padding-top: 10px;
        padding-bottom: 5px;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 20px;
    }
    
    /* Custom styling for the generate button (Orange) */
    div.stButton > button {
        background-color: #f7931e; /* Vibrant Orange */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        background-color: #e5820d; /* Darker orange on hover */
    }
    
    /* Style for the Evaluation Summary Box (Right Column) */
    .report-box {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        border-radius: 12px;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08); /* More pronounced shadow */
        min-height: 350px; /* Ensures the box has a minimum size */
        white-space: pre-wrap;
    }

    /* Input/Select box styling */
    .stTextArea textarea, .stTextInput input, .stSelectbox > div {
        border-radius: 8px;
        border: 1px solid #dcdcdc;
        padding: 10px;
        font-size: 14px;
    }
    
    </style>
""", unsafe_allow_html=True)

# Application Header (Top Bar Area)
st.markdown("## 💡 AI Performance Insights", unsafe_allow_html=True)
st.markdown("<div class='main-header'>Automated Employee Performance Evaluator</div>", unsafe_allow_html=True)


# --- PDF Generation Function ---
def create_pdf(summary_text: str, employee_id: str) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 40, f"Performance Evaluation for Employee: {employee_id}")
    c.line(50, height - 50, width - 50, height - 50)
    
    c.setFont("Helvetica", 12)
    
    # Write summary text line by line
    y = height - 80
    for line in summary_text.split("\n"):
        c.drawString(50, y, line)
        y -= 18
        if y < 50:  # New page if needed
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 12) # Reapply font for new page

    c.save()
    buffer.seek(0)
    return buffer.read()
# ------------------------------

# --- Main Two-Column Layout ---
col1, col2 = st.columns([1, 1.2]) # Input column slightly smaller than Output column

# LEFT COLUMN: Generate Evaluation (Input)
with col1:
    st.subheader("Generate Evaluation")
     # --- Sample Buttons ---
    col_btn1, col_btn2 = st.columns(2)
    default_logs=""
    
    sample_logs_1="""Employee E001 is a Data Scientist. 
On 11th September, they completed 11 tasks, made 5 commits, fixed 2 bugs, implemented 2 features, 
created 4 reports, attended 1 meeting, and worked actively for 6.1 hours. 
Manager’s review: Excellent. They also ran 2 experiments.

Employee E002 is a Software Engineer.
On 11th September, they completed 9 tasks, made 12 commits, fixed 5 bugs, implemented 3 features, reviewed 2 pull requests, attended 2 meetings, and worked actively for 7.4 hours.
Manager’s review: Very Good. They also refactored 1 module.

Employee E003 is a Business Analyst.
On 11th September, they completed 7 tasks, prepared 3 reports, conducted 2 requirement sessions, analyzed 1 dataset, attended 3 meetings, and worked actively for 6.0 hours.
Manager’s review: Good. They also created 1 presentation deck.

Employee E004 is a DevOps Engineer.
On 11th September, they completed 8 tasks, deployed 2 releases, fixed 3 CI/CD issues, wrote 4 infrastructure scripts, monitored 2 incidents, attended 1 meeting, and worked actively for 7.2 hours.
Manager’s review: Excellent. They also automated 1 backup routine.

Employee E005 is a QA Engineer.
On 11th September, they completed 10 tasks, executed 35 test cases, logged 6 bugs, verified 3 bug fixes, wrote 2 automation scripts, attended 1 meeting, and worked actively for 6.5 hours.
Manager’s review: Very Good. They also prepared 1 test plan."""

    sample_logs_2 = """Employee E006 is a UI/UX Designer.
On 12th September, they completed 6 tasks, created 3 wireframes, designed 2 prototypes, reviewed 1 design audit, collaborated on 2 feedback sessions, attended 2 meetings, and worked actively for 6.8 hours.
Manager’s review: Excellent. They also updated 1 design guideline document.

Employee E007 is a Data Scientist.
On 13th September, they completed 10 tasks, made 4 commits, built 1 machine learning model, fixed 2 data pipeline issues, prepared 2 reports, attended 1 meeting, and worked actively for 7.0 hours.
Manager’s review: Very Good. They also ran 3 experiments.

Employee E008 is a Software Engineer.
On 12th September, they completed 12 tasks, made 15 commits, fixed 4 bugs, implemented 2 features, reviewed 1 pull request, attended 2 meetings, and worked actively for 7.6 hours.
Manager’s review: Excellent. They also optimized 1 database query.

Employee E009 is a DevOps Engineer.
On 14th September, they completed 9 tasks, deployed 1 release, fixed 2 monitoring alerts, wrote 3 automation scripts, updated 1 server configuration, attended 2 meetings, and worked actively for 6.9 hours. Manager’s review: Good. They also tested 1 disaster recovery drill.

Employee E010 is a QA Engineer.
On 13th September, they completed 11 tasks, executed 42 test cases, logged 7 bugs, verified 4 bug fixes, created 2 regression test suites, attended 1 meeting, and worked actively for 6.3 hours.
Manager’s review: Very Good. They also updated 1 automation framework."""

    
    logs_input=""

    with col_btn1:
        if st.button("Insert Sample Logs 1"):
           default_logs=sample_logs_1
    with col_btn2:
        if st.button("Insert Sample Logs 2"):
           default_logs=sample_logs_2
    
   
     # Text Area (if no button pressed, it stays empty)
    logs_input = st.text_area(
        "Enter your company's employee performance logs or relevant data:",
        value=default_logs if default_logs else st.session_state.get("_logs_backup", ""),
        height=180,
        placeholder="e.g., Provide a balanced performance review..."
    )
    
     # Save current input as backup to survive re-runs
    st.session_state["_logs_backup"] = logs_input
    

    # Dropdown for Employee Name
    employees = ["E001", "E002", "E003", "E004","E005", "E006", "E007", "E008", "E009", "E010"]
    employee_id = st.selectbox("Select Employee:", employees)

    st.markdown("---")

    generate_btn = st.button("Generate Summary")


# RIGHT COLUMN: Evaluation Summary (Output)
with col2:
    st.subheader("Evaluation Summary")

    if generate_btn:
        if not logs_input:
            st.warning("⚠️ Please enter logs first.")
        else:
            with st.spinner(f"🔎 Analyzing logs for {employee_id}..."):
                try:
                    payload = {"employee_name": employee_id, "logs": logs_input}
                    data=generate_report(payload)
                    summary_report = data.get("report", "Error: No report returned from API.")
                    st.success("✅ Report generated successfully!")
                    st.write(data["report"])
                except Exception as e:
                    st.error(f"❌ API error: {e}")
                    summary_report = "Summary generation failed."

            st.success(f"✅ Summary generated for {employee_id}")
            # st.markdown(f"<div class='report-box'>{summary_report}</div>", unsafe_allow_html=True)

            pdf_bytes = create_pdf(summary_report, employee_id)
            st.download_button(
                label="⬇️ Download Summary as PDF",
                data=pdf_bytes,
                file_name=f"performance_summary_{employee_id}_{int(time.time())}.pdf",
                mime="application/pdf"
            )
    else:
        st.markdown("<div class='report-box'>Summary will appear here after generation.</div>", unsafe_allow_html=True)