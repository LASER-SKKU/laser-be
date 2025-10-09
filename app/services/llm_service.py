from openai import OpenAI
from app.core.config import secrets

client = OpenAI(api_key=secrets["openai"]["api_key"])

# -------------------------------
# 1️⃣ Lab Summary (Research Lab)
# -------------------------------
def generate_lab_summary(lab_intro_text: str) -> str:
    """
    Summarize research lab introductions into concise English summaries.
    Emphasize research areas, technologies, and main goals.
    """
    if not lab_intro_text or len(lab_intro_text.strip()) == 0:
        return None

    prompt = f"""
You are a professional research assistant specializing in summarizing university lab introductions.

Below is the introduction text of a research lab (possibly written in Korean).
Summarize it into **3–4 concise English sentences** that clearly describe:
- The lab’s research focus and goals
- Core technologies or methodologies
- Application areas or unique strengths

Write naturally in clear academic English.

--- LAB INTRODUCTION ---
{lab_intro_text}
"""

    try:
        response = client.chat.completions.create(
            model=secrets["openai"]["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM ERROR: Lab Summary] {e}")
        return None


# -------------------------------
# 2️⃣ Paper Summary (Academic Paper)
# -------------------------------
def generate_paper_summary(title: str, abstract: str) -> str:
    """
    Summarize research papers into concise English summaries.
    Include purpose, methods, and main findings.
    """
    if not abstract or len(abstract.strip()) == 0:
        return None

    prompt = f"""
You are an academic summarization assistant.

Summarize the following paper into **3–4 English sentences** focusing on:
- The research objective
- The key methodology
- The main findings or contributions

Write the summary in clear, formal academic English.

--- PAPER TITLE ---
{title}

--- PAPER ABSTRACT ---
{abstract}
"""

    try:
        response = client.chat.completions.create(
            model=secrets["openai"]["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM ERROR: Paper Summary] {e}")
        return None
