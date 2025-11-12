# llm_service.py
from openai import OpenAI
from app.core.config import secrets

client = OpenAI(api_key=secrets["openai"]["api_key"])


# -------------------------------
# 1️⃣ Lab Summary (Research Lab)
# -------------------------------
def generate_lab_summary(lab_intro_text: str, department: str | None = None) -> str:
    """
    Summarize a research lab introduction into a short English profile
    suitable for text embedding and semantic similarity search.

    The output should be factual and information-dense.
    If `department` is provided, it will be used as context,
    but the model is instructed not to infer or assume extra topics.
    """
    if not lab_intro_text or len(lab_intro_text.strip()) == 0:
        return None

    # 프롬프트 구성
    prompt = f"""
You are a research profiling assistant.

Your job is to read a university lab introduction (possibly written in Korean or informal English)
and produce a 3–4 sentence English summary.

{"The lab is part of the following department or program (for context only; do not assume research topics that are not stated in the text): " + department if department else ""}

Write the summary in clear academic English. Use third-person wording such as
"The lab studies...", "The group focuses on...", or "The team aims to...".

The summary should include, ONLY IF that information is explicitly stated in the lab text:
1. The lab’s main research topics and objectives
2. The key methods, model systems, datasets, or technologies the lab uses
3. The application domains or problems the lab aims to address

Very important:
- Only include information that is directly supported by the provided text.
- Do NOT infer, guess, generalize typical applications, or add details that are not stated.
- If some of the items above are not mentioned, simply omit them.

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
    Summarize a research paper (title + abstract) into a concise, factual English summary.
    The output will be used for text embedding and semantic similarity search,
    so it must be clear, information-dense, and free of interpretation.
    """
    if not abstract or len(abstract.strip()) == 0:
        return None

    prompt = f"""
You are an expert academic summarization assistant.

Your task is to produce a compact and factual English summary (3–4 sentences) of the research paper described below.  

Guidelines:
- Write in clear, formal academic English using third-person style  
- The summary should be information-dense and avoid repetition or filler words.
- Include only what is explicitly stated in the abstract.
- Do NOT infer or interpret unstated aims, results, or implications.

Include, if available:
1. The research goal or problem
2. The core methods, models, or datasets used
3. The main findings or contributions

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
