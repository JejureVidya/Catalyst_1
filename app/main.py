"""
Catalyst - AI-Powered Skill Assessment Agent
Uses Google Gemini API (FREE - 1,500 req/day, no credit card)
Get your free key at: https://aistudio.google.com/apikey
"""

import os
import json
import re
import uuid
import base64
from datetime import datetime
from typing import Optional, Dict, List, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI(title="Catalyst - Skill Assessment Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/samples", StaticFiles(directory="samples"), name="samples")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"

sessions: Dict[str, Any] = {}


async def call_gemini(prompt: str, system: str = "", parts_extra: Optional[List] = None) -> str:
    contents = []
    if system:
        combined = "[SYSTEM INSTRUCTIONS]\n" + system + "\n[/SYSTEM INSTRUCTIONS]\n\n" + prompt
        contents.append({"role": "user", "parts": [{"text": combined}]})
    else:
        parts = [{"text": prompt}]
        if parts_extra:
            parts = parts_extra + parts
        contents.append({"role": "user", "parts": parts})

    payload = {
        "contents": contents,
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048}
    }
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + GEMINI_MODEL + ":generateContent?key=" + GEMINI_API_KEY
    )
    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(url, json=payload)
        res.raise_for_status()
        data = res.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        raise ValueError("Unexpected Gemini response: " + str(data)) from e


async def call_gemini_with_pdf(prompt: str, pdf_bytes: bytes) -> str:
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    parts_extra = [{"inline_data": {"mime_type": "application/pdf", "data": pdf_b64}}]
    return await call_gemini(prompt, parts_extra=parts_extra)


def parse_json_safe(raw: str) -> Any:
    cleaned = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
    cleaned = re.sub(r'^```\s*|\s*```$', '', cleaned, flags=re.MULTILINE).strip()
    return json.loads(cleaned)


JD_PROMPT = """You are an expert technical recruiter.
Analyse this job description and extract ALL required skills.
Rules:
- Extract EXPLICIT skills (directly named) and IMPLICIT skills (inferred)
- importance: "must-have" or "nice-to-have"
- category: technical or soft or domain or tool or certification
- confidence: 1.0 for explicit, 0.7-0.9 for implicit
Return ONLY valid JSON, no markdown:
{"role_title": "...", "skills": [{"skill": "Python", "importance": "must-have", "category": "technical", "explicit": true, "confidence": 1.0, "source_phrase": "2+ years Python"}]}
JOB DESCRIPTION:
"""

RESUME_PROMPT = """You are an expert technical recruiter analysing a candidate resume.
Extract ALL skills with concrete evidence.
proficiency: "beginner" or "intermediate" or "advanced" or "expert"
Return ONLY valid JSON, no markdown:
{"candidate_name": "Full Name", "current_role": "Job Title", "skills": [{"skill": "Python", "proficiency": "advanced", "years_of_experience": 2, "evidence": "Built APIs at Infosys", "context": "professional"}]}
RESUME:
"""


async def extract_jd_skills(jd_text: str) -> Dict:
    raw = await call_gemini(JD_PROMPT + jd_text[:10000])
    return parse_json_safe(raw)


async def extract_resume_skills(resume_bytes: bytes, filename: str) -> Dict:
    if filename.lower().endswith(".pdf"):
        raw = await call_gemini_with_pdf(
            RESUME_PROMPT + "[Resume is in the attached PDF - extract all skills from it]",
            resume_bytes
        )
    else:
        text = resume_bytes.decode("utf-8", errors="ignore")
        raw = await call_gemini(RESUME_PROMPT + text[:10000])
    return parse_json_safe(raw)


def build_skill_matrix(jd: Dict, resume: Dict) -> List:
    jd_map = {s["skill"].lower(): s for s in jd.get("skills", [])}
    res_map = {s["skill"].lower(): s for s in resume.get("skills", [])}
    matrix = []
    for key, jd_skill in jd_map.items():
        match = res_map.get(key)
        if match:
            status = "partial" if jd_skill["importance"] == "must-have" and match["proficiency"] == "beginner" else "matched"
        else:
            fuzzy = next((v for k, v in res_map.items() if key in k or k in key), None)
            status = "partial" if fuzzy else "missing"
            if fuzzy:
                match = fuzzy
        matrix.append({
            "skill": jd_skill["skill"],
            "jd_importance": jd_skill["importance"],
            "status": status,
            "resume_proficiency": match["proficiency"] if match else None,
            "evidence": match.get("evidence") if match else None,
        })
    return matrix


PRIORITY = {("must-have", "partial"): 0, ("must-have", "matched"): 1, ("nice-to-have", "partial"): 2, ("nice-to-have", "matched"): 3}
MAX_Q_PER_SKILL = 2


def build_queue(matrix: List) -> List:
    queue = []
    for s in matrix:
        entry = {
            "skill": s["skill"], "jd_importance": s["jd_importance"],
            "status": s["status"], "resume_proficiency": s.get("resume_proficiency"),
            "evidence": s.get("evidence"), "status_assess": "pending",
            "score": None, "assessor_note": "", "questions_asked": 0, "conversation": [],
        }
        if s["status"] == "missing":
            entry["status_assess"] = "skipped"
        queue.append(entry)
    queue.sort(key=lambda x: 99 if x["status_assess"] == "skipped" else PRIORITY.get((x["jd_importance"], x["status"]), 50))
    return queue


def get_current_skill(session: Dict) -> Optional[Dict]:
    q = session["queue"]
    i = session["idx"]
    return q[i] if i < len(q) else None


def advance_queue(session: Dict) -> None:
    q = session["queue"]
    i = session["idx"]
    while i < len(q) and q[i]["status_assess"] in ("assessed", "skipped"):
        i += 1
    session["idx"] = i


ASSESSOR_SYSTEM = (
    "You are a senior technical interviewer. Ask one focused conversational question at a time. "
    "Ground questions in the candidate's resume evidence. Never ask yes/no questions. "
    "No preamble. Just the question."
)


async def generate_question(skill: Dict) -> str:
    is_followup = skill["questions_asked"] > 0
    evidence_line = ('Resume evidence: "' + skill["evidence"] + '"') if skill["evidence"] else "No resume evidence for this skill."
    history = "\n".join(m["role"].title() + ": " + m["content"] for m in skill["conversation"][-4:]) or "None yet."

    if not is_followup:
        instruction = (
            "PARTIAL MATCH - bridge from their experience to what the JD needs."
            if skill["status"] == "partial"
            else "MATCHED - anchor to their resume evidence, ask about a specific decision or trade-off."
        )
    else:
        score_val = round(skill["score"] if skill["score"] is not None else 0.5, 1)
        instruction = "FOLLOW-UP - last answer scored " + str(score_val) + "/1.0. Probe the weakest part. Do not repeat the previous question."

    prompt = (
        "Skill: " + skill["skill"] + " (" + skill["jd_importance"] + ")\n"
        + evidence_line + "\n\nConversation so far:\n" + history
        + "\n\nInstruction: " + instruction
        + "\n\nGenerate exactly ONE interview question. Max 2 sentences. Conversational."
    )
    return await call_gemini(prompt, system=ASSESSOR_SYSTEM)


async def evaluate_response(skill_name: str, question: str, response: str) -> Dict:
    prompt = (
        'Score this interview response for the skill "' + skill_name + '".\n\n'
        "Question: " + question + "\nResponse: " + response + "\n\n"
        "Rubric: 0.0-0.3 vague, 0.4-0.6 basic, 0.7-0.8 hands-on, 0.9-1.0 expert\n\n"
        'Return ONLY JSON: {"score": 0.7, "note": "one sentence"}'
    )
    raw = await call_gemini(prompt)
    result = parse_json_safe(raw)
    return {"score": float(result.get("score", 0.5)), "note": result.get("note", "")}


def classify_score(score: Optional[float]) -> str:
    if score is None: return "missing"
    if score >= 0.7: return "strong"
    if score >= 0.4: return "weak"
    return "missing"


def get_priority_label(importance: str, classification: str) -> str:
    table = {
        ("must-have", "missing"): "critical", ("must-have", "weak"): "critical",
        ("nice-to-have", "missing"): "moderate", ("nice-to-have", "weak"): "moderate",
    }
    return table.get((importance, classification), "minor")


def build_gap_report(session: Dict) -> Dict:
    gaps: Dict[str, List] = {"critical": [], "moderate": [], "minor": []}
    for s in session["queue"]:
        cl = classify_score(s["score"])
        if cl == "strong" and s["jd_importance"] == "must-have":
            continue
        pl = get_priority_label(s["jd_importance"], cl)
        gaps[pl].append({
            "skill": s["skill"], "jd_importance": s["jd_importance"],
            "classification": cl, "score": s["score"],
            "note": s["assessor_note"] or (
                "Skill not present in resume - not assessed." if s["score"] is None
                else "Some familiarity but lacks hands-on depth."
            ),
        })
    return {
        "candidate_name": session["candidate_name"],
        "role_title": session["role_title"],
        "summary": {
            "critical": len(gaps["critical"]), "moderate": len(gaps["moderate"]),
            "minor": len(gaps["minor"]),
            "assessed": sum(1 for s in session["queue"] if s["status_assess"] == "assessed"),
        },
        "gaps": gaps,
    }


PLAN_PROMPT_TPL = """Create a personalised learning plan for {name} applying for: {role}

Skill gaps:
{gaps_text}

For each gap produce: objective, estimated_hours, and 2-3 resources with real URLs.
Also produce total_weeks (assume 10 hrs/week) and sequence.

Return ONLY valid JSON, no markdown:
{{
  "total_weeks": 8, "weekly_hours": 10,
  "sequence": ["Skill A", "Skill B"],
  "skills": [
    {{
      "skill": "Kubernetes", "priority": "critical",
      "objective": "Deploy a containerised app to K8s with basic scaling",
      "estimated_hours": 25,
      "note": "Not in resume - build on Docker knowledge",
      "resources": [
        {{"title": "Kubernetes Tutorials", "url": "https://kubernetes.io/docs/tutorials/", "type": "tutorial", "why": "Browser-based, zero setup", "free": true}}
      ]
    }}
  ]
}}"""


async def generate_learning_plan(gap_report: Dict) -> Dict:
    actionable = gap_report["gaps"]["critical"] + gap_report["gaps"]["moderate"]
    if not actionable:
        return {"message": "No significant gaps found - candidate is well-matched!"}
    gaps_text = "\n".join("- " + g["skill"] + " [" + g["jd_importance"] + "] (" + g["classification"] + "): " + g["note"] for g in actionable)
    prompt = PLAN_PROMPT_TPL.format(name=gap_report["candidate_name"], role=gap_report["role_title"], gaps_text=gaps_text)
    raw = await call_gemini(prompt)
    plan = parse_json_safe(raw)
    plan["candidate_name"] = gap_report["candidate_name"]
    plan["role_title"] = gap_report["role_title"]
    plan["generated_at"] = datetime.utcnow().isoformat()
    return plan


@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": GEMINI_MODEL, "api_key_set": bool(GEMINI_API_KEY)}


@app.post("/api/start")
async def start_session(jd_text: str = Form(...), resume: UploadFile = File(...)):
    resume_bytes = await resume.read()
    filename = resume.filename or "resume.txt"
    jd_data = await extract_jd_skills(jd_text)
    resume_data = await extract_resume_skills(resume_bytes, filename)
    matrix = build_skill_matrix(jd_data, resume_data)
    session_id = str(uuid.uuid4())
    session = {
        "session_id": session_id,
        "candidate_name": resume_data.get("candidate_name", "Candidate"),
        "role_title": jd_data.get("role_title", "the role"),
        "queue": build_queue(matrix), "idx": 0, "phase": "assessing",
    }
    advance_queue(session)
    sessions[session_id] = session
    sk = get_current_skill(session)
    if sk is None:
        session["phase"] = "complete"
        return {"session_id": session_id, "phase": "complete"}
    question = await generate_question(sk)
    sk["conversation"].append({"role": "assessor", "content": question})
    sk["questions_asked"] += 1
    sk["status_assess"] = "probing"
    assessed = sum(1 for s in session["queue"] if s["status_assess"] == "assessed")
    to_assess = sum(1 for s in session["queue"] if s["status_assess"] != "skipped")
    return {
        "session_id": session_id, "phase": "assessing",
        "candidate_name": session["candidate_name"], "role_title": session["role_title"],
        "question": question, "skill_being_assessed": sk["skill"],
        "progress": {"assessed": assessed, "total": to_assess},
    }


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    session_id = body.get("session_id")
    message = body.get("message")
    if not session_id or not message:
        raise HTTPException(400, "session_id and message required")
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    sk = get_current_skill(session)
    if sk is None or session["phase"] == "complete":
        return {"phase": "complete"}
    last_q = sk["conversation"][-1]["content"] if sk["conversation"] else ""
    eval_result = await evaluate_response(sk["skill"], last_q, message)
    sk["score"] = eval_result["score"]
    sk["assessor_note"] = eval_result["note"]
    sk["conversation"].append({"role": "candidate", "content": message})
    needs_followup = eval_result["score"] < 0.6 and sk["questions_asked"] < MAX_Q_PER_SKILL
    if not needs_followup:
        sk["status_assess"] = "assessed"
        advance_queue(session)
        sk = get_current_skill(session)
    if sk is None:
        session["phase"] = "complete"
        return {"phase": "complete", "session_id": session_id}
    question = await generate_question(sk)
    sk["conversation"].append({"role": "assessor", "content": question})
    sk["questions_asked"] += 1
    sk["status_assess"] = "probing"
    assessed = sum(1 for s in session["queue"] if s["status_assess"] == "assessed")
    to_assess = sum(1 for s in session["queue"] if s["status_assess"] != "skipped")
    return {
        "phase": "assessing", "question": question,
        "skill_being_assessed": sk["skill"],
        "progress": {"assessed": assessed, "total": to_assess},
    }


@app.get("/api/results/{session_id}")
async def results(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    gap_report = build_gap_report(session)
    plan = await generate_learning_plan(gap_report)
    return {"gap_report": gap_report, "learning_plan": plan}
