"""
Catalyst — AI-Powered Skill Assessment Agent
Uses Google Gemini API (FREE — 1,500 req/day, no credit card)
Get your free key at: https://aistudio.google.com/apikey
"""

import os, json, re, uuid, base64
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

app = FastAPI(title="Catalyst — Skill Assessment Agent")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"]
)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/samples", StaticFiles(directory="samples"), name="samples")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.0-flash"          # free tier, very capable
GEMINI_URL     = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
)

# ── In-memory session store ─────────────────────────────────────────────
sessions: dict[str, dict] = {}


# ═══════════════════════════════════════════════════════════════════════
# GEMINI API WRAPPER
# ═══════════════════════════════════════════════════════════════════════

async def call_gemini(prompt: str, system: str = "", parts_extra: list = None) -> str:
    """Call Gemini REST API. Returns the text response."""
    contents = []

    if system:
        contents.append({
            "role": "user",
            "parts": [{"text": f"[SYSTEM INSTRUCTIONS]\n{system}\n[/SYSTEM INSTRUCTIONS]\n\n{prompt}"}]
        })
    else:
        parts = [{"text": prompt}]
        if parts_extra:
            parts = parts_extra + parts
        contents.append({"role": "user", "parts": parts})

    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2048,
        }
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    async with httpx.AsyncClient(timeout=60) as client:
        res = await client.post(url, json=payload)
        res.raise_for_status()
        data = res.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected Gemini response: {data}") from e


async def call_gemini_with_pdf(prompt: str, pdf_bytes: bytes) -> str:
    """Call Gemini with a PDF file attachment (native PDF support)."""
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    parts_extra = [{
        "inline_data": {
            "mime_type": "application/pdf",
            "data": pdf_b64
        }
    }]
    return await call_gemini(prompt, parts_extra=parts_extra)


def parse_json_safe(raw: str) -> dict | list:
    """Strip markdown fences and parse JSON."""
    cleaned = re.sub(r'^```json\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
    # Sometimes Gemini wraps in single backtick blocks too
    cleaned = re.sub(r'^```\s*|\s*```$', '', cleaned, flags=re.MULTILINE).strip()
    return json.loads(cleaned)


# ═══════════════════════════════════════════════════════════════════════
# NODE 1 — SKILL EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════

JD_PROMPT = """You are an expert technical recruiter.

Analyse this job description and extract ALL required skills.

Rules:
- Extract EXPLICIT skills (directly named: "Python", "FastAPI") 
- Extract IMPLICIT skills (inferred: "build microservices" → Docker, API design)
- importance: "must-have" (in requirements section) | "nice-to-have" (preferred/bonus)
- category: technical | soft | domain | tool | certification
- confidence: 1.0 for explicit, 0.7-0.9 for implicit

Return ONLY valid JSON, absolutely no markdown, no explanation:
{
  "role_title": "Backend Software Engineer",
  "skills": [
    {"skill": "Python", "importance": "must-have", "category": "technical", "explicit": true, "confidence": 1.0, "source_phrase": "2+ years Python"}
  ]
}

JOB DESCRIPTION:
"""

RESUME_PROMPT = """You are an expert technical recruiter analysing a candidate's resume.

Extract ALL skills with concrete evidence.

proficiency levels:
- "beginner": mentioned once, no project context
- "intermediate": used in 1-2 projects, basic application  
- "advanced": used across multiple roles/projects, non-trivial application
- "expert": led others, designed systems, or 5+ years explicit

Return ONLY valid JSON, absolutely no markdown, no explanation:
{
  "candidate_name": "Full Name",
  "current_role": "Most recent job title",
  "skills": [
    {
      "skill": "Python",
      "proficiency": "advanced",
      "years_of_experience": 2,
      "evidence": "Built 6 REST APIs at Infosys serving 8,000 employees",
      "context": "professional"
    }
  ]
}

RESUME:
"""


async def extract_jd_skills(jd_text: str) -> dict:
    raw = await call_gemini(JD_PROMPT + jd_text[:10000])
    return parse_json_safe(raw)


async def extract_resume_skills(resume_bytes: bytes, filename: str) -> dict:
    if filename.lower().endswith(".pdf"):
        # Gemini reads PDFs natively — much better than text extraction
        raw = await call_gemini_with_pdf(
            RESUME_PROMPT + "[Resume is in the attached PDF — extract all skills from it]",
            resume_bytes
        )
    else:
        text = resume_bytes.decode("utf-8", errors="ignore")
        raw = await call_gemini(RESUME_PROMPT + text[:10000])
    return parse_json_safe(raw)


def build_skill_matrix(jd: dict, resume: dict) -> list:
    jd_map  = {s["skill"].lower(): s for s in jd.get("skills", [])}
    res_map = {s["skill"].lower(): s for s in resume.get("skills", [])}

    matrix = []
    for key, jd_skill in jd_map.items():
        match = res_map.get(key)
        if match:
            status = (
                "partial"
                if jd_skill["importance"] == "must-have"
                and match["proficiency"] in ("beginner",)
                else "matched"
            )
        else:
            # fuzzy: check if any resume skill name contains this skill word
            fuzzy = next(
                (v for k, v in res_mapclass.items()
                 if key in k or k in key), None
            )
            status = "partial" if fuzzy else "missing"
            if fuzzy:
                match = fuzzy

        matrix.append({
            "skill":            jd_skill["skill"],
            "jd_importance":    jd_skill["importance"],
            "status":           status,
            "resume_proficiency": match["proficiency"] if match else None,
            "evidence":         match.get("evidence") if match else None,
        })
    return matrix


# ═══════════════════════════════════════════════════════════════════════
# NODE 2 — ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

PRIORITY = {
    ("must-have",    "partial"):  0,
    ("must-have",    "matched"):  1,
    ("nice-to-have", "partial"):  2,
    ("nice-to-have", "matched"):  3,
}
MAX_Q_PER_SKILL = 2


def build_queue(matrix: list) -> list:
    queue = []
    for s in matrix:
        entry = {
            **s,
            "status_assess":  "pending",
            "score":          None,
            "assessor_note":  "",
            "questions_asked": 0,
            "conversation":   [],
        }
        if s["status"] == "missing":
            entry["status_assess"] = "skipped"
        queue.append(entry)

    queue.sort(key=lambda x: (
        99 if x["status_assess"] == "skipped"
        else PRIORITY.get((x["jd_importance"], x["status"]), 50)
    ))
    return queue


def get_current_skill(session: dict) -> Optional[dict]:
    q = session["queue"]
    i = session["idx"]
    return q[i] if i < len(q) else None


def advance_queue(session: dict):
    q = session["queue"]
    i = session["idx"]
    while i < len(q) and q[i]["status_assess"] in ("assessed", "skipped"):
        i += 1
    session["idx"] = i


# ═══════════════════════════════════════════════════════════════════════
# NODE 3 — CONVERSATIONAL ASSESSOR
# ═══════════════════════════════════════════════════════════════════════

ASSESSOR_SYSTEM = (
    "You are a senior technical interviewer conducting a real skill assessment. "
    "Ask one focused, conversational question at a time. "
    "Ground questions in the candidate's actual resume evidence. "
    "Never ask yes/no questions. Sound human, not robotic. "
    "No preamble like 'Great!' or 'That's interesting'. Just the question."
)


async def generate_question(skill: dict) -> str:
    is_followup  = skill["questions_asked"] > 0
    evidence_line = (
        f'Resume evidence: "{skill["evidence"]}"'
        if skill["evidence"]
        else "No prior resume evidence for this skill."
    )
    history = "\n".join(
        f"{m['role'].title()}: {m['content']}"
        for m in skill["conversation"][-4:]
    ) or "None yet."

    if not is_followup:
        instruction = (
            "PARTIAL MATCH — bridge from their existing experience to what the JD needs. "
            "Test whether the gap is conceptual (serious) or just tool familiarity (minor)."
            if skill["status"] == "partial"
            else
            "MATCHED — anchor your question directly to their resume evidence. "
            "Ask about a specific technical decision or trade-off they faced."
        )
    else:
        instruction = (
            f"FOLLOW-UP — last answer scored {skill['score']:.1f}/1.0 (too vague). "
            "Identify the weakest part of their answer and probe exactly that. "
            "Do not repeat the previous question."
        )

    prompt = (
        f"Skill: {skill['skill']} ({skill['jd_importance']})\n"
        f"{evidence_line}\n\n"
        f"Conversation so far:\n{history}\n\n"
        f"Instruction: {instruction}\n\n"
        f"Generate exactly ONE interview question. Max 2 sentences. Conversational."
    )

    return await call_gemini(prompt, system=ASSESSOR_SYSTEM)


async def evaluate_response(skill_name: str, question: str, response: str) -> dict:
    prompt = (
        f'Score this interview response for the skill "{skill_name}".\n\n'
        f"Question: {question}\n"
        f"Response: {response}\n\n"
        "Rubric:\n"
        "0.0–0.3: vague, no specifics\n"
        "0.4–0.6: basic familiarity, shallow\n"
        "0.7–0.8: clear hands-on experience, specific details\n"
        "0.9–1.0: expert depth — trade-offs, edge cases\n\n"
        'Return ONLY JSON, no markdown: {"score": 0.7, "note": "one sentence summary"}'
    )
    raw = await call_gemini(prompt)
    result = parse_json_safe(raw)
    return {
        "score": float(result.get("score", 0.5)),
        "note":  result.get("note", "")
    }


# ═══════════════════════════════════════════════════════════════════════
# NODE 4 — GAP ANALYZER
# ═══════════════════════════════════════════════════════════════════════

def classify_score(score: Optional[float]) -> str:
    if score is None:  return "missing"
    if score >= 0.7:   return "strong"
    if score >= 0.4:   return "weak"
    return "missing"


def get_priority(importance: str, classification: str) -> str:
    table = {
        ("must-have",    "missing"): "critical",
        ("must-have",    "weak"):    "critical",
        ("nice-to-have", "missing"): "moderate",
        ("nice-to-have", "weak"):    "moderate",
    }
    return table.get((importance, classification), "minor")


def build_gap_report(session: dict) -> dict:
    gaps: dict[str, list] = {"critical": [], "moderate": [], "minor": []}

    for s in session["queue"]:
        cl = classify_score(s["score"])
        if cl == "strong" and s["jd_importance"] == "must-have":
            continue  # strong must-have — no action needed

        pl = get_priority(s["jd_importance"], cl)
        gaps[pl].append({
            "skill":          s["skill"],
            "jd_importance":  s["jd_importance"],
            "classification": cl,
            "score":          s["score"],
            "note": s["assessor_note"] or (
                "Skill not present in resume — not assessed."
                if s["score"] is None
                else "Some familiarity but lacks hands-on depth."
            ),
        })

    return {
        "candidate_name": session["candidate_name"],
        "role_title":     session["role_title"],
        "summary": {
            "critical": len(gaps["critical"]),
            "moderate": len(gaps["moderate"]),
            "minor":    len(gaps["minor"]),
            "assessed": sum(
                1 for s in session["queue"]
                if s["status_assess"] == "assessed"
            ),
        },
        "gaps": gaps,
    }


# ═══════════════════════════════════════════════════════════════════════
# NODE 5 — LEARNING PLAN GENERATOR
# ═══════════════════════════════════════════════════════════════════════

PLAN_PROMPT_TPL = """Create a personalised learning plan for {name} applying for: {role}

Their specific skill gaps (with assessor notes on exactly what is weak):
{gaps_text}

For each gap:
1. A focused learning objective — what they specifically need to DO
2. Realistic hours to reach job-ready proficiency (be honest, not optimistic)
3. 2-3 specific resource suggestions with real URLs if you know them

Also provide: total_weeks (assume 10 hrs/week) and a learning sequence (skill names in order).

Return ONLY valid JSON, no markdown:
{{
  "total_weeks": 8,
  "weekly_hours": 10,
  "sequence": ["Docker", "Kubernetes", "FastAPI async"],
  "skills": [
    {{
      "skill": "Kubernetes",
      "priority": "critical",
      "objective": "Deploy a containerised app to a K8s cluster with basic scaling",
      "estimated_hours": 25,
      "note": "Not present in resume — build on existing Docker knowledge",
      "resources": [
        {{"title": "Kubernetes Official Tutorials", "url": "https://kubernetes.io/docs/tutorials/", "type": "tutorial", "why": "Browser-based cluster, zero local setup needed", "free": true}},
        {{"title": "KodeKloud K8s for Beginners", "url": "https://kodekloud.com/courses/kubernetes-for-the-absolute-beginners-hands-on/", "type": "course", "why": "Structured progression from Docker to K8s", "free": false}}
      ]
    }}
  ]
}}"""


async def generate_learning_plan(gap_report: dict) -> dict:
    actionable = (
        gap_report["gaps"]["critical"] +
        gap_report["gaps"]["moderate"]
    )
    if not actionable:
        return {"message": "No significant gaps found — candidate is well-matched!"}

    gaps_text = "\n".join(
        f"- {g['skill']} [{g['jd_importance']}] ({g['classification']}): {g['note']}"
        for g in actionable
    )
    prompt = PLAN_PROMPT_TPL.format(
        name=gap_report["candidate_name"],
        role=gap_report["role_title"],
        gaps_text=gaps_text,
    )

    raw  = await call_gemini(prompt)
    plan = parse_json_safe(raw)

    plan["candidate_name"] = gap_report["candidate_name"]
    plan["role_title"]     = gap_report["role_title"]
    plan["generated_at"]   = datetime.utcnow().isoformat()
    return plan


# ═══════════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def root():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": GEMINI_MODEL,
            "api_key_set": bool(GEMINI_API_KEY)}


@app.post("/api/start")
async def start_session(
    jd_text: str  = Form(...),
    resume:  UploadFile = File(...),
):
    resume_bytes = await resume.read()
    filename     = resume.filename or "resume.txt"

    # Run extraction
    jd_data     = await extract_jd_skills(jd_text)
    resume_data = await extract_resume_skills(resume_bytes, filename)
    matrix      = build_skill_matrix(jd_data, resume_data)

    session_id = str(uuid.uuid4())
    session = {
        "session_id":     session_id,
        "candidate_name": resume_data.get("candidate_name", "Candidate"),
        "role_title":     jd_data.get("role_title", "the role"),
        "queue":          build_queue(matrix),
        "idx":            0,
        "phase":          "assessing",
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
    sk["status_assess"]    = "probing"

    assessed  = sum(1 for s in session["queue"] if s["status_assess"] == "assessed")
    to_assess = sum(1 for s in session["queue"] if s["status_assess"] != "skipped")

    return {
        "session_id":           session_id,
        "phase":                "assessing",
        "candidate_name":       session["candidate_name"],
        "role_title":           session["role_title"],
        "question":             question,
        "skill_being_assessed": sk["skill"],
        "progress":             {"assessed": assessed, "total": to_assess},
    }


class ChatPayload(BaseModel):
    session_id: str = Field(...)
    message:    str = Field(...)

    class Config:
        anystr_strip_whitespace = True


@app.post("/api/chat")
async def chat(payload: ChatPayload):
    session = sessions.get(payload.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    sk = get_current_skill(session)
    if sk is None or session["phase"] == "complete":
        return {"phase": "complete"}

    # Evaluate candidate's response
    last_q      = sk["conversation"][-1]["content"] if sk["conversation"] else ""
    eval_result = await evaluate_response(sk["skill"], last_q, payload.message)

    sk["score"]        = eval_result["score"]
    sk["assessor_note"] = eval_result["note"]
    sk["conversation"].append({"role": "candidate", "content": payload.message})

    # Decide: follow-up or move to next skill
    needs_followup = (
        eval_result["score"] < 0.6 and
        sk["questions_asked"] < MAX_Q_PER_SKILL
    )

    if not needs_followup:
        sk["status_assess"] = "assessed"
        advance_queue(session)
        sk = get_current_skill(session)

    if sk is None:
        session["phase"] = "complete"
        return {"phase": "complete", "session_id": payload.session_id}

    question = await generate_question(sk)
    sk["conversation"].append({"role": "assessor", "content": question})
    sk["questions_asked"] += 1
    sk["status_assess"]    = "probing"

    assessed  = sum(1 for s in session["queue"] if s["status_assess"] == "assessed")
    to_assess = sum(1 for s in session["queue"] if s["status_assess"] != "skipped")

    return {
        "phase":                "assessing",
        "question":             question,
        "skill_being_assessed": sk["skill"],
        "progress":             {"assessed": assessed, "total": to_assess},
    }


@app.get("/api/results/{session_id}")
async def results(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    gap_report = build_gap_report(session)
    plan       = await generate_learning_plan(gap_report)
    return {"gap_report": gap_report, "learning_plan": plan}
