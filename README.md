# Catalyst — AI-Powered Skill Assessment & Personalised Learning Plan Agent
> Deccan AI Hackathon submission · Powered by Google Gemini (FREE)

## Quick Start (3 steps)

### 1. Get your FREE Gemini API key
- Go to https://aistudio.google.com/apikey
- Sign in with Google → click "Create API key"
- No credit card required. 1,500 free requests/day.

### 2. Install and run
```bash
git clone https://github.com/<your-username>/catalyst-hackathon
cd catalyst-hackathon
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here   # Windows: set GEMINI_API_KEY=your_key_here
uvicorn app.main:app --reload --port 8000
```

### 3. Open the app
Visit http://localhost:8000 — click **"Load samples"** to try with the included sample JD and resume.

---

## Sample Files
| File | Description |
|------|-------------|
| `samples/sample_job_description.txt` | Backend Software Engineer @ FinFlow Technologies |
| `samples/sample_resume.txt` | Akshay Kalwaghe — 2.5 yrs Python/Flask/PostgreSQL/Docker |

The sample is intentionally realistic: the candidate has *most* but not all required skills (missing Kubernetes, Redis, Kafka), making for a rich assessment with a meaningful learning plan.

---

## Architecture

```
JD text + Resume (PDF or TXT)
          │
          ▼
┌─────────────────────┐
│  Node 1: Extractor  │  Single Gemini call → structured skill matrix
│                     │  matched | partial | missing
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Node 2: Orchestrator│ Priority queue: partial must-haves first
│                     │  Max 2 questions per skill
└─────────┬───────────┘
          │ conversation loop
          ▼
┌─────────────────────┐
│  Node 3: Assessor   │  Generates questions anchored to resume evidence
│                     │  Evaluates depth 0.0–1.0, follows up if < 0.6
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Node 4: Gap Analyzer│  Classifies: strong / weak / missing
│                     │  Ranks by importance × severity
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Node 5: Plan Gen   │  Gemini generates objectives + real resources
│                     │  with time estimates and learning sequence
└─────────────────────┘
          │
          ▼
   Personalised PDF Learning Plan
```

## Scoring Logic

| Score | Classification | Action |
|-------|---------------|--------|
| ≥ 0.7 | Strong | Skipped in learning plan |
| 0.4–0.69 | Weak | Added to learning plan |
| < 0.4 or null | Missing | Added as critical/moderate gap |

**Priority ranking:**
- `must-have + missing/weak` → **Critical** (addressed first in plan)
- `nice-to-have + missing/weak` → **Moderate**
- Assessment capped at 2 questions per skill to keep session under 15 minutes

## Deploy to Render (Free)

1. Push to GitHub
2. render.com → New Web Service → connect repo
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Environment variable: `GEMINI_API_KEY=your_key`

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Google Gemini 2.0 Flash (free, 1,500 req/day) |
| Backend | Python + FastAPI |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| PDF parsing | Gemini native PDF vision |
| Deployment | Render (free tier) |

## Submission Checklist
- [x] Working prototype
- [x] Source code with README
- [x] Architecture diagram (above)
- [x] Scoring/logic description (above)
- [x] Sample inputs (`samples/` folder)
- [x] PDF export (browser print)
- [ ] Deployed URL
- [ ] Demo video (3–5 min)
