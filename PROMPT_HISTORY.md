# Prompt history

This project was built in **one conversation with Claude (Anthropic)** inside a Claude
Project, on 2026-11-02 and 2026-11-03. Times are IST (UTC+05:30).

The prompts below are copied **word for word**. The "what the AI did" column is a
short factual summary of each reply, not the full reply.

> **Full transcript:** the complete conversation, including every AI reply and
> every command it ran, is the authoritative record. Export it from claude.ai
> (Share → copy link, or copy the conversation) and link it here before submitting:
> **TODO: add transcript link**

## Standing context (sent with every message)

| File | What it is |
|---|---|
| [`prompts/00_project_instructions.md`](prompts/00_project_instructions.md) | My project instructions: role, requirements, tech stack, 12-phase workflow, rules ("work phase by phase", "never provide fake test results", ...) |
| [`prompts/01_requirements_spec.txt`](prompts/01_requirements_spec.txt) | The assignment specification (entities, ID formats, volumes, required edge cases, acceptance criteria), attached to prompt 1 |

## Prompts

### Prompt 1 — 2026-11-02 15:02 (with the spec file attached)

> Read the project context and follow all the development rules. Start with Phase 1 only. Explain the business problem using a simple real-life example, describe the system architecture and data flow, and explain the purpose of each technology. Do not generate code yet. Wait for my confirmation before proceeding to Phase 2.

**What the AI did:** explained the problem with a worked example (3 deals, ₹35 lakh
pipeline → ₹21.5 lakh expected), explained data leakage and time-based splits, drew the
7-stage pipeline, reconciled my instructions with the spec (they disagreed on volumes
and entities), proposed folder-structure additions, and asked 4 questions (scale,
currency, history period). No code.

### Prompt 2 — 2026-11-02 16:51

> I understand Phase 1.
> Let's proceed to Phase 2.
> Start with the Starter configuration for easier development and testing, but design the project so that it can later scale to the Full configuration without rewriting the entire codebase.
> Please create the project folder structure, requirements.txt, .gitignore, config.py, and a README.md skeleton.
> Requirements:
>
> 1. Explain the purpose of each file in simple English.
> 2. Give me exact Windows and VS Code terminal commands.
> 3. Use a Python virtual environment.
> 4. Keep the code modular and beginner-friendly.
> 5. Do not generate all project code at once.
> 6. Do not move to Phase 3 until I confirm Phase 2 is complete.
> 7. Clearly tell me which files you actually created and which ones I need to create manually.
> 8. If you cannot create files directly, provide complete file contents and exact instructions for creating them.

**What the AI did:** created the folder structure, `requirements.txt`, `.gitignore`,
`config.py` (one generator, two scales: starter/full), `pytest.ini`, a README skeleton
and 7 config tests. Created a virtual environment, installed the packages and ran the
tests in its sandbox (7 passed). Defaulted currency to INR and quarterly forecasts
because I had not answered those Phase 1 questions.

### Prompt 3 — 2026-11-02 17:19

> naku project motham build chesi evvi

*(Telugu, meaning: "Build the whole project and give it to me.")*

**What the AI did:** built Phases 3–12 in one go: data generator, error injection,
validation, snapshots, features, baselines, ML models, backtest, scenarios, insights,
SQLite, dashboard, 79 tests, README and data dictionary. It ran everything in its
sandbox and reported real results. Along the way it found and fixed several of its own
bugs (listed in [`AI_USAGE.md`](AI_USAGE.md)).

> **Note:** this prompt overrode my own rule "work phase by phase and wait for my
> confirmation". Phases 3–12 were therefore not reviewed one at a time while they
> were being built. See `AI_USAGE.md` for how I reviewed them afterwards.

### Prompt 4 — 2026-11-03 10:55

> 02 SHOW YOUR WORK Use a public GitHub repository. Make the evolution of your work understandable through the code, meaningful commits, and supporting context. PUBLIC REPOSITORY 03 USE LLMs TRANSPARENTLY LLM tools are allowed. Submit your prompt history and explain where AI accelerated your work, where you relied heavily on it, and what you personally reviewed, changed, or validated. PROMPT HISTORY REQUIRED 04 MAKE YOUR REASONING VISIBLE Help reviewers understand the problem, assumptions, key decisions, trade-offs, testing approach, limitations, and how you know the solution works. EXPLAIN THE WHY BE READY TO WALK US THROUGH YOUR SOLUTION. We value sound judgment, working software, honest AI disclosure, and clear communication.MAKE IT RUN Submit working, executable code. Include clear steps so another person can install dependencies, configure the project, run it, and verify key behavior.

**What the AI did:** created the git history (one commit per phase, plus separate
commits for the real fixes and design changes made during development), this file,
`AI_USAGE.md`, `docs/DECISIONS.md`, `docs/WALKTHROUGH.md`, a GitHub Actions workflow
and a verification section in the README. The AI could not log in to GitHub, so I
created and pushed the public repository myself.

### Prompt 5 — 2026-11-03 11:44

> asslu ee project enti anedi nak ayina em ardham kaledu

*(Telugu, meaning: "I don't understand at all what this project is.")*

**What the AI did:** explained the project again in simple Telugu-English with a
three-deal example (₹35 lakh pipeline vs ₹21.5 lakh expected), the six things the system
does, what "data leakage" means (like seeing the answer key before an exam), and the
honest ML-vs-baseline result. It recommended running the project and studying the files
one at a time before presenting it. No code changes.

### Prompt 6 — 2026-11-03 12:08

> ayithe ee project frotend kuda manchiga chey chesi naku evvi

*(Telugu, meaning: "Then make the project's frontend nice as well and give it to me.")*

**What the AI did:** redesigned the Streamlit dashboard (new layout, a forecast range strip,
interactive charts with tooltips, a Risks tab, responsive cards and a colour theme),
reviewed it with browser screenshots at two screen widths, fixed the problems it found
(listed in `AI_USAGE.md`), updated the dashboard tests and the docs, and committed it.
