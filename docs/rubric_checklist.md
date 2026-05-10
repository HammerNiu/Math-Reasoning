# Rubric Checklist

This checklist maps the EECS E6895 grading rubric and task-split document to
concrete files in the repository.

## Final Report - 40 pts

| Rubric item | Points | Repository evidence |
|---|---:|---|
| Problem formulation and motivation | 10 | `report/final_report.tex` / `report/final_report.pdf`, Sections 1 and 2 |
| System design and methodology | 10 | `report/final_report.tex` / `report/final_report.pdf`, Section 3; `docs/implementation_notes.md` |
| Implementation and data | 10 | `report/final_report.tex` / `report/final_report.pdf`, Section 4; `src/`, `tools/`, `experiments/`, `data/README.md` |
| Evaluation and results | 10 | `report/final_report.tex` / `report/final_report.pdf`, Section 5; `docs/experiment_analysis.md`; ablation docs |

## Presentation and Demo - 30 pts

| Rubric item | Points | Repository evidence |
|---|---:|---|
| Clarity and organization | 10 | `math_reasoning_visual_polish.pptx`; `docs/demo_guide.md` |
| System demonstration | 10 | `app.py`; `backend/main.py`; `docs/demo_video.gif` |
| Understanding and explanation | 10 | `docs/implementation_notes.md`; `docs/experiment_analysis.md`; report discussion section |

## GitHub Repository - 20 pts

| Rubric item | Points | Repository evidence |
|---|---:|---|
| Repository structure | 5 | `README.md` project structure and organized source folders |
| Documentation and usage | 10 | `README.md`; `docs/implementation_notes.md`; `docs/demo_guide.md` |
| Implementation completeness | 5 | MCTS, PPM, verifier, backend, Streamlit, evaluation scripts |

## Overall Project Quality - 10 pts

| Expected quality | Repository evidence |
|---|---|
| Complete end-to-end system | Baseline and improved paths in Streamlit/FastAPI |
| Coherence between report, demo, and implementation | README, report, docs, and demo now use the same terminology |
| Technical depth | Adaptive MCTS plus hybrid process scoring |
| Reliability | No-key ablations, compile checks, API-backed evaluation scripts |

## Team Task Split

| Member role | Required deliverables | Status |
|---|---|---|
| Member 1: Search/MCTS lead | Clean search code, ablation table, slide/demo example | Complete: `src/core/mcts.py`, `tools/member1_search_ablation.py`, `docs/member1_search_ablation.md` |
| Member 2: PPM/verifier lead | Improved scoring module, experiment table, error analysis | Complete: `src/core/scoring.py`, `tools/member2_scoring_ablation.py`, `docs/member2_scoring_ablation.md` |
| Member 3: System/evaluation/demo lead | Comparison interface, run instructions, final results artifacts | Complete for runnable demo and deterministic results; API benchmark command ready |

## Submission Reminders

- CourseWorks report filename must be `uni_uni_uni.pdf`.
- The final PDF must include the GitHub repository link and demo video link.
- Before submission, open both links in an incognito/private browser to verify accessibility.
