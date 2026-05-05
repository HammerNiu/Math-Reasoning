# Demo Guide

Target length: 5 minutes or less.

## Required Demo Points

1. Project goal: improve math reasoning by searching over multiple reasoning paths.
2. System workflow: problem input, candidate generation, MCTS selection, process scoring, branch pruning, selected trajectory.
3. Example use case: run one algebra or calculus problem in the Streamlit comparison UI.
4. Result explanation: compare model calls, pruned branches, and selected first step.
5. Limitation: deterministic ablations are committed; full API benchmark run depends on keys and dataset access.

## Live Demo Commands

```bash
.venv/bin/streamlit run app.py
```

Use the default `demo` model for the live presentation. It does not require an
API key and should complete in a few seconds. If you switch to `openai` or
another cloud provider, keep `Simulations = 1` for the first test because the
search loop performs multiple sequential model calls.

Alternative API demo:

```bash
.venv/bin/uvicorn backend.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Suggested 5-Minute Script

| Time | Content |
|---:|---|
| 0:00-0:30 | State the problem: direct generation can commit too early on multi-step math. |
| 0:30-1:20 | Show the architecture: MCTS explores branches, process scorer ranks steps. |
| 1:20-2:30 | Run Streamlit comparison on `Solve for x: x + 3 = 5`. |
| 2:30-3:20 | Point out selected step, pruning count, latency, and model calls. |
| 3:20-4:10 | Show Member 1 and Member 2 ablation tables. |
| 4:10-5:00 | Close with limitations and full benchmark command. |

## Demo Video Link

When the repository is pushed, the included GIF artifact is available at:

```text
https://github.com/HammerNiu/Math-Reasoning/blob/main/docs/demo_video.gif
```

For final submission, the preferred option is to replace or supplement this GIF
with a narrated recording hosted on CourseWorks, Google Drive, Panopto, or
another accessible video platform.
