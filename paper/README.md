# PAP paper workspace

Two agents write the PAP paper together through this directory, coordinated by
the human author:

- **research agent** (remote): proposes ideas and hypotheses, owns the paper
  structure and writing.
- **exp agent** (this host, with the GPUs): runs experiments, reports evidence,
  and maintains the experiment code under the rest of this repository.

Both agents exchange information only through git on branch
`feature/pap-v026`. The human author reviews disagreements with the research
agent; the exp agent never needs a human approval to run a request.

## Layout

```text
paper/
├── README.md        # this protocol
├── <writing>/       # research agent: outline, sections, LaTeX, figure scripts
├── figures/         # final figures and the small CSV/JSON they are drawn from
└── handoff/
    ├── claims.md    # every paper claim and its evidence status
    ├── requests/    # research → exp: R-NNN-<slug>.md
    ├── reports/     # exp → research: R-NNN.md (same number as the request)
    └── TEMPLATES.md # request and report formats
```

The research agent chooses the name and structure of its writing directory.

## Write ownership

| Path | research agent | exp agent |
| --- | --- | --- |
| writing directory | write | read |
| `handoff/requests/` | write | update `status:` only |
| `handoff/reports/` | read | write |
| `handoff/claims.md` | claim text, add/remove claims | `Status` and `Evidence` columns |
| `figures/` | figure scripts and outputs | exported data files |
| everything outside `paper/` | read | write |

The exp agent may also open a request itself (prefix `X-NNN`) to propose an
experiment, report an unexpected result, or challenge a claim.

## Evidence rules

- Raw run data stays in the experiment directories under
  `benchmarks/pap/experiments/` and is not copied here. Untracked raw runs stay
  on the exp host; a report names their local path.
- Every number in the paper must trace to a report, and every report names
  the source commit and the experiment directory ID it used.
- `figures/` holds only summarized, plot-ready data (keep each file small).
- A refuted claim stays in `claims.md` with status `refuted` and its report;
  it is not silently rewritten.

## Request lifecycle

`status:` in the request header moves through:

```text
open → accepted → running → done
                ↘ blocked   (needs research-agent / human input; reason in report)
                ↘ rejected  (infeasible or flawed design; reason in report)
```

The exp agent pushes on `accepted` so the research agent sees the request was
taken. Risky or questionable requests are not refused silently: the exp agent
writes the concern in the report, sets `blocked` or `rejected`, and the
research agent decides with the human author.

## Git conventions

- `feature/pap-v026` is the only exchange branch. Do not create per-plan or
  per-result branches; this protocol supersedes branch rules written in any
  earlier plan or runbook.
- Everything the paper depends on (plans, outlines, novelty notes, figure
  data) must be committed under `paper/`; a request must not cite documents
  the exp agent cannot read.
- Code changes from the research agent outside `paper/` go through a PR
  against `feature/pap-v026`; the exp agent builds and validates before the
  human author merges.
- `git pull --rebase` before every push; write scopes do not overlap, so
  conflicts should be rare.
- Commits under `paper/` do not mix with code changes.
- Commit subjects start with `paper:` for writing, `handoff:` for requests,
  reports and claim updates.
