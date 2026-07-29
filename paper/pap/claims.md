# PAP Claim Ledger

No paper claim is registered yet. This file records claim maturity separately
from experiment evidence grades.

## Claim maturity

| Status | Meaning |
| --- | --- |
| `hypothesis` | Falsifiable statement with no supporting experiment yet |
| `observed` | Supported by preliminary or single-run evidence |
| `supported` | Repeated and causally tested within the stated conditions |
| `paper-ready` | Supported across the generality required by the manuscript |
| `falsified` | Contradicted by authoritative evidence |
| `superseded` | Replaced by a more precise claim |

## Active claims

| ID | Claim | Conditions | Status | Evidence | Counterevidence | Next falsification |
| --- | --- | --- | --- | --- | --- | --- |

## Entry requirements

Each claim must:

1. be precise enough to falsify;
2. state the workload, model, hardware, topology, and SLO conditions that bound
   it;
3. link supporting and contradictory PAP experiment IDs;
4. distinguish observed correlation from causal evidence;
5. identify the manuscript section and the next test needed for promotion.

Use the following shape when adding an entry:

```markdown
### CXX: Short claim name

- **Statement:**
- **Conditions:**
- **Status:** `hypothesis`
- **Paper section:**
- **Supporting evidence:**
- **Counterevidence:**
- **Falsification condition:**
- **Next test:**
```

Do not promote a claim to `paper-ready` without completing the paper-level
evidence checks in `README.md`.
