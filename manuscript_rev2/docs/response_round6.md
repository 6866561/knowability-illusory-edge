# Response to ratings (Knowability 8/10, Illusory Edge 7.5/10)

The ratings named independent checking of both guidance and realized EPS as the most important remaining improvement. Work done since (`audit_crosscheck.py`, `fetch_audit_xbrl.py`; output `data/d7_audit_crosscheck.csv`, `results/audit_crosscheck.json`):

**Reported-EPS side.** This was not examined by the AI adjudicator. It is now checked against SEC XBRL company facts, using the first-reported diluted EPS for the matching period end and length.
- Matches the recorded value for **38 of 40** sampled rows.
- The other 2 are fiscal fourth quarters, which XBRL does not tag separately.
- No reported-EPS errors were found.

**Guidance side.** A rule-based screen of the press-release text was run, independent of the AI.
- It found the recorded range in all 40 releases.
- **33** rows have unambiguous GAAP diluted context; the AI judged all of them correct.
- The screen independently flags continuing-operations wording at **both** rows the AI judged wrong.
- **4** rows the AI judged correct are flagged because "adjusted" or "non-GAAP" wording appears nearby. Neither check resolves them.

**Bound sensitivity** (one-sided 95%, hypergeometric, weighted):

| Treatment | Errors | K_U | Tightened | Plain |
|---|---|---|---|---|
| AI verdicts, unverifiable counted wrong (primary) | 3/40 | 12 | −6.49 | −7.14 |
| + screen-flagged rows counted wrong | 7/40 | 22 | −9.75 | −12.36 |
| + EPS-unchecked rows counted wrong | 9/40 | 26 | −11.23 | −13.93 |

"No positive edge can be certified" holds under every treatment.

**Still needed from the author.**
1. A human second adjudication of the 4 screen-flagged rows (ideally all 40).
2. The public release, which waits on the GitHub login.
