# CLAUDE.md — ember

From-scratch GPT model + training library, built as a learning project against
`../PLAN.md` (sibling meta repo). Current milestone brief lives in
`docs/milestones/` (A0 now); status table in README.md.

## Learning mode (default: ON)

Rafe writes the library code — model.py, data.py, train.py, tokenizer,
finetune, eval, and the tests that gate milestones. Claude must not implement
or patch these, even for "small fixes", unless Rafe explicitly says
**"override learning mode"**.

Claude does:
- Scaffolding/infra: packaging, CI, bin/, configs, data-download plumbing.
- Concept explanations; self-test discussion after Rafe attempts the answers.
- Code review after the gate test passes or a real attempt exists — name bugs
  by symptom and principle ("position t can attend to t+1 — check your mask's
  triangle/offset"), and let Rafe write the fix.
- Debugging help as questions and ideas, not patches.
- Generate the next milestone brief from ../PLAN.md when the current
  definition-of-done is checked (`git tag aN-done` exists).

## Practical
- Entry points: `bin/ember {train,test,eval}`. Run tests with `bin/ember test`.
- The stubs in `ember/*.py` carry the spec in docstrings — they stay specs.
- `tests/test_overfit.py` is the permanent CI gate. Never weaken its threshold
  or skip it to get CI green.
- Lab notes go in NOTES.md (newest first); the weekly log lives in ../LOG.md.
- Hardware: develop on the Mac (MPS/CPU, tiny configs); metric runs on the
  2080S (Turing: fp16+GradScaler, no bf16); A2 rents H100s.
