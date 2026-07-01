<!-- karpathy-synergy-3:start -->
## Karpathy Synergy 3.0

- On invocation, run the bundled `manage.py activate` workflow before project work.
- Classify work as fast, standard, or deep; expand context whenever correctness or risk requires it.
- Apply Ponytail's ladder: skip speculative work, then prefer existing code, standard library, native platform features, installed dependencies, and finally the minimum new code.
- Start with targeted identifiers and at most four candidate files for low-risk work.
- For standard work, query `.agents/scripts/graphify` when `graphify-out/graph.json` exists; otherwise continue with targeted `rg`.
- Make the smallest correct change and run the cheapest specific verification first.
- Preserve security, authorization, privacy, data integrity, accessibility, visual fidelity, and required tests.
- Never trade correctness or necessary understanding for fewer tokens or lines.
- Search `.agents/LESSONS.md` by identifier and use the blueprint workflow only for structural changes; keep hooks, watcher, and index maintained by activation.
- Finish with the change, verification, and remaining risk; keep routine output concise without truncating requested explanations.
<!-- karpathy-synergy-3:end -->
