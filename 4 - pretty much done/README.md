
# CSAT Analyzer — LLM-only, Grouped

- LLM writes Suggestions, Commendations, and the Wrap-up (no heuristic phrasing).
- If the LLM errors or returns nothing, we still render grouped stats + reviews and show an explicit LLM error at the top (and in the diagnostics expander).
- Reviews section shows ALL comments for up to N low-scoring stores per supervisor; blanks and 'nan' are removed.
- Sidebar lets you choose model, tokens, threshold, and max stores.
