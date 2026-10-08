## Cold start

For non-trivial work, read in this order:

1. `README.md`
2. `AGENTS.md`
3. `dev/README.md`


## Developer journal
Keep a running journal at `dev/journals/<agent_name>.md` (e.g.
`dev/journals/codex.md`) to capture the story of the work (decisions, progress,
challenges). This is not a changelog.  Write at a high level for future
maintainers: enough context for someone to pick up where you left off.

- Format: Each entry starts with `## YYYY-MM-DD HH:MM:SS -ZZZZ` (local time).
- Must include: summary of user intent. If still editing this entry then you can update this if the user provides a clarifying prompt. The idea is that this is a compressed form of the prompt you are responding to. You can refer to previous entries to keep this concise. The journal should give a reader an idea of what you were responding to. The diff should make sense next to it.
- Must include: your exact model name and configuration.
- Must include: what you were working on, a substantive entry about your state of mind / reflections, uncertainties/risks, tradeoffs, what might break, what you're confident about.
- May include: what happened, rationale, testing notes, next steps, open questions.
- Rules: Prefer append-only. You may edit only the most recent entry *during the same session* (use timestamp + context to judge); never modify the timestamp line; once a new session starts, create a new entry. Never modify older entries. Avoid large diffs; reference files/modules/issues instead.
- Write journal entries as design narratives: capture the user's underlying goal, the constraints that matter, the alternatives you considered, why the chosen approach won, what tradeoffs were accepted, and 1-3 reusable design takeaways that could teach a future engineer how to make a similar decision.

### Lessons Learned

Maintain `dev/lessons/lessons.md` for confirmed, reusable lessons only.

Add a lesson only when you have confirmed a generic, non-obvious behavior, or
when the lesson required a longer investigation to establish. Not every bug,
decision, or observation deserves a lesson.

Each lesson must cite solid evidence:
- an external reference, such as a file, test, issue, commit, doc, or journal
  entry; or
- an MWE created under `dev/lessons/mwe/`.

Keep entries short and scoped.

Format:

- **Lesson:** reusable takeaway.
- **Evidence / MWE:** supporting reference.
- **Applies when:** conditions where this lesson is relevant.

Rules: no speculation; prefer append-only; supersede incorrect lessons with a
new entry; keep diffs small.

If a lesson seems important, but a MWE is not possible, label the lesson as
SPECULATIVE.

## Commit attribution

Every agent-authored commit must carry a `Co-authored-by:` trailer naming the exact model (and weights/config variant, when known) that produced it, per its own runtime identification — never copied from earlier commits; if undeterminable, say so.
