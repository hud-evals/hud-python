# Docs restructure - handoff

A plan to reorganize the v6 docs (`hud-python/docs`, Mintlify) into a clean **Build** (narrative) vs
**Reference** (per-object API) split. This doc is a handoff for another agent to finish the work.

## Why

Today the workflow story is re-told on nearly every page (`start/overview`, `build/*`, `run/*`), so
there's no single place to read the full flow end to end, and narrative is constantly interrupted by
object definitions. We're separating the two concerns (Diátaxis: explanation/how-to vs reference).

## Principles (hold the line on these)

- **Build = the narrative spine.** Read top to bottom to learn the whole workflow. Never exhaustive.
- **Reference = the dry per-object truth.** Object → signature → params → returns → minimal example.
  No storytelling. This is what the docs skill / MCP server (`docs.hud.ai`) serves to agents.
- When a Reference page starts telling a story, or a Build page starts enumerating every param, stop -
  that's the old problem coming back.
- Follow `.agents/skills/docs-write/SKILL.md`: no em dashes (spaced `-` only), cut filler, motivate
  concepts, define terms before use, verify claims against the code, hard-wrap prose ~100 cols.

## Target sidebar structure

```
Start here   start/index, start/quickstart
Build        build/index (spine), protocol, environments, tasks, run, train, advice
Reference    environment, tasks, capabilities, graders, agents, runtime, robots, training, types, cli
Advanced     integrations, subagents, chat, patterns, harbor-convert
Cookbooks    coding-agent, ops-diagnostics, a2a-chat, robot-benchmark
More         faq, migrate-v6, contributing
```

Run & scale is dissolved: `signal` → Build/advice, `deploy`+`models` → Build/run, `training` → Build/train.

## Content mapping (old → new)

| New Build page | Merge content from |
|---|---|
| `build/index` (The full flow) | `start/overview` (the 5-step flow + mermaid, trimmed to the spine) |
| `build/protocol` | `core/protocol` |
| `build/environments` | already good, keep |
| `build/tasks` | already good, keep |
| `build/run` (Run & deploy) | `run/models` + `run/deploy` + runtime section of `core/runtime` |
| `build/train` | `run/training` (keep dense `TrainingClient` API in Reference) |
| `build/advice` | `run/signal` |

Reference = the current `core/*` pages, minus `protocol` (which moves to Build).

## Status

### Done
- Created Build filler pages: `build/{index,protocol,run,train,advice}.mdx` (skeletons with frontmatter
  + a `<Note>` draft banner + a "to merge here" list pointing at source pages).
- Rewired `docs.json` navigation to the target structure above. `The Core` is relabeled **Reference**;
  `protocol` pulled out of it.
- Rewrote "Reading the docs" in `start/index.mdx` as a `<CardGroup>` linking every section.
- Nothing deleted: `start/overview`, `core/protocol`, and all `run/*` files still exist on disk and are
  reachable by URL (just not in the sidebar) so their content can be copied into the new pages.

### To do
1. **Fill the Build pages** by merging the source content per the mapping table, in docs-write voice.
2. **Move `core/ → reference/`** (file moves), update the `Reference` group paths in `docs.json`, and add
   redirects. Note: `docs.json` already has `"/v6/reference/:slug* → /v6/core/:slug*"` and several pages
   already link to `/v6/reference/*`, so the rename aligns paths with links already in use.
3. **Audit and rewrite all internal links + anchors.** Every `href="/v6/..."` that points at a moved page
   (`run/*`, `core/protocol`, `start/overview`) must be updated, and a redirect added in `docs.json` for
   each moved URL. This is the highest-rot risk; treat it as a first-class step, not cleanup.
4. **Delete the now-redundant source pages** (`start/overview`, `run/*`, old `core/protocol`) only after
   their content is merged and links are fixed.

## Open decisions
- Granularity of Build (current plan: spine + 6 pages, vs fewer/bigger pages).
- Whether `protocol` stays in Build or gets its own tiny "Concepts" group.
- Final name for **Advice** ("Designing tasks" or "Best practices" reads clearer in the sidebar).
