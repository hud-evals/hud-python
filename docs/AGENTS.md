# Writing HUD docs

Guidance for any human or agent editing this docs site (Mintlify). Read this before adding or restructuring pages.

## What the docs are for

The docs are **the product surface for agents**, not just a human reference. Most readers arrive mid-site from a search or an LLM, and many "readers" are coding agents building HUD environments on a user's behalf (via the `skill.md` and the docs MCP). So every page must be a valid entry point, state its own model, and be literally correct — an agent will copy what it reads.

## The model and terminology (one name per concept)

The whole SDK is one atom: a **trace** is one graded evaluation of a **task** in an **environment**. Keep these names exact; do not introduce synonyms.

| Concept | Use this | Never |
|---------|----------|-------|
| Where the agent acts | **environment** | "gym", "sandbox" (sandbox = the substrate instance) |
| A connection the env exposes | **capability** (`ssh`/`mcp`/`cdp`/`rfb`/`robot`) | "tool" (tools belong to the harness) |
| The model's tool layer over a capability | **harness** | — |
| The prompt-then-reward generator | **task** | "scenario" (v5) |
| One graded evaluation (the recorded unit) | **trace** | "run" as the noun |
| The live SDK handle for a trace | **`Run`** (code only) | — |
| The act of running one | **rollout** | — |
| A named dataset of tasks | **taskset** | — |
| The built artifact | **image** / **container** | "box" |
| Where a container is provisioned | **runtime** / **provider** | — |

If you rename a concept in the SDK, update this table, every page, and `skill.md` in the same pass — terminology drift is the most common docs bug here.

## Page quality rubric

Each principle has a test, because the failures that matter are the silent ones (a wrong example reads fine until someone runs it).

1. **Executable truth.** Every command and code block on a golden path must run against the *current* SDK before it ships. Symbol-grepping is necessary but not sufficient. Test: concatenate a page's code blocks and run them.
2. **Self-contained pages.** If a page's run command targets `env.py`, everything that command needs is on that page — no invisible dependency on a later file (the classic cookbook trap: defining a task but never minting a runnable from it, so `hud eval` finds nothing).
3. **Runs on a contributor's laptop.** Examples work on macOS/Windows local iteration or carry an explicit "Linux/in-image only" callout *before* the code. No bare absolute paths (`/workspace`) in locally-run examples.
4. **Verify APIs against source.** Never invent a symbol, signature, or flag. If you can't find it in `hud/`, it's wrong. Re-read the source; the API moves.
5. **One job per page.** Concept *or* how-to *or* reference — not all three. Reference is exhaustive; the learning path is singular.
6. **Model before mechanics.** State the one concept, then the API.
7. **Tiny time-to-first-success.** A copy-pasteable working result early.
8. **One golden path.** Be opinionated. Don't present five ways to do one thing in a tutorial (reference may enumerate).
9. **Progressive disclosure.** 80% path clean; edge cases in a `<Note>`/`<Warning>`/`<Accordion>`.
10. **No DRY-by-copy.** Content owned by another page is *linked*, not restated. Repeated blocks (prereqs, the capability table, the signal checklist) belong in `/snippets` and are `<Snippet>`-included so they can't drift.
11. **Warnings can't contradict the example.** If a page warns against an anti-pattern, its own golden example must not embody it. Show the correct version; the anti-pattern appears only as a labeled counter-example.
12. **Skill–docs lockstep.** Every `skill.md` trigger cites a page+section that exists and agrees; every doctrine rule in the docs has a skill trigger.

## Validate before shipping

```bash
# 1. docs.json parses and every nav page exists on disk
uv run python -c "import json,pathlib; r=pathlib.Path('docs'); d=json.load(open(r/'docs.json',encoding='utf-8')); \
nav=[]; \
walk=lambda n: [walk(v) if k!='pages' else nav.extend(p for p in v if isinstance(p,str)) for k,v in n.items()] if isinstance(n,dict) else [walk(i) for i in n] if isinstance(n,list) else None; \
walk(d['navigation']); \
print('missing:', [p for p in nav if not ((r/(p+'.mdx')).exists() or (r/(p+'.md')).exists())])"

# 2. build + link check (Mintlify CLI)
npx mint@latest dev          # surfaces build errors
npx mint@latest broken-links
```

Also: run any code block you added; grep `hud/` for every symbol you reference.

## Styling and customization (Mintlify)

Site-wide config lives in `docs.json`; component styling in `custom.css` (project root). Favor built-in components over custom ones.

**`docs.json` levers (low-effort, high-impact):**

| Lever | Options | Effect |
|-------|---------|--------|
| `theme` | `mint · maple · palm · willow · linden · almond · aspen · sequoia · luma` | Whole layout/nav personality (`linden` = mono/terminal; `aspen`/`sequoia` = complex nav + custom components) |
| `background.decoration` | `gradient · grid · windows` | Ambient texture |
| `styling.codeblocks` | `system · dark` | `dark` = always-dark codeblocks (Stripe-style, code-forward) |
| `styling.eyebrows` | `section · breadcrumbs` | `breadcrumbs` reinforces every-page-is-an-entry-point |
| `styling.latex` | bool | Math rendering (the signal/IRT pages) |
| `fonts.family` | string | Brand typography |
| `appearance.default` | `system/light/dark` | Default color mode |
| `interaction.drilldown` | bool | Expandable nested sidebar |
| `contextual.options` / `display` | `header`/`toc` | The Copy / Claude / ChatGPT / Perplexity buttons — the docs-as-agent-surface lever; keep prominent |
| `banner` | content/type/color | Top banner (e.g. advertise `npx skills add docs.hud.ai`) |

**Navigation patterns** (mix/nest in `navigation`): `groups` (default), `tabs` (distinct audiences), `anchors` (persistent sidebar-top links), `dropdowns` (section switcher), `products` (multi-product), `versions` (we use this for v6/v5), `languages`.

**Content components** (MDX): `<Steps>` (tutorials), `<CodeGroup>` (per-model tabs), `<Tabs>`, `<Accordion>`/`<AccordionGroup>`, `<Card>`/`<Columns>`, `<Panel>` (persistent right rail → true three-column Stripe layout), `<Note>`/`<Warning>`/`<Tip>`/`<Check>`, `<Update>` (changelog), `<Frame>` (images), `<Snippet>` (reusable includes), `<Mermaid>` diagrams.

**Deep customization:** `custom.css` for component restyling (e.g. echo the platform's brutalist + glass design language); custom React components on `aspen`/`sequoia`; `$ref` to split `docs.json` as it grows.

## v5 vs v6

`docs.json` serves two `versions` on the SDK tab: **v6** (default, under `docs/v6/`) and **v5** (legacy, the original top-level pages). Never edit v5 pages; never change which version is `default` without sign-off. New work goes under `docs/v6/`.
