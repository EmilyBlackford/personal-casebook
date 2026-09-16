# Casebook

Casebook is a document Q&A system for a consulting firm's internal knowledge
base. A user asks a question in plain English and receives an answer grounded
in the firm's documents, with a citation to the source.

Casebook is a tool-using agent, not a single source look-up. It has access to:
- `retrieve`: search over the firm's internal document corpus (local tool)
- `web_search`: search the live web for current information outside the corpus, provided exclusively by an MCP server

The central constraint: every answer must be grounded in a passage actually returned by `retrieve` or `web_search`, with a citation to its source. Casebook never answers from unstates/parametric knowledge, and never blends an ungrounded claim in with a cited one. If neither tool returns the information, Casebook says so plainly rather than filling the gap.

Tool selection must be driven by clear, non-overlapping tool descriptions,
not by hard-coded keyword rules. When adding, editing, or connecting a new
tool, always:
- Write or review its description so the model can tell when to use it
  versus the other tools
- Preserve the shared result shape (`content`, `source`, `title` or
  equivalent) so downstream code and citation logic don't need to know which
  tool produced a given passage, or whether it was local or MCP-connected
- Never write logic that lets a tool bypass the citation requirement above
- Never write code that connects to `web_search` any way other than through
  the MCP client session

Never generate code or prompts that weaken any of these constraints.

## Stack

<!--
List what you have actually chosen, so the tool does not suggest plausible
alternatives you are not using. Update it when the stack grows.
-->

- Python, LangChain (LangGraph from Week 10)
- Gemini 2.5 Flash via Vertex AI; Pro only where we say so
- Vertex AI text-embedding-004 for embeddings
- pgvector on Cloud SQL for vector storage
- RAGAS for evaluation, Langfuse for tracing
- `mcp` and `langchain_mcp_adapters` for MCP client connections
- Tavily for web search (accessed only via the `web_search` MCR server, never called directly)

## Key conventions

<!--
Record interfaces and decisions that must not drift. If other code (or a
future week) depends on a signature, it belongs here.
-->

- `retrieve(query: str, vector_store, k: int = 4)` returns
  `[{"content": str, "source": str, "title": str}]`. This signature is a contract. Do not change it.
- `web_search` (MCP tool, not a local function) return results in the same shape: `{"content": str, "source": str, "title": str}`. Any tool added to Casebook must conform to this shape so downstream code and citation logic don't need to know which tool produced a given passage.
- Every answer-generation prompt instructs the model to cite sources and to say when neither `retrieve` nor `web_search` contains the answer.
- Connection strings use the `postgresql+psycopg://` prefix.

## Grill me before you build

<!--
The "grilling" skill from https://github.com/mattpocock/skills, used here
as a standing instruction because the tool reading this file may not
support skills directly. Wording kept as close to the source skill as
this project's no-em-dash rule allows.
-->

Interview me relentlessly about every aspect of this plan until we reach a shared understanding. Walk down each branch of the decision tree, resolving dependencies between decisions one-by-one. For each question, provide your recommended answer.

Ask the questions one at a time, waiting for feedback on each question before continuing. Asking multiple questions at once is bewildering.

If a fact can be found by exploring the environment (filesystem, tools, etc.), look it up rather than asking me. The decisions, though, are mine. Put each one to me and wait for my answer.

Do not act on it until I confirm we have reached a shared understanding.

## What you should do

<!--
Tell the tool how you want to work with it. These lines set the default
behaviour for every session.
-->

- Explain what generated code does and why, not just what to paste.
- Point out when a change would affect the RAGAS scores or the retrieve()
  contract.

## What you should not do

- Do not write or edit `golden_dataset.json`. Reference answers are
  human-written. That is what makes the evaluation meaningful.
- Do not add new dependencies or frameworks without asking first.
- Do not generate code we have not discussed. Plan first, then build.
