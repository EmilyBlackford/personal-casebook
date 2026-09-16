# Casebook

<!--
Start with what the project is and the constraint that shapes it. The
constraint matters most here: a tool that does not know about the citation
rule will happily generate code that answers from the model's training data,
and it will look fine until you test it.
-->

Casebook is a document Q&A system for a consulting firm's internal knowledge
base. A user asks a question in plain English and receives an answer grounded in the firm's documents, with a citation to the source.

The central constraint: answers come from retrieved passages only. If the
information is not in the corpus, Casebook says so. Never generate code or
prompts that weaken this constraint.

## Stack

- Python, LangChain (LangGraph from Week 10)
- Gemini 2.5 Flash via Vertex AI; Pro only where we say so
- Vertex AI text-embedding-004 for embeddings
- pgvector on Cloud SQL for vector storage
- RAGAS for evaluation, Langfuse for tracing

## Key conventions

<!--
Record interfaces and decisions that must not drift. If other code (or a
future week) depends on a signature, it belongs here.
-->

- `retrieve(query: str, vector_store, k: int = 4)` returns
  `[{"content": str, "source": str, "score": float}]`. This signature is a
  contract. Do not change it.
- Every answer-generation prompt instructs the model to cite sources and to
  say when the corpus does not contain the answer.
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
