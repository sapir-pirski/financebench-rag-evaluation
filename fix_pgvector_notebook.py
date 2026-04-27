from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "rag_pgvector" / "assignment_2_rag_pgvector.ipynb"
DATASET_PATH = ROOT / "data" / "financebench_filtered.jsonl"
ARTIFACT_DIR = ROOT / "rag_pgvector" / "artifacts"
OUTPUT_DIR = ROOT / "rag_pgvector" / "outputs"


def read_notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def write_notebook(nb: dict) -> None:
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def cell_source(nb: dict, index: int) -> str:
    src = nb["cells"][index]["source"]
    return "".join(src) if isinstance(src, list) else str(src)


def set_cell_source(nb: dict, index: int, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    nb["cells"][index]["source"] = text.splitlines(keepends=True)


def replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        return text
    return text.replace(old, new, 1)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def apply_code_fixes() -> None:
    nb = read_notebook()
    task3 = cell_source(nb, 16)
    bonus = cell_source(nb, 32)

    task3 = replace_once(
        task3,
        """def build_chunks_from_pages(pages: list, *, chunk_size: int, chunk_overlap: int) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Created chunks: {len(chunks):,}")
    print("Example chunk metadata:")
    print(chunks[0].metadata)
    return chunks


def insert_chunks(
""",
        """def build_chunks_from_pages(pages: list, *, chunk_size: int, chunk_overlap: int) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(pages)
    print(f"Created chunks: {len(chunks):,}")
    print("Example chunk metadata:")
    print(chunks[0].metadata)
    return chunks


def stable_chunk_uid(metadata: dict, content: str, chunk_size: int, chunk_overlap: int) -> str:
    payload = json.dumps(
        {
            "doc_name": metadata.get("doc_name"),
            "page_number": metadata.get("page_number"),
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "content": content,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def insert_chunks(
""",
    )

    task3 = replace_once(
        task3,
        """                for offset, (chunk, vector) in enumerate(zip(batch, embeddings), start=start):
                    metadata = dict(chunk.metadata)
                    content_hash = hashlib.sha256(
                        chunk.page_content.replace("\\x00", "").encode("utf-8", errors="ignore")
                    ).hexdigest()[:16]
                    chunk_uid = (
                        f"{metadata.get('doc_name')}:{metadata.get('page_number')}:"
                        f"{offset}:{chunk_size}:{content_hash}"
                    )
                    metadata.update(
                        {
                            "chunk_index": int(offset),
                            "chunk_size": int(chunk_size),
                            "chunk_overlap": int(chunk_overlap),
                            "chunk_uid": chunk_uid,
                        }
                    )
                    rows.append(
                        {
                            "chunk_uid": chunk_uid,
                            "doc_name": metadata.get("doc_name"),
                            "company": metadata.get("company"),
                            "doc_period": None if metadata.get("doc_period") is None else str(metadata.get("doc_period")),
                            "page_number": int(metadata.get("page_number", 0)),
                            "chunk_index": int(offset),
                            "chunk_size": int(chunk_size),
                            "chunk_overlap": int(chunk_overlap),
                            "content": chunk.page_content.replace("\\x00", ""),
                            "embedding": pgvector_literal(vector),
                            "metadata": Jsonb(metadata),
                        }
                    )
""",
        """                for offset, (chunk, vector) in enumerate(zip(batch, embeddings), start=start):
                    metadata = dict(chunk.metadata)
                    content = chunk.page_content.replace("\\x00", "")
                    chunk_uid = stable_chunk_uid(metadata, content, chunk_size, chunk_overlap)
                    metadata.update(
                        {
                            "chunk_index": int(offset),
                            "chunk_size": int(chunk_size),
                            "chunk_overlap": int(chunk_overlap),
                            "chunk_uid": chunk_uid,
                        }
                    )
                    rows.append(
                        {
                            "chunk_uid": chunk_uid,
                            "doc_name": metadata.get("doc_name"),
                            "company": metadata.get("company"),
                            "doc_period": None if metadata.get("doc_period") is None else str(metadata.get("doc_period")),
                            "page_number": int(metadata.get("page_number", 0)),
                            "chunk_index": int(offset),
                            "chunk_size": int(chunk_size),
                            "chunk_overlap": int(chunk_overlap),
                            "content": content,
                            "embedding": pgvector_literal(vector),
                            "metadata": Jsonb(metadata),
                        }
                    )
""",
    )

    task3 = replace_once(
        task3,
        """    if rebuild:
        print(f"REBUILD_VECTORSTORE=true; truncating {table_name}")
        truncate_chunks(table_name)

    insert_chunks(
""",
        """    existing_rows = 0
    try:
        existing_rows = count_chunks(table_name)
    except Exception:
        existing_rows = 0
    if rebuild or existing_rows:
        reason = "REBUILD_VECTORSTORE=true" if rebuild else "Table is stale/incomplete"
        print(f"{reason}; truncating {table_name}")
        truncate_chunks(table_name)

    insert_chunks(
""",
    )

    task3 = replace_once(
        task3,
        """"chunk_count": int(len(chunks)),""",
        """"chunk_count": int(count_chunks(table_name)),""",
    )

    bonus = replace_once(
        bonus,
        """    if BONUS_REBUILD_INDICES:
        truncate_chunks(table_name)

    insert_chunks(
""",
        """    existing_rows = 0
    try:
        existing_rows = count_chunks(table_name)
    except Exception:
        existing_rows = 0
    if BONUS_REBUILD_INDICES or existing_rows:
        reason = "BONUS_REBUILD_INDICES=true" if BONUS_REBUILD_INDICES else "Table is stale/incomplete"
        print(f"{reason}; truncating {table_name}", flush=True)
        truncate_chunks(table_name)

    insert_chunks(
""",
    )

    bonus = replace_once(
        bonus,
        """"chunk_count": int(len(chunks)),""",
        """"chunk_count": int(count_chunks(table_name)),""",
    )

    set_cell_source(nb, 16, task3)
    set_cell_source(nb, 32, bonus)
    write_notebook(nb)


def update_discussions() -> None:
    nb = read_notebook()

    dataset_df = pd.read_json(DATASET_PATH, lines=True).sort_values("financebench_id", kind="stable")
    task1_df = pd.read_excel(OUTPUT_DIR / "assignment_2_naive_generation.xlsx")
    task5_df = pd.read_excel(OUTPUT_DIR / "assignment_2_run_and_compare.xlsx")
    task6_df = pd.read_excel(OUTPUT_DIR / "assignment_2_evaluation.xlsx")
    task7_df = pd.read_excel(OUTPUT_DIR / "assignment_2_improvement_cycles.xlsx")
    bonus_summary_df = pd.read_excel(OUTPUT_DIR / "assignment_2_bonus_multiscale_chunking.xlsx", sheet_name="summary")

    task5_raw = json.loads((ARTIFACT_DIR / "task5_rag_answers_pgvector_raw.json").read_text())
    task6_correctness_raw = json.loads((ARTIFACT_DIR / "task6_correctness_pgvector_raw.json").read_text())

    task1_counts = task1_df["verdict"].value_counts().to_dict()
    task1_correct = int(task1_counts.get("correct", 0))
    task1_refused = int(task1_counts.get("refused", 0))
    task1_total = int(len(task1_df))

    task5_join = dataset_df[["financebench_id", "doc_name", "evidence_page_nums"]].merge(
        pd.DataFrame(task5_raw), on="financebench_id", how="inner"
    )

    def as_list(value):
        if isinstance(value, list):
            return value
        if pd.isna(value):
            return []
        return [value]

    task5_doc_hits = []
    task5_page_hits = []
    for _, row in task5_join.iterrows():
        expected_doc = row["doc_name"]
        expected_pages = {int(page) for page in as_list(row["evidence_page_nums"])}
        retrieved_chunks = row["retrieved_chunks"]
        task5_doc_hits.append(any(chunk.get("doc_name") == expected_doc for chunk in retrieved_chunks))
        task5_page_hits.append(
            any(
                chunk.get("doc_name") == expected_doc
                and int(chunk.get("page_number")) in expected_pages
                for chunk in retrieved_chunks
            )
        )

    sample_ids = ["financebench_id_00005", "financebench_id_00283", "financebench_id_00288"]
    sample_lines = []
    sample_lookup = task5_join.set_index("financebench_id")
    for financebench_id in sample_ids:
        row = sample_lookup.loc[financebench_id]
        expected_doc = row["doc_name"]
        expected_pages = [int(page) for page in as_list(row["evidence_page_nums"])]
        retrieved = [(chunk.get("doc_name"), int(chunk.get("page_number"))) for chunk in row["retrieved_chunks"]]
        any_page = any(doc == expected_doc and page in expected_pages for doc, page in retrieved)
        sample_lines.append(
            f"- `{financebench_id}`: top-5 retrieved {expected_doc} with pages {retrieved}; exact evidence-page hit = {'yes' if any_page else 'no'}."
        )

    correctness_rate = float((task6_df["correctness"] == "correct").mean())
    support_rate = float((task6_df["support_verdict"] == "supported").mean())
    citation_rate = float((task6_df["support_citation_status"] == "valid").mean())
    faithfulness_rate = float(task6_df["faithfulness"].dropna().mean())
    page_hit_1 = float(task6_df["page_hit_at_1"].mean())
    page_hit_3 = float(task6_df["page_hit_at_3"].mean())
    page_hit_5 = float(task6_df["page_hit_at_5"].mean())
    judge_model = sorted({row["judge_model"] for row in task6_correctness_raw})[0]

    baseline = task7_df.iloc[0]
    prompt_no_few_shot = task7_df.loc[task7_df["experiment"] == "prompt_no_few_shot"].iloc[0]
    strict_prompt = task7_df.loc[task7_df["experiment"] == "strict_prompt"].iloc[0]
    top_k_8 = task7_df.loc[task7_df["experiment"] == "top_k_8"].iloc[0]
    reranker = task7_df.loc[task7_df["experiment"] == "bge_reranker_top4"].iloc[0]

    best_correctness_row = task7_df.loc[task7_df["correctness"].idxmax()]
    best_faithfulness_row = task7_df.loc[task7_df["faithfulness"].idxmax()]

    bonus_summary = {
        str(row["chunk_size"]): {
            "page_hit_at_5": float(row["page_hit_at_5"]),
            "hit_count": int(row["hit_count"]),
            "unique_winner_count": int(row["unique_winner_count"]),
        }
        for _, row in bonus_summary_df.iterrows()
    }
    oracle = bonus_summary["oracle_any_size"]

    set_cell_source(
        nb,
        13,
        f"""### Task 1 Discussion

The naive baseline remained intentionally conservative. It produced {task1_correct} correct answer out of {task1_total} questions and refused the other {task1_refused}, so it avoided inventing filing-specific numbers but was not useful for most FinanceBench questions. The only clear success in this 10-question sample was the JPM gross-margin question, where the model correctly said gross margin is not a meaningful metric for a bank.

This is the right baseline to compare RAG against: it is reasonably safe, but it cannot answer filing-grounded questions such as working capital, segment revenue, or evidence-page calculations without retrieval.
""",
    )

    set_cell_source(
        nb,
        14,
        """## Task 2 - RAG Reminder

I kept the same high-level design as the FAISS notebook and changed only the storage backend.

- Chunking: `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=150`. This is large enough to keep short tables, captions, and nearby narrative together, while the overlap reduces page-boundary fragmentation for ratio and delta questions.
- Embeddings: `BAAI/bge-small-en-v1.5` with normalized embeddings. It is inexpensive to run locally, produces 384-dimensional vectors that fit cleanly into pgvector, and keeps the retrieval setup aligned with the FAISS notebook.
- Vector store: cloud PostgreSQL + pgvector instead of local FAISS. Because this notebook runs in a cloud environment, pgvector gives persistent remote storage, SQL metadata columns, and the same `similarity_search(query, k)` interface used downstream by the shared RAG/evaluation code.
""",
    )

    set_cell_source(
        nb,
        18,
        "### Task 3 Observations\n\n"
        "The retrieval audit shows that pgvector is usually getting the right filing, but not always the exact evidence page. In the three fixed sample questions, the correct document appeared in the top five all three times, and the exact evidence page appeared in two of the three cases.\n\n"
        + "\n".join(sample_lines)
        + "\n\nThat pattern matches the later evaluation numbers: document-level narrowing is often good enough to surface the right filing, but page-level precision is still the main bottleneck for final answer quality.\n",
    )

    set_cell_source(
        nb,
        24,
        f"""### Task 5 Discussion

On this 10-question comparison set, top-5 pgvector retrieval found the correct document for {sum(task5_doc_hits)}/10 questions and the exact evidence page for {sum(task5_page_hits)}/10. That is enough to improve over the naive baseline on some questions, but it is still too weak to make the generator reliable.

The clearest gain is that RAG turned several naive refusals into concrete filing-grounded answers, such as Verizon capital intensity and MGM's EBITDAR region question. The main failure mode is still retrieval precision: Corning, PayPal, JPM segment revenue, and Pfizer PPNE all missed the evidence page, so the generator either refused or answered from incomplete context. Pfizer Upjohn is the other important failure case: the correct page was retrieved, but the model still answered with the wrong amount, which shows that answer synthesis is a separate bottleneck even when retrieval succeeds.
""",
    )

    set_cell_source(
        nb,
        27,
        f"""### Task 6 Notes

This saved run used `{judge_model}` as the judge model recorded in the raw Task 6 artifacts. On the 100 filtered FinanceBench questions, the pgvector baseline reached {pct(correctness_rate)} correctness, {pct(support_rate)} fully supported answers, {pct(citation_rate)} valid citations, and mean faithfulness {faithfulness_rate:.3f} on the first 20 scored examples.

Retrieval remained the limiting factor: `page_hit_at_1={pct(page_hit_1)}`, `page_hit_at_3={pct(page_hit_3)}`, and `page_hit_at_5={pct(page_hit_5)}`. In other words, even at `k=5`, the evidence page was present for only about one third of questions. That explains why support is higher than correctness: some answers are grounded in retrieved text, but not in the exact FinanceBench evidence needed to match the gold answer.
""",
    )

    set_cell_source(
        nb,
        28,
        """## Task 7 - Improvement Cycles

I kept the same four experiments as the FAISS notebook so that the pgvector backend stays comparable to the local FAISS version.

1. `prompt_no_few_shot`: remove only the few-shot examples while reusing the Task 6 retrieved chunks.
2. `strict_prompt`: strengthen the evidence/citation instructions while reusing the Task 6 retrieved chunks.
3. `top_k_8`: increase generator context from 5 retrieved chunks to 8.
4. `bge_reranker_top4`: retrieve a larger pgvector candidate set, rerank it with `BAAI/bge-reranker-base`, and send only the top 4 chunks to the generator.
""",
    )

    set_cell_source(
        nb,
        30,
        f"""### Task 7 Notes

The best overall result in this run came from `prompt_no_few_shot`, which improved correctness from {pct(float(baseline['correctness']))} to {pct(float(prompt_no_few_shot['correctness']))}, support from {pct(float(baseline['support_supported_rate']))} to {pct(float(prompt_no_few_shot['support_supported_rate']))}, valid-citation rate from {pct(float(baseline['support_valid_citation_rate']))} to {pct(float(prompt_no_few_shot['support_valid_citation_rate']))}, and faithfulness from {float(baseline['faithfulness']):.3f} to {float(prompt_no_few_shot['faithfulness']):.3f}. In this pgvector run, the few-shot examples were not helping; they appear to have constrained the answer style more than they improved reasoning.

`strict_prompt` gave only a small correctness lift ({pct(float(strict_prompt['correctness']))}) and slightly reduced faithfulness to {float(strict_prompt['faithfulness']):.3f}, so the stricter instructions mainly increased caution without fixing retrieval. `top_k_8` modestly improved correctness to {pct(float(top_k_8['correctness']))} and raised `page_hit_at_8` to {pct(float(top_k_8['page_hit_at_8']))}, which is directionally useful but still limited by noisy extra context. The reranker was the weakest change in this run: `bge_reranker_top4` reduced `page_hit_at_1` to {pct(float(reranker['page_hit_at_1']))} and did not beat the simpler baseline on correctness or faithfulness.

Overall, the best correctness experiment was `{best_correctness_row['experiment']}` at {pct(float(best_correctness_row['correctness']))}, and the best faithfulness experiment was `{best_faithfulness_row['experiment']}` at {float(best_faithfulness_row['faithfulness']):.3f}. The main takeaway is that prompt changes can help a little, but retrieval quality is still the dominant constraint.
""",
    )

    set_cell_source(
        nb,
        33,
        f"""### Bonus Notes

Across chunk sizes, `1000` was the best single fixed setting in this run with `page-hit@5={pct(bonus_summary['1000']['page_hit_at_5'])}` ({bonus_summary['1000']['hit_count']} hits out of 100 questions). Both `500` and `2000` reached {pct(bonus_summary['500']['page_hit_at_5'])} / {pct(bonus_summary['2000']['page_hit_at_5'])}, so the middle chunk size gave the best average retrieval precision.

That said, chunk-size choice is not uniform across questions. The oracle that picks the best chunk size per question would reach {pct(oracle['page_hit_at_5'])}, and {oracle['unique_winner_count']} questions showed disagreement across chunk sizes. The unique-winner counts also spread across all three settings (`500`: {bonus_summary['500']['unique_winner_count']}, `1000`: {bonus_summary['1000']['unique_winner_count']}, `2000`: {bonus_summary['2000']['unique_winner_count']}), which suggests that one global chunk size is a compromise rather than a universal optimum.
""",
    )

    write_notebook(nb)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-code-fixes", action="store_true")
    parser.add_argument("--update-discussions", action="store_true")
    args = parser.parse_args()

    if args.apply_code_fixes:
        apply_code_fixes()
    if args.update_discussions:
        update_discussions()
