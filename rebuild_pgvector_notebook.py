from __future__ import annotations

import json
import textwrap
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FAISS_NOTEBOOK = ROOT / "rag_faiss" / "assignment_2_rag_faiss.ipynb"
PGVECTOR_NOTEBOOK = ROOT / "rag_pgvector" / "assignment_2_rag_pgvector.ipynb"


def read_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cell_text(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def new_markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": lines(textwrap.dedent(text).strip("\n")),
    }


def new_code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": lines(textwrap.dedent(text).strip("\n")),
    }


def replace_exact(source: str, old: str, new: str) -> str:
    if old not in source:
        raise ValueError(f"Expected snippet not found:\n{old[:200]}")
    return source.replace(old, new, 1)


def build_markdown_cells() -> list[str]:
    return [
        """
        ## Dataset Preparation

        Load FinanceBench, drop `metrics-generated`, normalize evidence metadata, repair PDF filenames/links, and write the canonical filtered dataset used by both notebook variants.
        """,
        """
        ## Task 1 - Naive Generation

        Use `meta-llama/Llama-3.3-70B-Instruct` through an OpenAI-compatible Nebius endpoint to answer the first five `domain-relevant` and first five `novel-generated` questions without retrieval.
        """,
        """
        ### Task 1 Discussion

        This section follows the same workflow as the FAISS notebook. Re-run the generation cells if you want to refresh the saved outputs after changing the model, endpoint, or prompt configuration.
        """,
        """
        ## Task 2 - RAG Reminder

        Indexing stores the corpus as chunk embeddings plus metadata, retrieval selects likely evidence for each query, and generation answers using only the retrieved context.
        """,
        """
        ## Task 3 - Index Documents with pgvector

        This notebook mirrors the FAISS notebook's document preparation steps, but stores embeddings in a cloud PostgreSQL/pgvector table instead of a local FAISS index.
        """,
        """
        ### Task 3 Observations

        The retrieval audit uses the same three sample questions as the FAISS notebook so retrieval behaviour can be compared directly across backends.
        """,
        """
        ## Task 4 - RAG Pipeline

        Retrieve chunks from pgvector, format them with the same prompt template used by the FAISS notebook, and generate answers with the configured LLM endpoint.
        """,
        """
        ### Task 4 Usage

        Run `answer_with_rag("your question", k=4)` after Task 3. The returned dictionary includes the model answer and the retrieved chunks used as context.
        """,
        """
        ## Task 5 - Run and Compare

        Run the same 10 Task 1 questions through the pgvector-backed RAG pipeline and export the side-by-side comparison to Excel.
        """,
        """
        ### Task 5 Discussion

        Use this section to summarize where retrieval helped, where it still failed, and which questions remained bottlenecked by evidence-page recall.
        """,
        """
        ## Task 6 - Evaluation

        Evaluate the pgvector RAG pipeline with the same correctness, support, faithfulness, and retrieval page-hit metrics used in the FAISS notebook.
        """,
        """
        ### Task 6 Notes

        The evaluation logic is intentionally kept as close as possible to the FAISS notebook. The main implementation difference is that retrieved chunks come from PostgreSQL/pgvector rather than a local FAISS store.
        """,
        """
        ## Task 7 - Improvement Cycles

        Use the same experiment structure as the FAISS notebook so prompt and retrieval changes can be compared across backends with minimal code drift.
        """,
        """
        ### Task 7 Notes

        The baseline and experiments reuse the Task 6 evaluation logic. Only the retrieval/index backend differs: pgvector instead of FAISS.
        """,
        """
        ## Bonus - Multi-Scale Chunking

        Compare `page-hit@5` across chunk sizes `500`, `1000`, and `2000` while keeping the embedding model and overlap policy fixed.
        """,
        """
        ### Bonus Notes

        The bonus section mirrors the FAISS notebook's comparison structure, but builds/reuses separate pgvector tables for each chunk size instead of local FAISS directories.
        """,
    ]


def build_code_cell_2() -> str:
    return """
    import re
    import subprocess
    import sys
    from importlib.metadata import PackageNotFoundError, version

    REQUIRED_PACKAGES = [
        "urllib3<2",
        "jupyterlab",
        "ipykernel",
        "ipywidgets",
        "pandas",
        "openpyxl",
        "python-dotenv",
        "tqdm",
        "openai",
        "fsspec",
        "huggingface-hub",
        "langchain-community",
        "langchain-huggingface",
        "langchain-text-splitters",
        "sentence-transformers",
        "pypdf",
        "cryptography",
        "ragas",
        "eval_type_backport",
        "psycopg[binary]",
        "pgvector",
        "sshtunnel",
    ]


    def requirement_is_satisfied(requirement: str) -> bool:
        match = re.match(r"^([A-Za-z0-9_.-]+)(.*)$", requirement)
        if not match:
            return False
        dist_name, spec = match.groups()
        try:
            installed_version = version(dist_name)
        except PackageNotFoundError:
            return False
        if spec == "<2":
            major = installed_version.split(".", 1)[0]
            return major.isdigit() and int(major) < 2
        return True


    missing_requirements = [requirement for requirement in REQUIRED_PACKAGES if not requirement_is_satisfied(requirement)]
    if missing_requirements:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing_requirements], check=True)
        print(f"Installed missing notebook packages: {missing_requirements}")
    else:
        print("All required notebook packages are already installed in the active kernel.")
    """


def build_code_cell_3() -> str:
    return """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from getpass import getpass
    from pathlib import Path

    import ast
    import hashlib
    import json
    import math
    import os
    import platform
    import queue
    import re
    import shutil
    import time
    import urllib.parse
    import urllib.request
    from threading import Lock, Thread

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    import certifi
    import pandas as pd
    import psycopg
    import torch
    from dotenv import load_dotenv
    from IPython.display import display
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from openai import AsyncOpenAI, OpenAI
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness
    from sentence_transformers import CrossEncoder
    from tqdm import tqdm
    """


def build_code_cell_4(faiss_source: str) -> str:
    source = faiss_source

    source = replace_exact(
        source,
        """CURRENT_DIR = Path.cwd()
if CURRENT_DIR.name == "rag_faiss":
    REPO_ROOT = CURRENT_DIR.parent
    FAISS_ROOT = CURRENT_DIR
else:
    REPO_ROOT = CURRENT_DIR
    FAISS_ROOT = CURRENT_DIR / "rag_faiss"

DATA_DIR = REPO_ROOT / "data"
ARTIFACT_DIR = FAISS_ROOT / "artifacts"
OUTPUT_DIR = FAISS_ROOT / "outputs"
""",
        """CURRENT_DIR = Path.cwd()
if CURRENT_DIR.name == "rag_pgvector":
    REPO_ROOT = CURRENT_DIR.parent
    PGVECTOR_ROOT = CURRENT_DIR
else:
    REPO_ROOT = CURRENT_DIR
    PGVECTOR_ROOT = CURRENT_DIR / "rag_pgvector"

DATA_DIR = REPO_ROOT / "data"
ARTIFACT_DIR = PGVECTOR_ROOT / "artifacts"
OUTPUT_DIR = PGVECTOR_ROOT / "outputs"
""",
    )
    source = replace_exact(
        source,
        'os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"',
        'os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"\nos.environ.setdefault("TOKENIZERS_PARALLELISM", "false")',
    )
    source = replace_exact(
        source,
        'NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"',
        'NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1")',
    )
    source = replace_exact(
        source,
        """TASK1_RAW_PATH = ARTIFACT_DIR / "task1_naive_generation_raw.json"
TASK1_XLSX_PATH = OUTPUT_DIR / "assignment_2_naive_generation.xlsx"
FORCE_TASK1_REGEN = False
""",
        """TASK1_RAW_PATH = ARTIFACT_DIR / "task1_naive_generation_pgvector_raw.json"
TASK1_XLSX_PATH = OUTPUT_DIR / "assignment_2_naive_generation.xlsx"
FORCE_TASK1_REGEN = False
""",
    )
    source = replace_exact(
        source,
        """VECTORSTORE_DIR = DATA_DIR / "vectorstore" / "financebench_bge_small_v1_5"
VECTORSTORE_MANIFEST_PATH = VECTORSTORE_DIR / "manifest.json"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
REBUILD_VECTORSTORE = False
""",
        """VECTORSTORE_DIR = DATA_DIR / "vectorstore" / "financebench_bge_small_v1_5_pgvector"
VECTORSTORE_MANIFEST_PATH = VECTORSTORE_DIR / "manifest.json"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
REBUILD_VECTORSTORE = False
PGVECTOR_TABLE_PREFIX = "financebench_chunks_bge_small"
BASE_TABLE = f"{PGVECTOR_TABLE_PREFIX}_{CHUNK_SIZE}"
EMBEDDING_DIM = 384
""",
    )
    source = replace_exact(
        source,
        'RAG_VECTORSTORE_DIR = DATA_DIR / "vectorstore" / "financebench_bge_small_v1_5"',
        "RAG_VECTORSTORE_DIR = VECTORSTORE_DIR",
    )
    source = replace_exact(
        source,
        'TASK5_RAW_PATH = ARTIFACT_DIR / "task5_rag_answers_raw.json"',
        'TASK5_RAW_PATH = ARTIFACT_DIR / "task5_rag_answers_pgvector_raw.json"',
    )
    source = replace_exact(
        source,
        'TASK6_RAG_ALL_RAW_PATH = ARTIFACT_DIR / "task6_rag_all_answers_raw.json"',
        'TASK6_RAG_ALL_RAW_PATH = ARTIFACT_DIR / "task6_rag_all_answers_pgvector_raw.json"',
    )
    source = replace_exact(
        source,
        'TASK6_CORRECTNESS_RAW_PATH = ARTIFACT_DIR / "task6_correctness_raw.json"',
        'TASK6_CORRECTNESS_RAW_PATH = ARTIFACT_DIR / "task6_correctness_pgvector_raw.json"',
    )
    source = replace_exact(
        source,
        'TASK6_SUPPORT_RAW_PATH = ARTIFACT_DIR / "task6_support_raw.json"',
        'TASK6_SUPPORT_RAW_PATH = ARTIFACT_DIR / "task6_support_pgvector_raw.json"',
    )
    source = replace_exact(
        source,
        'TASK6_FAITHFULNESS_RAW_PATH = ARTIFACT_DIR / "task6_faithfulness_raw.json"',
        'TASK6_FAITHFULNESS_RAW_PATH = ARTIFACT_DIR / "task6_faithfulness_pgvector_raw.json"',
    )
    source = replace_exact(
        source,
        "Changed only the number of FAISS chunks sent to the generator from k=5 to k=8.",
        "Changed only the number of pgvector chunks sent to the generator from k=5 to k=8.",
    )
    source = replace_exact(source, '"retrieval_mode": "faiss"', '"retrieval_mode": "vectorstore"')
    source = replace_exact(
        source,
        "Added BAAI/bge-reranker-base: retrieve top-20 with FAISS, then rerank to top-4 for generation.",
        "Added BAAI/bge-reranker-base: retrieve top-20 with pgvector, then rerank to top-4 for generation.",
    )
    source = replace_exact(
        source,
        "Reranking should improve top-ranked evidence quality by selecting the best four chunks from the initial FAISS top-20 candidates.",
        "Reranking should improve top-ranked evidence quality by selecting the best four chunks from the initial pgvector top-20 candidates.",
    )

    pgvector_cloud_block = '''

    # Cloud pgvector backend configuration
    PG_USE_SSH_TUNNEL = os.getenv("PG_USE_SSH_TUNNEL", "false").strip().lower() in {"1", "true", "yes"}
    PGHOST = os.getenv("PGHOST", "your-postgres-host")
    PGPORT = int(os.getenv("PGPORT", "5432"))
    PGDATABASE = os.getenv("PGDATABASE", "your_database")
    PGUSER = os.getenv("PGUSER", "your_database_user")
    PGSSLMODE = os.getenv("PGSSLMODE", "verify-full")
    PGSSLROOTCERT = os.getenv("PGSSLROOTCERT", str(DATA_DIR / "nebius_msp_ca.pem"))

    SSH_HOST = os.getenv("SSH_HOST", "")
    SSH_PORT = int(os.getenv("SSH_PORT", "22"))
    SSH_USER = os.getenv("SSH_USER", "")
    SSH_KEY_PATH = os.getenv("SSH_KEY_PATH", "")
    SSH_KEY_PASSPHRASE = os.getenv("SSH_KEY_PASSPHRASE") or None
    REMOTE_PGHOST = os.getenv("REMOTE_PGHOST", "127.0.0.1")
    REMOTE_PGPORT = int(os.getenv("REMOTE_PGPORT", "5432"))


    def require_config_values(values: dict[str, object], *, source: str) -> None:
        missing = [name for name, value in values.items() if value in {None, ""}]
        if missing:
            raise RuntimeError(
                f"Missing required {source} values: " + ", ".join(missing) +
                ". Fill them before continuing."
            )


    def safe_table_name(name: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Unsafe SQL identifier: {name!r}")
        return name


    def pgvector_literal(values) -> str:
        return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


    def build_embedding_model(show_progress: bool = False) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": LOCAL_TORCH_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
            show_progress=show_progress,
        )


    _ssh_tunnel = None


    def pg_connection_params() -> dict:
        global _ssh_tunnel
        password = os.getenv("PGPASSWORD")
        if not password:
            raise ValueError("PGPASSWORD is not set. Add it to .env before connecting to PostgreSQL.")

        host = PGHOST
        port = PGPORT

        if PG_USE_SSH_TUNNEL:
            from sshtunnel import SSHTunnelForwarder

            require_config_values(
                {
                    "SSH_HOST": SSH_HOST,
                    "SSH_USER": SSH_USER,
                    "SSH_KEY_PATH": SSH_KEY_PATH,
                },
                source="notebook SSH configuration",
            )
            if _ssh_tunnel is None:
                _ssh_tunnel = SSHTunnelForwarder(
                    (SSH_HOST, SSH_PORT),
                    ssh_username=SSH_USER,
                    ssh_pkey=SSH_KEY_PATH,
                    ssh_private_key_password=SSH_KEY_PASSPHRASE,
                    remote_bind_address=(REMOTE_PGHOST, REMOTE_PGPORT),
                    local_bind_address=("127.0.0.1", 0),
                )
                _ssh_tunnel.start()
                print(f"SSH tunnel started on 127.0.0.1:{_ssh_tunnel.local_bind_port}")
            host = "127.0.0.1"
            port = _ssh_tunnel.local_bind_port

        params = {
            "host": host,
            "port": port,
            "dbname": PGDATABASE,
            "user": PGUSER,
            "password": password,
            "sslmode": PGSSLMODE,
            "row_factory": dict_row,
        }
        cert_path = Path(PGSSLROOTCERT)
        if cert_path.exists():
            params["sslrootcert"] = str(cert_path.resolve())
        return params


    def connect_pg():
        return call_with_retries(
            lambda: psycopg.connect(**pg_connection_params()),
            "PostgreSQL connect",
            max_retries=API_MAX_RETRIES,
        )


    class PGVectorFinanceBenchStore:
        def __init__(self, table_name: str, embedding_model=None):
            self.table_name = safe_table_name(table_name)
            self.embedding_model = embedding_model or build_embedding_model(show_progress=False)
            self._embed_lock = Lock()

        def _embed_query(self, query: str) -> list[float]:
            with self._embed_lock:
                return self.embedding_model.embed_query(query)

        def similarity_search(self, query: str, k: int = 4) -> list[Document]:
            sql = f"""
                SELECT
                    doc_name,
                    company,
                    doc_period,
                    page_number,
                    chunk_index,
                    chunk_size,
                    chunk_overlap,
                    content,
                    metadata,
                    1 - (embedding <=> %(embedding)s::vector) AS score
                FROM {self.table_name}
                ORDER BY embedding <=> %(embedding)s::vector
                LIMIT %(k)s
            """
            params = {
                "embedding": pgvector_literal(self._embed_query(query)),
                "k": int(k),
            }
            with connect_pg() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()

            docs = []
            for row in rows:
                metadata = row.get("metadata") or {}
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except Exception:
                        metadata = {}
                metadata = dict(metadata)
                metadata.update(
                    {
                        "doc_name": row.get("doc_name"),
                        "company": row.get("company"),
                        "doc_period": row.get("doc_period"),
                        "page_number": row.get("page_number"),
                        "chunk_index": row.get("chunk_index"),
                        "chunk_size": row.get("chunk_size"),
                        "chunk_overlap": row.get("chunk_overlap"),
                        "score": float(row.get("score", 0.0)),
                    }
                )
                docs.append(Document(page_content=row.get("content", ""), metadata=metadata))
            return docs
    '''

    source += textwrap.dedent(pgvector_cloud_block)
    return source


def build_code_cell_12() -> str:
    return '''
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    df_task3 = pd.read_json(FILTERED_DATASET_PATH, lines=True)


    def repo_pdf_filename(doc_name: str) -> str:
        return f"{doc_name}.pdf"


    def raw_pdf_url(filename: str) -> str:
        return f"{PDF_RAW_BASE}/{urllib.parse.quote(filename)}"


    relevant_docs = (
        df_task3[["doc_name", "company", "doc_period", "doc_filename", "doc_link_raw"]]
        .drop_duplicates(subset=["doc_name"])
        .sort_values("doc_name", kind="stable")
        .reset_index(drop=True)
    )
    relevant_docs["repo_pdf_filename"] = relevant_docs["doc_name"].map(repo_pdf_filename)
    relevant_docs["pdf_url"] = relevant_docs["repo_pdf_filename"].map(raw_pdf_url)
    relevant_docs["local_pdf_path"] = relevant_docs["repo_pdf_filename"].map(lambda name: str(PDF_DIR / name))

    print(f"Filtered dataset rows: {len(df_task3):,}")
    print(f"Relevant PDFs: {len(relevant_docs):,}")
    display(relevant_docs[["doc_name", "company", "doc_period", "repo_pdf_filename"]].head(10))


    def download_pdf(row: pd.Series) -> Path:
        target_path = PDF_DIR / row["repo_pdf_filename"]
        if target_path.exists() and target_path.stat().st_size > 0:
            return target_path

        request = urllib.request.Request(
            row["pdf_url"],
            headers={"User-Agent": "assignment_2-financebench-rag/1.0"},
        )
        tmp_path = target_path.with_suffix(target_path.suffix + ".part")
        with urllib.request.urlopen(request, timeout=120) as response, tmp_path.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)
        tmp_path.replace(target_path)
        return target_path


    pdf_paths = []
    for _, row in tqdm(relevant_docs.iterrows(), total=len(relevant_docs), desc="Downloading PDFs"):
        pdf_paths.append(download_pdf(row))

    missing_or_empty = [path for path in pdf_paths if not path.exists() or path.stat().st_size == 0]
    if missing_or_empty:
        raise RuntimeError(f"Missing or empty PDFs: {missing_or_empty[:5]}")

    print(f"PDFs available in {PDF_DIR}: {len(pdf_paths):,}")

    embedding_model = build_embedding_model(show_progress=False)
    _embedding_lock = Lock()


    def embed_documents(texts: list[str]) -> list[list[float]]:
        with _embedding_lock:
            return embedding_model.embed_documents(texts)


    def table_manifest_path(table_name: str) -> Path:
        return DATA_DIR / "vectorstore" / safe_table_name(table_name) / "manifest.json"


    def load_table_manifest(table_name: str):
        manifest_path = table_manifest_path(table_name)
        if not manifest_path.exists():
            return None
        return json.loads(manifest_path.read_text())


    def ensure_pgvector_schema(table_name: str = BASE_TABLE) -> None:
        table = safe_table_name(table_name)
        with connect_pg() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id BIGSERIAL PRIMARY KEY,
                        chunk_uid TEXT UNIQUE NOT NULL,
                        doc_name TEXT NOT NULL,
                        company TEXT,
                        doc_period TEXT,
                        page_number INTEGER NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        chunk_size INTEGER NOT NULL,
                        chunk_overlap INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        embedding vector({EMBEDDING_DIM}) NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_doc_page_idx ON {table} (doc_name, page_number)")
            conn.commit()

        try:
            with connect_pg() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw_idx "
                        f"ON {table} USING hnsw (embedding vector_cosine_ops)"
                    )
                conn.commit()
        except Exception as exc:
            print(f"HNSW index failed ({exc}); falling back to IVFFlat.")
            with connect_pg() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"CREATE INDEX IF NOT EXISTS {table}_embedding_ivfflat_idx "
                        f"ON {table} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
                    )
                conn.commit()


    def count_chunks(table_name: str = BASE_TABLE) -> int:
        table = safe_table_name(table_name)
        with connect_pg() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) AS n FROM {table}")
                return int(cur.fetchone()["n"])


    def truncate_chunks(table_name: str = BASE_TABLE) -> None:
        table = safe_table_name(table_name)
        with connect_pg() as conn:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {table}")
            conn.commit()


    def pgvector_table_is_ready(table_name: str, chunk_size: int, chunk_overlap: int) -> bool:
        manifest = load_table_manifest(table_name)
        if not manifest:
            return False
        if manifest.get("embedding_model") != EMBEDDING_MODEL_NAME:
            return False
        if int(manifest.get("chunk_size", -1)) != int(chunk_size):
            return False
        if int(manifest.get("chunk_overlap", -1)) != int(chunk_overlap):
            return False
        try:
            chunk_count = count_chunks(table_name)
        except Exception:
            return False
        return chunk_count == int(manifest.get("chunk_count", -1)) and chunk_count > 0


    def load_pages_for_docs(doc_frame: pd.DataFrame, *, desc: str = "Loading PDF pages") -> list:
        pages = []
        load_errors = []
        for _, row in tqdm(doc_frame.iterrows(), total=len(doc_frame), desc=desc):
            pdf_path = PDF_DIR / row["repo_pdf_filename"]
            try:
                loaded_pages = PyPDFLoader(str(pdf_path)).load()
            except Exception as exc:
                load_errors.append({"doc_name": row["doc_name"], "pdf_path": str(pdf_path), "error": repr(exc)})
                continue

            for fallback_page_number, page in enumerate(loaded_pages):
                page_number = int(page.metadata.get("page", fallback_page_number))
                doc_period = row["doc_period"]
                if pd.notna(doc_period):
                    try:
                        doc_period = int(doc_period)
                    except Exception:
                        doc_period = str(doc_period)
                else:
                    doc_period = None
                page.metadata.update(
                    {
                        "doc_name": row["doc_name"],
                        "company": row["company"],
                        "doc_period": doc_period,
                        "page_number": page_number,
                    }
                )
                page.metadata.pop("page", None)
                pages.append(page)

        if load_errors:
            display(pd.DataFrame(load_errors))
            raise RuntimeError("At least one PDF failed to load; see load_errors above.")

        print(f"Loaded pages: {len(pages):,}")
        return pages


    def build_chunks_from_pages(pages: list, *, chunk_size: int, chunk_overlap: int) -> list:
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
        chunks,
        *,
        table_name: str = BASE_TABLE,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        batch_size: int = 128,
    ) -> None:
        table = safe_table_name(table_name)
        sql = f"""
            INSERT INTO {table} (
                chunk_uid, doc_name, company, doc_period, page_number, chunk_index,
                chunk_size, chunk_overlap, content, embedding, metadata
            ) VALUES (
                %(chunk_uid)s, %(doc_name)s, %(company)s, %(doc_period)s, %(page_number)s,
                %(chunk_index)s, %(chunk_size)s, %(chunk_overlap)s, %(content)s,
                %(embedding)s::vector, %(metadata)s
            )
            ON CONFLICT (chunk_uid) DO UPDATE SET
                doc_name = EXCLUDED.doc_name,
                company = EXCLUDED.company,
                doc_period = EXCLUDED.doc_period,
                page_number = EXCLUDED.page_number,
                chunk_index = EXCLUDED.chunk_index,
                chunk_size = EXCLUDED.chunk_size,
                chunk_overlap = EXCLUDED.chunk_overlap,
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata
        """

        with connect_pg() as conn:
            with conn.cursor() as cur:
                for start in tqdm(range(0, len(chunks), batch_size), desc=f"Embedding/inserting into {table}"):
                    batch = chunks[start:start + batch_size]
                    texts = [chunk.page_content.replace("\\x00", "") for chunk in batch]
                    embeddings = embed_documents(texts)
                    rows = []
                    for offset, (chunk, vector) in enumerate(zip(batch, embeddings), start=start):
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
                    cur.executemany(sql, rows)
                conn.commit()


    def build_or_load_pgvector_store(
        table_name: str,
        *,
        relevant_docs: pd.DataFrame,
        dataset_rows: int,
        chunk_size: int,
        chunk_overlap: int,
        rebuild: bool = False,
    ) -> PGVectorFinanceBenchStore:
        ensure_pgvector_schema(table_name)
        manifest_path = table_manifest_path(table_name)

        if pgvector_table_is_ready(table_name, chunk_size, chunk_overlap) and not rebuild:
            print(f"Loaded existing pgvector table {table_name}")
            print(json.dumps(load_table_manifest(table_name), indent=2))
            return PGVectorFinanceBenchStore(table_name, embedding_model=embedding_model)

        pages = load_pages_for_docs(relevant_docs)
        chunks = build_chunks_from_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if rebuild:
            print(f"REBUILD_VECTORSTORE=true; truncating {table_name}")
            truncate_chunks(table_name)

        insert_chunks(
            chunks,
            table_name=table_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=128,
        )

        manifest = {
            "backend": "pgvector",
            "table_name": table_name,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(chunk_overlap),
            "filtered_dataset_rows": int(dataset_rows),
            "pdf_count": int(len(relevant_docs)),
            "page_count": int(len(pages)),
            "chunk_count": int(len(chunks)),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Saved pgvector manifest to {manifest_path}")
        print(json.dumps(manifest, indent=2))
        return PGVectorFinanceBenchStore(table_name, embedding_model=embedding_model)


    vectorstore = build_or_load_pgvector_store(
        BASE_TABLE,
        relevant_docs=relevant_docs,
        dataset_rows=len(df_task3),
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        rebuild=REBUILD_VECTORSTORE,
    )
    print(f"Rows currently in {BASE_TABLE}: {count_chunks(BASE_TABLE):,}")
    '''


def build_code_cell_14() -> str:
    return '''
    def get_nebius_client() -> OpenAI:
        if load_dotenv is not None:
            load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)

        api_key = os.getenv("NEBIUS_API_KEY")
        if not api_key:
            api_key = getpass("Nebius API key: ")
            if api_key:
                os.environ["NEBIUS_API_KEY"] = api_key

        if not api_key:
            raise ValueError("Set NEBIUS_API_KEY in .env or enter a Nebius API key to run RAG generation.")
        ensure_api_ca_bundle()
        return OpenAI(base_url=NEBIUS_BASE_URL, api_key=api_key, timeout=60.0, max_retries=API_MAX_RETRIES)


    def load_rag_vectorstore() -> PGVectorFinanceBenchStore:
        ensure_pgvector_schema(BASE_TABLE)
        row_count = count_chunks(BASE_TABLE)
        if row_count == 0:
            raise FileNotFoundError(f"pgvector table {BASE_TABLE} is empty. Run Task 3 first.")
        embeddings = build_embedding_model(show_progress=False)
        return PGVectorFinanceBenchStore(BASE_TABLE, embedding_model=embeddings)


    rag_vectorstore = load_rag_vectorstore()
    print(f"Loaded pgvector vector store from table {BASE_TABLE}")


    def prompt_terms(text: str) -> set[str]:
        terms = set(re.findall(r"[A-Za-z0-9][A-Za-z0-9&._/-]*", str(text).lower()))
        return {term for term in terms if term not in STOPWORDS and len(term) > 1}


    def build_rag_user_prompt(query: str, context: str) -> str:
        return f"""Question:
    {query}

    Retrieved FinanceBench context:
    {context}

    Instructions:
    - Answer using only the retrieved context above.
    - Prefer chunks whose company, filing name, fiscal period, and date match the question.
    - Ignore mismatched-company or mismatched-period chunks unless the question asks for a comparison.
    - For numeric questions, use Answer / Unit / Period/date / Formula / Evidence fields.
    - If the answer is not present, say exactly: The context does not contain enough information.
    - Cite each factual sentence with [doc_name, page N]."""


    def build_rag_messages(system_prompt: str, user_prompt: str, few_shot_messages: "list[dict] | None" = None) -> list[dict]:
        messages = [{"role": "system", "content": system_prompt}]
        if few_shot_messages is None:
            few_shot_messages = RAG_FEW_SHOT_MESSAGES
        messages.extend(few_shot_messages)
        messages.append({"role": "user", "content": user_prompt})
        return messages


    def format_chunk_for_prompt(chunk, rank: int) -> str:
        metadata = chunk.metadata
        doc_name = metadata.get("doc_name", "unknown_doc")
        page_number = metadata.get("page_number", "unknown_page")
        company = metadata.get("company", "unknown_company")
        doc_period = metadata.get("doc_period", "unknown_period")
        text = " ".join(chunk.page_content.split())

        return (
            f"--- Chunk {rank} ---\\n"
            f"doc_name: {doc_name}\\n"
            f"company: {company}\\n"
            f"doc_period: {doc_period}\\n"
            f"page_number: {page_number}\\n"
            f"content:\\n{text}"
        )


    def format_retrieved_context(retrieved_docs: list) -> str:
        if not retrieved_docs:
            return "NO_RELEVANT_CONTEXT_RETRIEVED"
        return "\\n\\n".join(format_chunk_for_prompt(chunk, rank) for rank, chunk in enumerate(retrieved_docs, start=1))


    def serialize_retrieved_chunk(chunk, rank: int) -> dict:
        metadata = chunk.metadata
        return {
            "rank": rank,
            "doc_name": metadata.get("doc_name"),
            "company": metadata.get("company"),
            "doc_period": metadata.get("doc_period"),
            "page_number": metadata.get("page_number"),
            "content": chunk.page_content,
        }


    def answer_with_rag(query: str, k: int = 4) -> dict:
        if not str(query).strip():
            raise ValueError("query must be a non-empty string")

        k = max(0, int(k))
        retrieved_docs = rag_vectorstore.similarity_search(query, k=k) if k > 0 else []
        context = format_retrieved_context(retrieved_docs)
        user_prompt = build_rag_user_prompt(query, context)
        client = get_nebius_client()

        def _request():
            response = client.chat.completions.create(
                model=RAG_GENERATION_MODEL,
                messages=build_rag_messages(RAG_SYSTEM_PROMPT, user_prompt),
                temperature=0,
                max_tokens=700,
            )
            return response.choices[0].message.content.strip()

        answer = call_with_retries(_request, "RAG generation")

        return {
            "answer": answer,
            "retrieved_chunks": [
                serialize_retrieved_chunk(chunk, rank)
                for rank, chunk in enumerate(retrieved_docs, start=1)
            ],
        }
    '''


def build_code_cell_17(faiss_source: str) -> str:
    source = faiss_source
    source = replace_exact(
        source,
        "Task 6 baseline: query-only FAISS top-5, original prompt, Llama-3.3-70B-Instruct.",
        "Task 6 baseline: query-only pgvector top-5, original prompt, Llama-3.3-70B-Instruct.",
    )
    source = replace_exact(source, 'if mode == "faiss":', 'if mode == "vectorstore":')
    return source


def build_code_cell_18() -> str:
    return '''
    BONUS_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    bonus_df = pd.read_json(BONUS_DATASET_PATH, lines=True)
    bonus_df = bonus_df.sort_values("financebench_id", kind="stable").reset_index(drop=True)
    bonus_relevant_docs = (
        bonus_df[["doc_name", "company", "doc_period"]]
        .drop_duplicates(subset=["doc_name"])
        .sort_values("doc_name", kind="stable")
        .reset_index(drop=True)
    )
    bonus_relevant_docs["repo_pdf_filename"] = bonus_relevant_docs["doc_name"].map(lambda doc_name: f"{doc_name}.pdf")

    bonus_table_names = {
        500: f"{PGVECTOR_TABLE_PREFIX}_500",
        1000: BASE_TABLE,
        2000: f"{PGVECTOR_TABLE_PREFIX}_2000",
    }


    def bonus_as_list(value):
        if isinstance(value, list):
            return value
        if value is None:
            return []
        try:
            if pd.isna(value):
                return []
        except Exception:
            pass
        return [value]


    def bonus_expected_pages(row: pd.Series) -> set[int]:
        return {int(page) for page in bonus_as_list(row.get("evidence_page_nums")) if str(page).strip() != ""}


    def bonus_load_pages() -> list:
        return load_pages_for_docs(bonus_relevant_docs, desc="Bonus loading PDF pages")


    def bonus_build_or_load_index(chunk_size: int, pages: list) -> PGVectorFinanceBenchStore:
        table_name = bonus_table_names[chunk_size]
        ensure_pgvector_schema(table_name)
        if pgvector_table_is_ready(table_name, chunk_size, BONUS_CHUNK_OVERLAP) and not BONUS_REBUILD_INDICES:
            print(f"Loaded existing chunk_size={chunk_size} pgvector table {table_name}", flush=True)
            manifest = load_table_manifest(table_name)
            if manifest is not None:
                print(json.dumps(manifest, indent=2), flush=True)
            return PGVectorFinanceBenchStore(table_name, embedding_model=embedding_model)

        chunks = build_chunks_from_pages(pages, chunk_size=chunk_size, chunk_overlap=BONUS_CHUNK_OVERLAP)
        print(
            f"Building chunk_size={chunk_size} pgvector table from {len(pages):,} pages and {len(chunks):,} chunks",
            flush=True,
        )

        if BONUS_REBUILD_INDICES:
            truncate_chunks(table_name)

        insert_chunks(
            chunks,
            table_name=table_name,
            chunk_size=chunk_size,
            chunk_overlap=BONUS_CHUNK_OVERLAP,
        )

        manifest = {
            "backend": "pgvector",
            "table_name": table_name,
            "embedding_model": BONUS_EMBEDDING_MODEL_NAME,
            "chunk_size": int(chunk_size),
            "chunk_overlap": int(BONUS_CHUNK_OVERLAP),
            "filtered_dataset_rows": int(len(bonus_df)),
            "pdf_count": int(len(bonus_relevant_docs)),
            "page_count": int(len(pages)),
            "chunk_count": int(len(chunks)),
        }
        manifest_path = table_manifest_path(table_name)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Saved chunk_size={chunk_size} manifest to {manifest_path}", flush=True)
        print(json.dumps(manifest, indent=2), flush=True)
        return PGVectorFinanceBenchStore(table_name, embedding_model=embedding_model)


    def bonus_page_hit_from_docs(docs: list, expected_doc: str, expected_pages: set[int]) -> int:
        if not expected_pages:
            return 0
        for doc in docs[:5]:
            metadata = doc.metadata
            if metadata.get("doc_name") == expected_doc and metadata.get("page_number") in expected_pages:
                return 1
        return 0


    def bonus_save_json(path: Path, rows: list[dict]) -> None:
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))


    bonus_pages = bonus_load_pages()
    bonus_vectorstores = {}
    for chunk_size in BONUS_CHUNK_SIZES:
        bonus_vectorstores[chunk_size] = bonus_build_or_load_index(chunk_size, bonus_pages)

    try:
        embedding_model.show_progress = False
    except Exception:
        pass

    per_size_rows = {}
    for chunk_size in BONUS_CHUNK_SIZES:
        artifact_path = BONUS_ARTIFACT_DIR / f"page_hit_at5_chunk{chunk_size}.json"
        rows = []
        vectorstore = bonus_vectorstores[chunk_size]

        for _, row in tqdm(bonus_df.iterrows(), total=len(bonus_df), desc=f"Bonus page-hit@5 chunk_size={chunk_size}"):
            financebench_id = row["financebench_id"]
            docs = vectorstore.similarity_search(row["question"], k=5)
            retrieved = [
                {
                    "rank": rank,
                    "doc_name": doc.metadata.get("doc_name"),
                    "page_number": doc.metadata.get("page_number"),
                    "company": doc.metadata.get("company"),
                    "doc_period": doc.metadata.get("doc_period"),
                }
                for rank, doc in enumerate(docs, start=1)
            ]
            expected_pages = bonus_expected_pages(row)
            raw_row = {
                "financebench_id": financebench_id,
                "doc_name": row["doc_name"],
                "question": row["question"],
                "evidence_page_nums": sorted(expected_pages),
                "chunk_size": int(chunk_size),
                "page_hit_at_5": bonus_page_hit_from_docs(docs, row["doc_name"], expected_pages),
                "retrieved": retrieved,
            }
            rows.append(raw_row)
            bonus_save_json(artifact_path, rows)

        bonus_save_json(artifact_path, rows)
        per_size_rows[chunk_size] = rows

    bonus_comparison_rows = []
    for _, row in bonus_df.iterrows():
        financebench_id = row["financebench_id"]
        comparison = {
            "financebench_id": financebench_id,
            "question_type": row.get("question_type"),
            "doc_name": row["doc_name"],
            "question": row["question"],
            "evidence_page_nums": row.get("evidence_page_nums"),
        }
        hits = []
        for chunk_size in BONUS_CHUNK_SIZES:
            hit = next(item for item in per_size_rows[chunk_size] if item["financebench_id"] == financebench_id)["page_hit_at_5"]
            comparison[f"page_hit_at_5_chunk{chunk_size}"] = hit
            hits.append(hit)
        comparison["any_chunk_size_hit"] = int(any(hits))
        comparison["all_chunk_sizes_agree"] = int(len(set(hits)) == 1)
        comparison["chunk_size_disagreement"] = int(len(set(hits)) > 1)
        winning_sizes = [size for size, hit in zip(BONUS_CHUNK_SIZES, hits) if hit == max(hits)]
        comparison["best_chunk_sizes"] = ",".join(map(str, winning_sizes))
        comparison["unique_winning_chunk_size"] = winning_sizes[0] if len(winning_sizes) == 1 else pd.NA
        bonus_comparison_rows.append(comparison)

    bonus_comparison_df = pd.DataFrame(bonus_comparison_rows)

    summary_rows = []
    for chunk_size in BONUS_CHUNK_SIZES:
        col = f"page_hit_at_5_chunk{chunk_size}"
        summary_rows.append(
            {
                "chunk_size": chunk_size,
                "page_hit_at_5": bonus_comparison_df[col].mean(),
                "hit_count": int(bonus_comparison_df[col].sum()),
                "unique_winner_count": int((bonus_comparison_df["unique_winning_chunk_size"] == chunk_size).sum()),
            }
        )

    summary_rows.append(
        {
            "chunk_size": "oracle_any_size",
            "page_hit_at_5": bonus_comparison_df["any_chunk_size_hit"].mean(),
            "hit_count": int(bonus_comparison_df["any_chunk_size_hit"].sum()),
            "unique_winner_count": int(bonus_comparison_df["chunk_size_disagreement"].sum()),
        }
    )
    bonus_summary_df = pd.DataFrame(summary_rows)

    bonus_disagreement_count = int(bonus_comparison_df["chunk_size_disagreement"].sum())
    bonus_disagreement_rate = bonus_disagreement_count / len(bonus_comparison_df)
    bonus_unique_winner_count = int(bonus_comparison_df["unique_winning_chunk_size"].notna().sum())
    bonus_unique_winner_rate = bonus_unique_winner_count / len(bonus_comparison_df)

    with pd.ExcelWriter(BONUS_XLSX_PATH) as writer:
        bonus_summary_df.to_excel(writer, sheet_name="summary", index=False)
        bonus_comparison_df.to_excel(writer, sheet_name="per_question", index=False)

    print(f"Saved bonus comparison to {BONUS_XLSX_PATH}")
    print(f"Disagreement count: {bonus_disagreement_count}/{len(bonus_comparison_df)} ({bonus_disagreement_rate:.1%})")
    print(f"Unique winner count: {bonus_unique_winner_count}/{len(bonus_comparison_df)} ({bonus_unique_winner_rate:.1%})")
    display(bonus_summary_df)
    display(bonus_comparison_df)
    '''


def build_notebook() -> dict:
    faiss_nb = read_notebook(FAISS_NOTEBOOK)
    pg_nb = read_notebook(PGVECTOR_NOTEBOOK)

    faiss_code_cells = [cell_text(cell) for cell in faiss_nb["cells"] if cell["cell_type"] == "code"]
    if len(faiss_code_cells) != 18:
        raise ValueError(f"Expected 18 FAISS code cells, found {len(faiss_code_cells)}")

    markdown_sources = build_markdown_cells()
    if len(markdown_sources) != 16:
        raise ValueError("Expected 16 markdown cells")

    replacement_code_cells = {
        1: build_code_cell_2(),
        2: build_code_cell_3(),
        3: build_code_cell_4(faiss_code_cells[3]),
        11: build_code_cell_12(),
        13: build_code_cell_14(),
        16: build_code_cell_17(faiss_code_cells[16]),
        17: build_code_cell_18(),
    }

    code_sources = []
    for idx, source in enumerate(faiss_code_cells):
        code_sources.append(replacement_code_cells.get(idx, source))

    code_iter = iter(code_sources)
    markdown_iter = iter(markdown_sources)
    new_cells = []
    for cell in faiss_nb["cells"]:
        if cell["cell_type"] == "markdown":
            new_cells.append(new_markdown_cell(next(markdown_iter)))
        else:
            new_cells.append(new_code_cell(next(code_iter)))

    try:
        next(code_iter)
        raise ValueError("Unused code cells after reconstruction")
    except StopIteration:
        pass
    try:
        next(markdown_iter)
        raise ValueError("Unused markdown cells after reconstruction")
    except StopIteration:
        pass

    return {
        "cells": new_cells,
        "metadata": pg_nb.get("metadata", {}),
        "nbformat": pg_nb.get("nbformat", 4),
        "nbformat_minor": max(int(pg_nb.get("nbformat_minor", 5)), 5),
    }


def main() -> None:
    notebook = build_notebook()
    PGVECTOR_NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Rebuilt {PGVECTOR_NOTEBOOK}")


if __name__ == "__main__":
    main()
