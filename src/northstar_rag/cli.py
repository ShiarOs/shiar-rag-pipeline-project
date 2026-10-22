"""Command-line interface for indexing, asking, calibration, and evaluation."""

import json
from pathlib import Path

import click

from .embeddings import SentenceTransformerEmbedder
from .evaluation import evaluate, load_cases, write_report
from .generation import ExtractiveGenerator, GroqGenerator
from .hybrid_search import HybridSearcher
from .pipeline import RAGPipeline
from .settings import Settings, get_settings


def build_pipeline(
    settings: Settings,
    rebuild: bool = False,
    live: bool = False,
    min_score: float | None = None,
    top_k: int | None = None,
) -> RAGPipeline:
    overrides = {}
    if min_score is not None:
        overrides["min_score"] = min_score
    if top_k is not None:
        overrides["top_k"] = top_k
        overrides["candidate_k"] = max(settings.candidate_k, top_k)
    if overrides:
        settings = settings.model_copy(update=overrides)
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        settings.embedding_revision,
    )
    searcher = HybridSearcher.build_or_load(settings, embedder, force_rebuild=rebuild)
    if live:
        if not settings.groq_api_key:
            raise click.ClickException(
                "GROQ_API_KEY is empty; configure it locally in .env"
            )
        generator = GroqGenerator(settings.groq_api_key, settings.groq_model)
    else:
        generator = ExtractiveGenerator()
    return RAGPipeline(settings, embedder, searcher, generator)


@click.group()
def main() -> None:
    """Run the local-first Northstar hybrid RAG application."""


@main.command()
@click.option(
    "--rebuild", is_flag=True, help="Rebuild the index from source documents."
)
def index(rebuild: bool) -> None:
    """Build or load the vector and lexical indexes."""

    settings = get_settings()
    pipeline = build_pipeline(settings, rebuild=rebuild)
    try:
        counts: dict[str, int] = {}
        for chunk in pipeline.searcher.chunks:
            counts[chunk.document_id] = counts.get(chunk.document_id, 0) + 1
        click.echo(f"Indexed {len(pipeline.searcher.chunks)} chunks")
        click.echo(
            f"Embedding: {settings.embedding_model} @ {settings.embedding_revision}"
        )
        for source_id, count in sorted(counts.items()):
            click.echo(f"  {source_id}: {count}")
        click.echo("Sample chunk:")
        click.echo(
            json.dumps(
                pipeline.searcher.chunks[0].__dict__, indent=2, ensure_ascii=False
            )
        )
    finally:
        pipeline.close()


@main.command()
@click.argument("question")
@click.option("--live", is_flag=True, help="Use the optional Groq generation provider.")
@click.option("--min-score", type=click.FloatRange(0.0, 1.0))
@click.option("--top-k", type=click.IntRange(1, 50))
def ask(question: str, live: bool, min_score: float | None, top_k: int | None) -> None:
    """Answer one question and show status, evidence, and citations."""

    settings = get_settings()
    pipeline = build_pipeline(settings, live=live, min_score=min_score, top_k=top_k)
    try:
        result = pipeline.answer(question)
        click.echo(json.dumps(result.__dict__, indent=2, ensure_ascii=False))
    finally:
        pipeline.close()


@main.command()
@click.option("--cases", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50")
def calibrate(cases: Path | None, thresholds: str) -> None:
    """Compare false accepts and false rejections on calibration cases only."""

    settings = get_settings()
    cases_path = cases or settings.calibration_file
    evaluation_cases = load_cases(cases_path)
    try:
        candidates = [float(value.strip()) for value in thresholds.split(",")]
    except ValueError as error:
        raise click.ClickException(
            "Thresholds must be comma-separated numbers"
        ) from error
    if not candidates or any(not 0 <= value <= 1 for value in candidates):
        raise click.ClickException("Every threshold must be between 0 and 1")

    pipeline = build_pipeline(settings)
    try:
        scores = []
        for case in evaluation_cases:
            hits, _ = pipeline.retrieve(case.question)
            scores.append((case, max((hit.dense_score for hit in hits), default=0.0)))
        click.echo("threshold | false_rejections | false_acceptances")
        for threshold in candidates:
            false_rejections = sum(
                not case.should_abstain and score < threshold for case, score in scores
            )
            false_acceptances = sum(
                case.should_abstain and score >= threshold for case, score in scores
            )
            click.echo(
                f"{threshold:9.2f} | {false_rejections:16d} | {false_acceptances:16d}"
            )
        click.echo("Choose and document a trade-off; do not tune on questions.json.")
    finally:
        pipeline.close()


@main.command(name="evaluate")
@click.option("--cases", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", type=click.Path(path_type=Path))
@click.option(
    "--live", is_flag=True, help="Use Groq; generated paraphrases require review."
)
@click.option("--min-score", type=click.FloatRange(0.0, 1.0))
@click.option("--top-k", type=click.IntRange(1, 50))
def evaluate_command(
    cases: Path | None,
    output: Path | None,
    live: bool,
    min_score: float | None,
    top_k: int | None,
) -> None:
    """Run a labeled question set and write Markdown plus JSON evidence."""

    settings = get_settings()
    cases_path = cases or settings.evaluation_file
    output_path = output or (
        Path("storage") / "calibration_report.md"
        if cases_path.name == "calibration.json"
        else Path("reports") / "evaluation_report.md"
    )
    pipeline = build_pipeline(settings, live=live, min_score=min_score, top_k=top_k)
    settings = pipeline.settings
    try:
        summary = evaluate(pipeline, load_cases(cases_path))
        calibration_summary = None
        if (
            cases_path.name != "calibration.json"
            and settings.calibration_file.is_file()
        ):
            calibration_cases = load_cases(settings.calibration_file)
            calibration_rows = []
            for case in calibration_cases:
                hits, _ = pipeline.retrieve(case.question)
                score = max((hit.dense_score for hit in hits), default=0.0)
                calibration_rows.append(
                    {
                        "id": case.case_id,
                        "should_abstain": case.should_abstain,
                        "top_score": score,
                        "would_abstain": score < settings.min_score or score <= 0,
                    }
                )
            calibration_summary = {
                "threshold": settings.min_score,
                "rows": calibration_rows,
                "false_rejections": sum(
                    not row["should_abstain"] and row["would_abstain"]
                    for row in calibration_rows
                ),
                "false_acceptances": sum(
                    row["should_abstain"] and not row["would_abstain"]
                    for row in calibration_rows
                ),
            }
        write_report(summary, output_path, cases_path.name, calibration_summary)
        click.echo(f"Wrote {output_path}")
        click.echo(json.dumps(summary["metrics"], indent=2))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
