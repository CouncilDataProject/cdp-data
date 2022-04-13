#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import shutil
import sys
import traceback
from pathlib import Path
from typing import Union

import pandas as pd

from cdp_data import CDPInstances, datasets, keywords

###############################################################################

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s: %(module)s:%(lineno)4s %(asctime)s] %(message)s",
)
log = logging.getLogger(__name__)

###############################################################################

DATASET_ARCHIVE = "dataset.zip"
DATASET_CONTENT_SUMMARY = "cdp-councils-in-action-content-summary"
SEATTLE_DISCUSSION_TRENDS_PLOT = "seattle-discussion-trends"
ALL_INSTANCES_DISCUSSION_TRENDS_PLOT = "councils-in-action-discussion_trends"
ALL_INSTANCES_SPLIT_DISCUSSION_TRENDS_PLOT = (
    "councils-in-action-split-discussion-trends"
)

PLOT_NGRAMS = [
    "police",
    "housing",
    "union",
    "homelessness",
]

###############################################################################


class Args(argparse.Namespace):
    def __init__(self) -> None:
        self.__parse()

    def __parse(self) -> None:
        p = argparse.ArgumentParser(
            prog="generate_cdp_councils_in_action_2022_paper_content",
            description=(
                "Generate all paper content (tables, figures, stats, etc.) as well "
                "archiving data used for such content generation. Note: This script "
                "requires installing cdp-data with the `[plot]` extras."
            ),
        )
        p.add_argument(
            "-d",
            "--storage-directory",
            dest="storage_directory",
            default=Path("cdp-councils-in-action-2022/"),
            help="Optional path to content and dataset storage directory.",
        )
        p.add_argument(
            "--clean-archived-data",
            dest="clean_archived_data",
            action="store_true",
            help=(
                "Should any cached and zipped data be removed before generating "
                "new paper content? Note: this will download and reprocess all data."
            ),
        )
        p.add_argument(
            "--clean-content",
            dest="clean_content",
            action="store_true",
            help=(
                "Should any paper content (tables, figures, stats, etc.) be cleaned "
                "and regenerated? Note: this will add minor processing time."
            ),
        )
        p.parse_args(namespace=self)


###############################################################################


def generate_paper_content(
    storage_directory: Union[str, Path],
    clean_archived_data: bool,
    clean_content: bool,
) -> Path:
    # Resolve storage dir
    storage_directory = Path(storage_directory).resolve()

    # Handle clean archived data
    if clean_archived_data and storage_directory.is_dir():
        shutil.rmtree(storage_directory)

    # Make / remake storage dir
    storage_directory.mkdir(parents=True, exist_ok=True)

    # Raw dataset storage
    full_dataset_storage_dir = storage_directory / "dataset"
    full_dataset_storage_dir.mkdir(parents=True, exist_ok=True)

    # Download raw dataset
    log.info("Downloading datasets from each instance")
    full_datasets = []
    full_datasets_stats = []
    for infra in [
        CDPInstances.Seattle,
        CDPInstances.KingCounty,
        CDPInstances.Portland,
    ]:
        infra_ds = datasets.get_session_dataset(
            infrastructure_slug=infra,
            start_datetime="2021-01-01",
            end_datetime="2022-04-01",
            store_full_metadata=True,
            store_transcript=True,
            transcript_selection="confidence",
            store_video=False,
            store_audio=False,
            cache_dir=full_dataset_storage_dir,
        )
        # Add minor info to make navigation with dataset easier
        infra_ds["event_id"] = infra_ds["event"].apply(lambda e: e.id)
        infra_ds["transcript_path"] = infra_ds["transcript_path"].apply(
            lambda p: str(p.relative_to(storage_directory))
        )
        # Drop non-serializable columns
        infra_ds = infra_ds.drop(["event", "transcript"], axis=1)
        full_datasets.append(infra_ds)

        # Compute stats
        full_datasets_stats.append(
            {
                "Instance": infra,
                "Events": infra_ds["event_id"].nunique(),
                "First Event": infra_ds["session_datetime"].min().date(),
                "Last Event": infra_ds["session_datetime"].max().date(),
            }
        )

    # Store parquet file
    pd.concat(full_datasets, ignore_index=True).to_parquet(
        storage_directory / "councils-in-actions-2022.parquet",
    )

    # Check or generate content
    dataset_archive = storage_directory / DATASET_ARCHIVE
    dataset_content_table_csv = storage_directory / f"{DATASET_CONTENT_SUMMARY}.csv"
    dataset_content_table_latex = dataset_content_table_csv.with_suffix(".tex")
    seattle_discussion_trends_pdf = storage_directory / SEATTLE_DISCUSSION_TRENDS_PLOT
    seattle_discussion_trends_png = seattle_discussion_trends_pdf.with_suffix(".png")
    all_instances_discussion_trends_pdf = (
        storage_directory / ALL_INSTANCES_DISCUSSION_TRENDS_PLOT
    )
    all_instances_discussion_trends_png = (
        all_instances_discussion_trends_pdf.with_suffix(".png")
    )
    all_instances_split_discussion_trends_pdf = (
        storage_directory / ALL_INSTANCES_SPLIT_DISCUSSION_TRENDS_PLOT
    )
    all_instances_split_discussion_trends_png = (
        all_instances_split_discussion_trends_pdf.with_suffix(".png")
    )

    # Check content
    if clean_content:
        for content_piece in [
            dataset_content_table_csv,
            dataset_content_table_latex,
            seattle_discussion_trends_pdf,
            seattle_discussion_trends_png,
            all_instances_discussion_trends_pdf,
            all_instances_discussion_trends_png,
            all_instances_split_discussion_trends_pdf,
            all_instances_split_discussion_trends_png,
        ]:
            # Delete dataset content tables
            if content_piece.exists():
                content_piece.unlink()

    # Generate new archive if needed
    if not dataset_archive.exists():
        log.info("Generating dataset archive")
        shutil.make_archive(
            str(dataset_archive.with_suffix("")),
            "zip",
            full_dataset_storage_dir,
        )

    # Generate content stats
    if (
        not dataset_content_table_csv.exists()
        or not dataset_content_table_latex.exists()
    ):
        log.info("Generating dataset content stats")
        dataset_content = pd.DataFrame(full_datasets_stats)
        dataset_content.to_csv(dataset_content_table_csv, index=False)
        dataset_content.to_latex(
            dataset_content_table_latex,
            index=False,
            caption="CDP Councils in Action Dataset Composition",
            label="dataset-content",
        )

    # Generate Seattle only discussion trends
    if (
        not seattle_discussion_trends_pdf.exists()
        or not seattle_discussion_trends_png.exists()
    ):
        log.info("Generate Seattle discussion trends plot")
        seattle_ngram_usage = keywords.compute_ngram_usage_history(
            infrastructure_slug=CDPInstances.Seattle,
            ngram_size=1,  # generate unigrams
            strict=False,  # stem grams
            start_datetime="2021-01-01",
            end_datetime="2022-04-01",
            cache_dir=full_dataset_storage_dir,
        )
        seattle_discussion_trends_grid = keywords.plot_ngram_usage_histories(
            ngram=PLOT_NGRAMS,
            gram_usage=seattle_ngram_usage,
            strict=False,  # stem provided grams
            lmplot_kws=dict(  # extra plotting params
                col="ngram",
                hue="ngram",
                scatter_kws={"alpha": 0.2},
                aspect=1.6,
            ),
        )
        seattle_discussion_trends_grid.savefig(seattle_discussion_trends_pdf)
        seattle_discussion_trends_grid.savefig(seattle_discussion_trends_png)

    # Generate all instances discussion trends
    if (
        not all_instances_discussion_trends_pdf.exists()
        or not all_instances_discussion_trends_png.exists()
        or not all_instances_split_discussion_trends_pdf.exists()
        or not all_instances_split_discussion_trends_png.exists()
    ):
        log.info("Generating all instances discussion trends (unified and split) plots")
        all_instances_ngram_usage = keywords.compute_ngram_usage_history(
            infrastructure_slug=[
                CDPInstances.Seattle,
                CDPInstances.KingCounty,
                CDPInstances.Portland,
            ],
            ngram_size=1,  # generate unigrams
            strict=False,  # stem grams
            start_datetime="2021-10-01",  # data available for all instances
            end_datetime="2022-04-01",
            cache_dir=full_dataset_storage_dir,
        )

        # Show all instances discussion trends
        all_instances_discussion_trends_grid = keywords.plot_ngram_usage_histories(
            ngram=PLOT_NGRAMS,
            gram_usage=all_instances_ngram_usage,
            strict=False,  # stem provided grams
            lmplot_kws=dict(  # extra plotting params
                col="ngram",
                hue="ngram",
                scatter_kws={"alpha": 0.2},
                aspect=1.6,
            ),
        )
        all_instances_discussion_trends_grid.savefig(
            all_instances_discussion_trends_pdf,
        )
        all_instances_discussion_trends_grid.savefig(
            all_instances_discussion_trends_png,
        )

        # Show all instances discussion trends split by infra
        all_instances_split_discussion_trends_grid = (
            keywords.plot_ngram_usage_histories(
                ngram=PLOT_NGRAMS,
                gram_usage=all_instances_ngram_usage,
                strict=False,  # stem provided grams
                lmplot_kws=dict(  # extra plotting params
                    col="infrastructure",
                    hue="ngram",
                    row="ngram",
                    scatter_kws={"alpha": 0.2},
                    aspect=1.6,
                ),
            )
        )
        all_instances_split_discussion_trends_grid.savefig(
            all_instances_split_discussion_trends_pdf,
        )
        all_instances_split_discussion_trends_grid.savefig(
            all_instances_split_discussion_trends_png,
        )

    return storage_directory


def main() -> None:
    try:
        args = Args()
        generate_paper_content(
            storage_directory=args.storage_directory,
            clean_archived_data=args.clean_archived_data,
            clean_content=args.clean_content,
        )
    except Exception as e:
        log.error("=============================================")
        log.error("\n\n" + traceback.format_exc())
        log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)
