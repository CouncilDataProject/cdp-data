#!/usr/bin/env python

import argparse
import json
import logging
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, Union

import matplotlib as mpl
import pandas as pd
import seaborn as sns

from cdp_data import CDPInstances, datasets, keywords, plots
from cdp_data.keywords import _stem_n_gram

sns.set_theme(color_codes=True)
mpl.rcParams["axes.labelsize"] = 16

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
ALL_INSTANCES_DISCUSSION_TRENDS_PLOT = "councils-in-action-discussion-trends"
ALL_INSTANCES_SPLIT_DISCUSSION_TRENDS_PLOT = (
    "councils-in-action-split-discussion-trends"
)

PLOT_NGRAMS = [
    "police",
    "housing",
    "union",
    "homelessness",
]

PLOT_INFRASTRUCTURES = [
    CDPInstances.Seattle,
    CDPInstances.KingCounty,
    CDPInstances.Portland,
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


def generate_paper_content(  # noqa: C901
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
        if infra == CDPInstances.Seattle:
            start_datetime = "2021-01-01"
        else:
            start_datetime = "2021-10-01"

        infra_ds = datasets.get_session_dataset(
            infrastructure_slug=infra,
            start_datetime=start_datetime,
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
    seattle_discussion_trends_pdf = (
        storage_directory / f"{SEATTLE_DISCUSSION_TRENDS_PLOT}.pdf"
    )
    seattle_discussion_trends_png = seattle_discussion_trends_pdf.with_suffix(".png")
    all_instances_discussion_trends_pdf = (
        storage_directory / f"{ALL_INSTANCES_DISCUSSION_TRENDS_PLOT}.pdf"
    )
    all_instances_discussion_trends_png = (
        all_instances_discussion_trends_pdf.with_suffix(".png")
    )
    all_instances_split_discussion_trends_pdf = (
        storage_directory / f"{ALL_INSTANCES_SPLIT_DISCUSSION_TRENDS_PLOT}.pdf"
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
        seattle_ngram_usage.to_parquet(
            seattle_discussion_trends_pdf.with_suffix(".parquet"),
            index=False,
        )

        # Compute and store selected stats
        collected_seattle_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Month subsets
        jan_2021 = seattle_ngram_usage.loc[
            (seattle_ngram_usage.session_datetime >= "2021-01-01")
            & (seattle_ngram_usage.session_datetime < "2021-02-01")
        ]
        march_2022 = seattle_ngram_usage.loc[
            (seattle_ngram_usage.session_datetime >= "2022-03-01")
            & (seattle_ngram_usage.session_datetime < "2022-04-01")
        ]

        # Calc per gram
        for gram in PLOT_NGRAMS:
            stemmed_gram = _stem_n_gram(gram)
            collected_seattle_stats[stemmed_gram] = {
                "january_2021": {},
                "march_2022": {},
            }
            collected_seattle_stats[stemmed_gram]["january_2021"][
                "mean"
            ] = jan_2021.loc[
                jan_2021.ngram == stemmed_gram
            ].day_ngram_percent_usage.mean()
            collected_seattle_stats[stemmed_gram]["january_2021"]["std"] = jan_2021.loc[
                jan_2021.ngram == stemmed_gram
            ].day_ngram_percent_usage.std()
            collected_seattle_stats[stemmed_gram]["march_2022"][
                "mean"
            ] = march_2022.loc[
                march_2022.ngram == stemmed_gram
            ].day_ngram_percent_usage.mean()
            collected_seattle_stats[stemmed_gram]["march_2022"]["std"] = march_2022.loc[
                march_2022.ngram == stemmed_gram
            ].day_ngram_percent_usage.std()

        # Store stats
        with open(seattle_discussion_trends_pdf.with_suffix(".json"), "w") as open_f:
            json.dump(collected_seattle_stats, open_f, indent=4)

        # Plot
        seattle_discussion_trends_grid = plots.plot_ngram_usage_histories(
            ngram=PLOT_NGRAMS,
            gram_usage=seattle_ngram_usage,
            strict=False,  # stem provided grams
            lmplot_kws={  # extra plotting params
                "col": "ngram",
                "hue": "ngram",
                "col_wrap": 2,
                "scatter_kws": {"alpha": 0.2},
                "aspect": 1.6,
            },
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
            infrastructure_slug=PLOT_INFRASTRUCTURES,
            ngram_size=1,  # generate unigrams
            strict=False,  # stem grams
            start_datetime="2021-10-01",  # data available for all instances
            end_datetime="2022-04-01",
            cache_dir=full_dataset_storage_dir,
        )
        all_instances_ngram_usage.to_parquet(
            all_instances_discussion_trends_pdf.with_suffix(".parquet"),
            index=False,
        )

        # Compute and store selected stats
        collected_all_instance_stats: Dict[str, Dict[str, Dict[str, float]]] = {}

        # Month subsets
        oct_2021 = all_instances_ngram_usage.loc[
            (all_instances_ngram_usage.session_datetime >= "2021-10-01")
            & (all_instances_ngram_usage.session_datetime < "2021-11-01")
        ]
        march_2022 = all_instances_ngram_usage.loc[
            (all_instances_ngram_usage.session_datetime >= "2022-03-01")
            & (all_instances_ngram_usage.session_datetime < "2022-04-01")
        ]

        # Calc per gram
        for gram in PLOT_NGRAMS:
            stemmed_gram = _stem_n_gram(gram)
            collected_all_instance_stats[stemmed_gram] = {
                "october_2021": {},
                "march_2022": {},
            }
            collected_all_instance_stats[stemmed_gram]["october_2021"][
                "mean"
            ] = oct_2021.loc[
                oct_2021.ngram == stemmed_gram
            ].day_ngram_percent_usage.mean()
            collected_all_instance_stats[stemmed_gram]["october_2021"][
                "std"
            ] = oct_2021.loc[
                oct_2021.ngram == stemmed_gram
            ].day_ngram_percent_usage.std()
            collected_all_instance_stats[stemmed_gram]["march_2022"][
                "mean"
            ] = march_2022.loc[
                march_2022.ngram == stemmed_gram
            ].day_ngram_percent_usage.mean()
            collected_all_instance_stats[stemmed_gram]["march_2022"][
                "std"
            ] = march_2022.loc[
                march_2022.ngram == stemmed_gram
            ].day_ngram_percent_usage.std()

        # Store stats
        with open(
            all_instances_discussion_trends_pdf.with_suffix(".json"), "w"
        ) as open_f:
            json.dump(collected_all_instance_stats, open_f, indent=4)

        # Compute and store selected stats for splits
        collected_all_instance_split_stats: Dict[
            str, Dict[str, Dict[str, Dict[str, float]]]
        ] = {}

        # Calc per gram and infrastructure
        for infra in PLOT_INFRASTRUCTURES:
            oct_2021_infra_subset = oct_2021.loc[oct_2021.infrastructure == infra]
            march_2022_infra_subset = march_2022.loc[march_2022.infrastructure == infra]
            collected_all_instance_split_stats[infra] = {}
            for gram in PLOT_NGRAMS:
                stemmed_gram = _stem_n_gram(gram)
                collected_all_instance_split_stats[infra][stemmed_gram] = {
                    "october_2021": {},
                    "march_2022": {},
                }
                collected_all_instance_split_stats[infra][stemmed_gram]["october_2021"][
                    "mean"
                ] = oct_2021_infra_subset.loc[
                    oct_2021_infra_subset.ngram == stemmed_gram
                ].day_ngram_percent_usage.mean()
                collected_all_instance_split_stats[infra][stemmed_gram]["october_2021"][
                    "std"
                ] = oct_2021_infra_subset.loc[
                    oct_2021_infra_subset.ngram == stemmed_gram
                ].day_ngram_percent_usage.std()
                collected_all_instance_split_stats[infra][stemmed_gram]["march_2022"][
                    "mean"
                ] = march_2022_infra_subset.loc[
                    march_2022_infra_subset.ngram == stemmed_gram
                ].day_ngram_percent_usage.mean()
                collected_all_instance_split_stats[infra][stemmed_gram]["march_2022"][
                    "std"
                ] = march_2022_infra_subset.loc[
                    march_2022_infra_subset.ngram == stemmed_gram
                ].day_ngram_percent_usage.std()

        # Store stats
        with open(
            all_instances_split_discussion_trends_pdf.with_suffix(".json"), "w"
        ) as open_f:
            json.dump(collected_all_instance_split_stats, open_f, indent=4)

        # Show all instances discussion trends
        all_instances_discussion_trends_grid = plots.plot_ngram_usage_histories(
            ngram=PLOT_NGRAMS,
            gram_usage=all_instances_ngram_usage,
            strict=False,  # stem provided grams
            lmplot_kws={  # extra plotting params
                "col": "ngram",
                "hue": "ngram",
                "col_wrap": 2,
                "scatter_kws": {"alpha": 0.2},
                "aspect": 1.6,
            },
        )
        all_instances_discussion_trends_grid.savefig(
            all_instances_discussion_trends_pdf,
        )
        all_instances_discussion_trends_grid.savefig(
            all_instances_discussion_trends_png,
        )

        # Show all instances discussion trends split by infra
        all_instances_split_discussion_trends_grid = plots.plot_ngram_usage_histories(
            ngram=PLOT_NGRAMS,
            gram_usage=all_instances_ngram_usage,
            strict=False,  # stem provided grams
            lmplot_kws={  # extra plotting params
                "col": "infrastructure",
                "hue": "ngram",
                "row": "ngram",
                "scatter_kws": {"alpha": 0.2},
                "aspect": 1.6,
            },
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
