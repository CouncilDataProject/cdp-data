#!/usr/bin/env python

from cdp_data import CDPInstances, keywords, plots

###############################################################################


def test_ngrams() -> None:
    ngram_usage = keywords.compute_ngram_usage_history(
        CDPInstances.Seattle,
        start_datetime="2021-01-01",
        end_datetime="2021-01-10",
    )
    grid = plots.plot_ngram_usage_histories(
        ["police", "housing"],
        ngram_usage,
    )
    grid.savefig("test-generated-ngram-usage-histories-plot.pdf")

    # Or with extra kwargs like order
    grid_third_order = plots.plot_ngram_usage_histories(
        ["police", "housing", "transportation"],
        ngram_usage,
        lmplot_kws={"order": 3},
    )
    grid_third_order.savefig("test-generated-ngram-usage-histories-plot.pdf")


def test_semantic_sim() -> None:
    semantic_hist = keywords.compute_query_semantic_similarity_history(
        [
            "defund the police",
            "missing middle housing",
        ],
        CDPInstances.Seattle,
        start_datetime="2021-01-01",
        end_datetime="2021-01-10",
    )
    grid = plots.plot_query_semantic_similarity_history(
        semantic_hist,
    )
    grid.savefig("test-generated-semantic-sim-histories-plot.pdf")

    # Or with extra kwargs like order
    grid_third_order = plots.plot_query_semantic_similarity_history(
        semantic_hist,
        lmplot_kws={"order": 3},
    )
    grid_third_order.savefig("test-generated-semantic-sim-histories-plot.pdf")
