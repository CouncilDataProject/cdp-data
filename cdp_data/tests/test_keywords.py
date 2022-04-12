#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cdp_data import keywords

###############################################################################


def test_keywords_quickstart() -> None:
    ngram_usage = keywords.compute_ngram_usage_history(
        "cdp-seattle-21723dcf",
        start_datetime="2021-01-01",
        end_datetime="2021-02-01",
    )
    grid = keywords.plot_ngram_usage_histories(
        ["police", "housing", "transportation"],
        ngram_usage,
    )
    grid.savefig("test-generated-ngram-usage-histories-plot.pdf")

    # Or with extra kwargs like order
    grid_third_order = keywords.plot_ngram_usage_histories(
        ["police", "housing", "transportation"],
        ngram_usage,
        lmplot_kws={"order": 3},
    )
    grid_third_order.savefig("test-generated-ngram-usage-histories-plot.pdf")
