#!/usr/bin/env python
# -*- coding: utf-8 -*-

from cdp_data import CDPInstances, keywords, plots
plots.set_cdp_plotting_styles()

START_DATETIME = "2022-03-01"
END_DATETIME = "2022-10-01"

# Compute ngrams for a bunch of infrastructures
print("Generating header")
ngram_usage = keywords.compute_ngram_usage_history(
    [CDPInstances.Seattle, CDPInstances.Portland, CDPInstances.Oakland],
    start_datetime=START_DATETIME,
    end_datetime=END_DATETIME,
)
grid = plots.plot_ngram_usage_histories(
    ["police", "housing", "equity"],
    ngram_usage,
    lmplot_kws=dict(  # extra plotting params
        col="ngram",
        row="infrastructure",
        hue="ngram",
        scatter_kws={"alpha": 0.2},
        aspect=1.6,
    ),
)
grid.savefig("header-keywords-over-time.png")

# Compute ngrams for just Seattle
print("Generating Seattle ngrams usage")
seattle_ngram_usage = ngram_usage.loc[ngram_usage.infrastructure == CDPInstances.Seattle]
grid = plots.plot_ngram_usage_histories(
    ["police", "housing"],
    seattle_ngram_usage,
    lmplot_kws=dict(  # extra plotting params
        col="ngram",
        hue="ngram",
        scatter_kws={"alpha": 0.2},
        aspect=1.6,
    ),
)
grid.savefig("seattle-keywords-over-time.png")

# Compute semantic sim for just Seattle
# print("Generating Seattle semantic sim")
# semantic_hist = keywords.compute_query_semantic_similarity_history(
#     ["defund the police", "missing middle housing"],
#     CDPInstances.Seattle,
#     start_datetime=START_DATETIME,
#     end_datetime=END_DATETIME,
# )
# grid = plots.plot_query_semantic_similarity_history(
#     semantic_hist,
#     lmplot_kws=dict(  # extra plotting params
#         col="query",
#         hue="pooling",
#         scatter_kws={"alpha": 0.2},
#         aspect=1.6,
#     ),
# )
# grid.savefig("seattle-semantic-sim-over-time.png")