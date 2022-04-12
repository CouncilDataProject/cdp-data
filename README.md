# cdp-data

[![Build Status](https://github.com/CouncilDataProject/cdp-data/workflows/Build/badge.svg)](https://github.com/CouncilDataProject/cdp-data/actions)
[![Documentation](https://github.com/CouncilDataProject/cdp-data/workflows/Documentation/badge.svg)](https://CouncilDataProject.github.io/cdp-data)

Data Utilities and Processing Generalized for All CDP Instances

---

## Installation

**Stable Release:** `pip install cdp-data`<br>
**Development Head:** `pip install git+https://github.com/CouncilDataProject/cdp-data.git`

## Documentation

For full package documentation please visit [councildataproject.github.io/cdp-data](https://councildataproject.github.io/cdp-data).

## Quickstart

### Pulling Datasets

Install basics: `pip install cdp-data`

```python
from cdp_data import datasets
ds = datasets.get_session_dataset(
    infrastructure_slug="cdp-seattle-21723dcf",
    start_datetime="2021-01-01",
    store_transcript=True,
)
```

### Plotting and Analysis

Install plotting support: `pip install cdp-data[plot]`

```python
from cdp_data import keywords
ngram_usage = keywords.compute_ngram_usage_history(
    "cdp-seattle-21723dcf",
    start_datetime="2021-01-01",
)
grid = keywords.plot_ngram_usage_histories(
    ["police", "housing", "transportation"],
    ngram_usage,
)
grid.savefig("seattle-keywords-over-time.pdf")

# Or with extra kwargs like order
grid_third_order = keywords.plot_ngram_usage_histories(
    ["police", "housing", "transportation"],
    ngram_usage,
    lmplot_kws={"order": 3},
)
grid_third_order.savefig("seattle-keywords-over-time-third-order.pdf")
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT license**
