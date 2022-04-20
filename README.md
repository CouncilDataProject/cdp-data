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
from cdp_data import CDPInstances, datasets
ds = datasets.get_session_dataset(
    infrastructure_slug=CDPInstances.Seattle,
    start_datetime="2021-01-01",
    store_transcript=True,
)
```

### Plotting and Analysis

![Seattle keyword usage over time](https://raw.githubusercontent.com/CouncilDataProject/cdp-data/main/docs/_static/seattle-keywords-over-time.png)

Install plotting support: `pip install cdp-data[plot]`

```python
from cdp_data import CDPInstances, keywords
ngram_usage = keywords.compute_ngram_usage_history(
    CDPInstances.Seattle,
    start_datetime="2021-01-01",
)
grid = keywords.plot_ngram_usage_histories(
    ["police", "housing", "transportation"],
    ngram_usage,
    lmplot_kws=dict(  # extra plotting params
        col="ngram",
        hue="ngram",
        scatter_kws={"alpha": 0.2},
        aspect=1.6,
    ),
)
grid.savefig("seattle-keywords-over-time.png")
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT license**
