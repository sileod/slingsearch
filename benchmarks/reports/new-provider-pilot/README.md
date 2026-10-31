# Expand and You.com discovery smoke comparison

| Provider | Successful queries | Results | Text-bearing results | URLs absent from the other two (summed per query) |
|---|---:|---:|---:|---:|
| exa | 2/2 | 10 | 10 | 10 |
| expand | 2/2 | 10 | 10 | 10 |
| you | 2/2 | 10 | 10 | 10 |

Unjudged URL diversity is not a quality or relevance gain; copies or paper versions can have different URLs. Two hand-picked questions do not justify changing defaults.
Reproduce: `python -m benchmarks.new_provider_pilot` with keys configured. Uses the shared experiment budget/cache. No models or full-page extraction are called.
