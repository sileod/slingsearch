# Scholar benchmark interpretation audit

The saved experiments measure recovery and ordering of annotated target papers. They do not exhaustively measure useful literature discovery or distinct evidence coverage. Historical scores remain unchanged.

## Concrete failure case: litsearch-39

The question asks for research on topic and frame conditioning in transformer language models for argument claims. The saved labels identify Aspect-Controlled Neural Argument Generation (2021), DOI 10.18653/v1/2021.naacl-main.34. Exa retrieves that target at rank three; the guarded blend omits it and receives zero nDCG@5.

The blend instead includes Strategies for framing argumentative conclusion generation (2022), DOI 10.18653/v1/2022.inlg-main.20. This alternative is unjudged, rather than an explicitly assessed negative. Its saved text discusses framing-conditioned generation with transformers. It merits independent assessment; this audit does not assign it a positive label or assert it fully satisfies every query constraint.

There are also genuine implementation problems in this case:

- The DOI URL and ACL PDF for the 2022 paper both occupy final result slots. The publisher confirms that these are the same paper: https://aclanthology.org/2022.inlg-main.20/ . Diversity is overstated unless versions are collapsed.
- The annotated target's supplied ranking text is essentially bibliographic metadata, whereas alternatives contain substantive text. The official target page has an abstract describing topic, stance, and aspect control: https://aclanthology.org/2021.naacl-main.34/ . Equal maximum text budgets do not imply equal evidence quality.

## What can and cannot be concluded

- The negative known-target result is real under the saved labels. It should remain reported.
- Treating every unjudged alternative as irrelevant is insufficient to establish that the blend is less useful for broad scholarly discovery.
- Label incompleteness does not establish that the blend is better. Duplicate versions and missing evidence are product defects, not excuses to discard negative outcomes.
- Sixteen held-out queries are a small sample. The guarded blend versus Exa confidence interval includes both gains and losses; Jev superiority is not established.

## Required complementary evaluation

Retain the known-target track. Add a separate pooled assessment of unique papers from every compared method, with provider and method hidden from assessors. Present equivalent title/abstract evidence; judge exact question constraints, useful evidence, and distinct contribution. Have humans adjudicate uncertain judgments. Report relevance and additional useful-paper coverage separately, alongside token cost. Keep passage usefulness separate from document relevance. Freeze policies before testing fresh questions, and never tune on those judgments.

This audit used saved artifacts and public publisher pages. No additional paid search or ranking requests were made.
