from slingsearch import Result, Source
from slingsearch.diversity import balanced_shortlist, diverse_order
from slingsearch.fusion import fuse
from slingsearch.providers.queries import scholar_keywords


def doc(identifier, provider, rank, text='', score=None):
    return Result(identifier, identifier, f'https://example.org/{identifier}',
                  passages=[text], sources=[Source(provider, rank)], model_score=score)


def test_shortlist_reserves_room_for_unique_provider_contributions():
    raw = [doc(f'exa{i}', 'exa', i) for i in range(1, 11)]
    raw += [doc(f'other{i}', 'openalex', i) for i in range(1, 11)]
    candidates = fuse(raw)
    # Deliberately put Exa first: the selector must still retain other retrieval.
    candidates.sort(key=lambda r: r.sources[0].provider != 'exa')
    selected = balanced_shortlist(candidates, 6)
    assert len(selected) == 6
    assert sum(r.sources[0].provider == 'exa' for r in selected) == 3
    assert sum(r.sources[0].provider == 'openalex' for r in selected) == 3
    assert {r.id for r in selected} <= {r.id for r in candidates}


def test_shortlist_does_not_duplicate_shared_papers_or_invent_provenance():
    shared = doc('shared','exa',1)
    shared.sources.append(Source('openalex',1))
    raw = [shared, doc('a','exa',2), doc('b','openalex',2)]
    selected = balanced_shortlist(raw,3)
    assert len({r.id for r in selected}) == 3
    assert selected[0].sources == [Source('exa',1), Source('openalex',1)]


def test_final_diversity_breaks_near_ties_without_promoting_irrelevance():
    first = doc('first','exa',1,'neural forecasting uncertainty calibration',5)
    similar = doc('second','exa',2,'neural forecasting uncertainty calibration',4.99)
    novel = doc('third','openalex',1,'conformal prediction coverage guarantees',4.98)
    weak = doc('weak','semantic_scholar',1,'completely distinct vocabulary',1)
    result = diverse_order([first,similar,novel,weak])
    assert result[0] is first and result[1] is novel
    assert result[-1] is weak
    assert [r.model_score for r in [first,similar,novel,weak]] == [5,4.99,4.98,1]


def test_keyword_cleanup_preserves_scientific_entities_and_original_question():
    question = 'Can you recommend papers using NLI for few-shot relation extraction with wikiHowToImprove?'
    words = scholar_keywords(question)
    assert 'NLI' in words and 'wikiHowToImprove' in words
    assert 'few shot' in words and 'recommend' not in words
    assert question.startswith('Can you recommend')


def test_benchmark_counts_novel_relevant_paper_retention():
    from dataclasses import asdict
    from benchmarks.datasets import Query
    from benchmarks.diversity import evaluate
    rows = [doc('target_a','exa',1,'forecasting neural uncertainty calibration',5)]
    rows += [doc(f'duplicate{i}','exa',i+1,'forecasting neural uncertainty calibration',4.99) for i in range(1,6)]
    rows += [doc('target_b','openalex',1,'conformal coverage formal guarantees',4.98),
             doc('weak','openalex',2,'unrelated biology species',1)]
    query = Query('q','forecasting uncertainty',candidates=rows,
                  qrels={'https://example.org/target_a':1,'https://example.org/target_b':1})
    pool = fuse(rows)
    # fuse deliberately clears model scores; restore the fixed scoring fixture.
    scores = {r.title:r.model_score for r in rows}
    for r in pool:r.model_score=scores[r.title]
    ranked=sorted(pool,key=lambda r:-r.model_score)
    result=evaluate([query],[{'query_id':'q','success':True,'results':[asdict(r) for r in ranked]}],{'heldout_indices':[]})
    summary=result['summary']['development']
    assert summary['global_balanced']['novel_relevant_retained'] == 0
    assert summary['diverse_0.05']['novel_relevant_retained'] == 1
    assert result['selected_on_development'].startswith('diverse_')
    assert all(r['output_tokens'] <= 1500 for r in result['records'])


def test_provider_anchor_preserves_scores_and_complementary_order():
    from slingsearch.diversity import preserve_provider_first
    other=doc('other','openalex',1,'different evidence',5)
    exa_later=doc('exa_later','exa',2,'later exa result',4.9)
    exa_first=doc('exa_first','exa',1,'first exa result',4.8)
    rows=[other,exa_later,exa_first]
    result=preserve_provider_first(rows,'exa')
    assert result == [exa_first,other,exa_later]
    assert [r.model_score for r in result] == [4.8,5,4.9]
    assert preserve_provider_first(rows,'missing') == rows
