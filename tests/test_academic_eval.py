from benchmarks.academic_eval import score
from benchmarks.datasets import Query
from slingsearch import Result
from benchmarks.search_cli_baseline import normalize_url, rank


def test_exact_gold_aliases_weighted_recall_no_title_matching():
    query = Query('q', 'question', qrels={'semantic_scholar:abc': 2, 'semantic_scholar:def': 1})
    gold = {'abc': {'paperId': 'abc', 'externalIds': {'DOI': '10.1/a'}, 'corpusId': 123}}
    paper = Result('doi:10.1/a', 'Title', 'https://doi.org/10.1/a', doi='10.1/a')
    unrelated = Result('other', 'Title', 'https://example.org/other')
    values = score(query, [paper, unrelated], gold)
    assert values['WeightedRecall@10'] == 2 / 3 and values['Recall@10'] == .5
    assert score(query, [unrelated], gold)['Hit@10'] == 0


def test_search_cli_port_url_contract_consensus_and_determinism():
    assert normalize_url('https://www.Example.org/a?utm_source=x&page=2') == 'example.org/a?page=2'
    assert normalize_url('https://example.org/files/www.x?redirect=http://a') == 'example.org/files/www.x?redirect=http://a'
    a = Result('a', 'a', 'https://example.org/a')
    b = Result('b', 'b', 'https://example.org/b')
    buckets = [('z', [a, b]), ('a', [b])]
    assert [r.id for r in rank(buckets)] == ['b', 'a']
    assert [r.id for r in rank(buckets[::-1])] == ['b', 'a']
