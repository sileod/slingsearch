import math
import pytest
from slingsearch import Config, Result
from slingsearch.models import Source
from slingsearch.fusion import fuse
from slingsearch.diversity import reserve_shortlist


def doc(name, provider, rank=1):
    return Result(name, name, 'https://example.org/'+name, sources=[Source(provider, rank)])


def test_weights_preserve_provenance_and_one_vote_per_provider():
    a = doc('a', 'semantic_scholar')
    duplicate = doc('a', 'semantic_scholar', 3)
    ranked = fuse([a, duplicate, doc('b', 'exa', 2)], {'semantic_scholar': .5})
    assert ranked[0].id.endswith('/b')
    assert ranked[1].rrf_score == pytest.approx(.5/61)
    assert {s.rank for s in ranked[1].sources} == {1, 3}
    assert fuse([a, doc('b', 'exa', 2)])[0].id.endswith('/a')


def test_reservation_admits_candidate_without_changing_fused_order():
    candidates = fuse([doc('a','exa'),doc('b','exa',2),doc('c','serpapi',20)], {'serpapi':.5})
    chosen = reserve_shortlist(candidates,2,['serpapi'])
    assert [r.title for r in chosen] == ['a','c']
    assert reserve_shortlist(candidates,0,['serpapi']) == []
    assert reserve_shortlist(candidates,2) == candidates[:2]
    assert len(reserve_shortlist(candidates,1,['serpapi','exa'])) == 1


@pytest.mark.parametrize('weights',[{'unknown':1},{'exa':-1},{'exa':0},{'exa':math.nan},{'exa':math.inf}])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        Config(scholar_provider_weights=weights)
