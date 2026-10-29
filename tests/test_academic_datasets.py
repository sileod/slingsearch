import json
import pytest
from benchmarks.datasets import sage, litqa2_search


def write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_sage_grades_and_no_reference_leakage(tmp_path):
    paper = {'paperId': 'abc', 'title': 'Hidden target'}
    path = write(tmp_path, 'open.json', {'questions': [
        {'question': 'Find evidence', 'generation_plan': 'secret',
         'ground_truth': {'most_relevant': [paper], 'relevant': [paper, {'paperId': 'def', 'title': 'Other'}]}}]})
    query, = sage.load(path, kind='open', domain='healthcare')
    assert query.question == 'Find evidence'
    assert query.qrels == {'semantic_scholar:abc': 2, 'semantic_scholar:def': 1}
    short = write(tmp_path, 'short.json', [{'complete_query': 'Find one paper', 'ground_truth': paper}])
    assert sage.load(short, kind='short', domain='humanities')[0].qrels == {'semantic_scholar:abc': 1}
    with pytest.raises(ValueError): sage.load(short, kind='other', domain='humanities')


def test_litqa_uses_official_subset_and_only_question(tmp_path):
    questions = write(tmp_path, 'questions.json', [
        {'id': 'a', 'question': 'Which evidence?', 'ideal': 'Secret', 'key-passage': 'Hidden'},
        {'id': 'b', 'question': 'Test question'}])
    mapping = write(tmp_path, 'mapping.json', {
        'a': {'query_id': 'litqa2_2', 'corpus_ids': [123]},
        'b': {'query_id': 'litqa2_3', 'corpus_ids': [456]}})
    subset = write(tmp_path, 'subset.json', {'validation': ['litqa2_2'], 'test': ['litqa2_3']})
    query, = litqa2_search.load(questions, mapping, subset)
    assert query.question == 'Which evidence?' and query.qrels == {'corpusid:123': 1}
    assert litqa2_search.load(questions, mapping, subset, split='test')[0].id == 'litqa2_3'
    write(tmp_path, 'subset.json', {'validation': ['missing']})
    with pytest.raises(ValueError, match='Missing'): litqa2_search.load(questions, mapping, subset)
