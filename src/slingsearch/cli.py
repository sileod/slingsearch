import argparse
import json
import sys
from .api import Client, search
from .review_models import load_papers
from .config import MODELS


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('novelty', 'related-work'):
        command = sys.argv[1]
        parser = argparse.ArgumentParser(prog='slingsearch ' + command)
        if command == 'novelty':
            parser.add_argument('contribution')
            parser.add_argument('--papers', help='Existing related-work Markdown, JSON, JSONL or one ID per line')
        else:
            parser.add_argument('papers')
            parser.add_argument('--contribution')
        parser.add_argument('--claim', dest='claims', action='append')
        parser.add_argument('--query', dest='queries', action='append')
        parser.add_argument('--providers', nargs='+')
        parser.add_argument('--ranker', choices=['rrf', 'ensemble', *MODELS], default='jev')
        parser.add_argument('--snippet-search', action='store_true')
        parser.add_argument('--enrich-snippets', action='store_true')
        parser.add_argument('--check', dest='checks', action='append')
        parser.add_argument('--check-threshold', type=float, default=.7)
        parser.add_argument('--min-relevance', type=float, default=.5)
        parser.add_argument('--no-relevance-filter', dest='min_relevance', action='store_const', const=None, default=argparse.SUPPRESS)
        parser.add_argument('--judge', choices=['jev', 'none'], default='jev')
        parser.add_argument('--no-graph', dest='expand_graph', action='store_false')
        parser.add_argument('--max-queries', type=int, default=8)
        parser.add_argument('--max-seeds', type=int, default=5)
        parser.add_argument('--graph-limit', type=int, default=5)
        parser.add_argument('--max-results', type=int, default=5)
        parser.add_argument('--token-budget', type=int, default=3000)
        parser.add_argument('--format', choices=['json','markdown'], default='markdown')
        args = vars(parser.parse_args(sys.argv[2:]))
        output_format = args.pop('format')
        args['papers'] = load_papers(args['papers']) if args.get('papers') else []
        args['judge'] = None if args['judge'] == 'none' else args['judge']
        with Client() as client:
            if command == 'novelty':
                args['related_work'] = args.pop('papers')
                report = client.novelty_search(**args)
            else:
                report = client.related_work_search(**args)
        print(report.to_json(indent=2) if output_format == 'json' else report.to_markdown())
        return
    # Preserve the original `slingsearch "question"` entry point.
    if len(sys.argv) > 1 and sys.argv[1] in ('doctor', 'usage', 'graph'):
        command = sys.argv[1]
        parser = argparse.ArgumentParser(prog='slingsearch ' + command)
        if command == 'graph':
            parser.add_argument('paper_id')
            parser.add_argument('--direction', choices=['references', 'citations', 'both'], default='both')
            parser.add_argument('--limit', type=int, default=10)
        else:
            parser.add_argument('--providers', nargs='+')
        args = vars(parser.parse_args(sys.argv[2:]))
        with Client() as client:
            result = client.citation_graph(**args) if command == 'graph' else getattr(client, command)(**args)
        print(json.dumps(result, indent=2))
        if command == 'doctor' and any(p['status'] not in ('ok', 'unconfigured') for p in result['providers']):
            raise SystemExit(1)
        return
    parser = argparse.ArgumentParser(description='Retrieve question-focused evidence; subcommands: doctor, usage, graph')
    parser.add_argument('question')
    parser.add_argument('--mode', choices=['web', 'scholar'], default='web')
    parser.add_argument('--ranker', choices=['rrf', 'jev', 'decider', 'voyage', 'd1', 'ensemble'], default='jev')
    parser.add_argument('--providers', nargs='+')
    parser.add_argument('--anchor-provider', help='Preserve this provider’s best available original result')
    parser.add_argument('--diversity', action='store_true', help='Balance shortlist coverage and diversify near relevance ties')
    parser.add_argument('--no-relevance-filter', dest='min_relevance', action='store_const', const=None, default=argparse.SUPPRESS)
    parser.add_argument('--min-relevance', type=float, default=.5, help='Minimum normalized relevance from 0 to 1 (default .5; .75: directly addresses the question)')
    parser.add_argument('--check', dest='checks', action='append')
    parser.add_argument('--check-threshold', type=float, default=.7)
    parser.add_argument('--snippet-search', action='store_true')
    parser.add_argument('--enrich-snippets', action='store_true')
    parser.add_argument('--assess', action='store_true', help='Packed Jev usefulness assessment of the displayed evidence')
    parser.add_argument('--assessment-threshold', type=float, default=.7)
    parser.add_argument('--fast', action='store_true', help='Stop after sufficient text-bearing results and two providers, with a short grace period')
    parser.add_argument('--deadline', type=float, help='Total network deadline in seconds (default 3 with --fast)')
    parser.add_argument('--fast-min-results', type=int)
    parser.add_argument('--max-results', type=int, default=5)
    parser.add_argument('--token-budget', type=int, default=1500)
    parser.add_argument('--format', choices=['json', 'markdown'], default='markdown')
    args = vars(parser.parse_args())
    format = args.pop('format')
    result = search(**args)
    print(result.to_json(indent=2) if format == 'json' else result.to_markdown())
