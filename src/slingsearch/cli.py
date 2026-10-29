import argparse
import json
import sys
from .api import Client, search


def main():
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
