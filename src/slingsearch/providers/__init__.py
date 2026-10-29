from .exa import Exa
from .parallel import Parallel
from .brave import Brave
from .tavily import Tavily
from .firecrawl import Firecrawl
from .semantic_scholar import SemanticScholar
from .openalex import OpenAlex
from .serpapi import SerpAPI
from .github import GitHub
from .huggingface import HuggingFace
from .papers_with_code import PapersWithCode

PROVIDERS = {p.name: p for p in (Exa, Parallel, Brave, Tavily, Firecrawl, SemanticScholar, OpenAlex, SerpAPI, GitHub, PapersWithCode, HuggingFace)}
