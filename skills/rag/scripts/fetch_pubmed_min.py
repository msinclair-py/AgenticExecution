"""Minimal PubMed fetcher: esearch + efetch abstracts -> RAG records.

Self-contained (requests only). Returns records with a 'text' field ready for
rag_build.build_store. For the richer pipeline (UniProt-curated PMIDs, Europe
PMC, Semantic Scholar, Unpaywall PDFs, PMC full text), use the bindify toolbelt's
fetch_pubmed.fetch_for_protein — see references/generators_reference.md.

Set NCBI_API_KEY to raise the rate limit (3 -> 10 req/s).
"""
from __future__ import annotations

import os
import time
from xml.etree import ElementTree as ET

import requests

EUTILS = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils'


def _key():
    k = os.getenv('NCBI_API_KEY')
    return {'api_key': k} if k else {}


def _sleep():
    time.sleep(0.11 if os.getenv('NCBI_API_KEY') else 0.34)


def esearch(term: str, retmax: int) -> list[str]:
    params = {'db': 'pubmed', 'term': term, 'retmax': min(retmax, 10000),
              'retmode': 'json', **_key()}
    data = requests.get(f'{EUTILS}/esearch.fcgi', params=params, timeout=30).json()
    _sleep()
    return data.get('esearchresult', {}).get('idlist', [])


def efetch_abstracts(pmids: list[str], batch_size: int = 200) -> list[dict]:
    out = []
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        params = {'db': 'pubmed', 'id': ','.join(batch),
                  'rettype': 'abstract', 'retmode': 'xml', **_key()}
        resp = requests.post(f'{EUTILS}/efetch.fcgi', data=params, timeout=120)
        _sleep()
        out.extend(_parse(resp.text))
    return out


def _parse(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for art in root.findall('.//PubmedArticle'):
        pmid_el = art.find('.//PMID')
        pmid = pmid_el.text if pmid_el is not None else None
        title_el = art.find('.//ArticleTitle')
        title = ''.join(title_el.itertext()).strip() if title_el is not None else ''
        parts = []
        for ab in art.findall('.//Abstract/AbstractText'):
            label, txt = ab.get('Label'), ''.join(ab.itertext()).strip()
            if txt:
                parts.append(f'{label}: {txt}' if label else txt)
        abstract = '\n'.join(parts)
        if not pmid or (not title and not abstract):
            continue
        out.append({'pmid': pmid, 'title': title, 'abstract': abstract,
                    'text': (title + '\n\n' + abstract).strip(), 'source_db': 'MED'})
    return out


def fetch_pubmed_abstracts(query: str, max_pmids: int = 200) -> list[dict]:
    """query can be a gene term ('NF1[Gene]') or free text. Returns RAG records."""
    pmids = esearch(query, max_pmids)
    return efetch_abstracts(pmids)


if __name__ == '__main__':
    import sys
    recs = fetch_pubmed_abstracts(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50)
    print(f'{len(recs)} records')
