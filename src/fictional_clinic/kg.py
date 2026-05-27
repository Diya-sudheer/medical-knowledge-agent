from __future__ import annotations

import html
import json
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fictional_clinic.models import Source
from fictional_clinic.rag import STOPWORDS, tokenize


class WikidataClient:
    """Small Wikidata adapter for optional live open-knowledge lookup."""

    endpoint = "https://www.wikidata.org/w/api.php"

    def search(self, query: str, limit: int = 2, timeout: float = 4.0) -> list[Source]:
        search_text = knowledge_query(query)
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "search": search_text,
            "limit": str(limit),
        }
        request = Request(
            f"{self.endpoint}?{urlencode(params)}",
            headers={"User-Agent": "fictional-clinic-learning-agent/0.1"},
        )

        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        sources: list[Source] = []
        for item in payload.get("search", []):
            entity_id = item.get("id", "")
            label = item.get("label", entity_id)
            description = item.get("description") or "No Wikidata description available."
            url = item.get("concepturi") or f"https://www.wikidata.org/wiki/{entity_id}"
            sources.append(
                Source(
                    title=f"Wikidata: {label}",
                    path=url,
                    snippet=description,
                    score=0.0,
                )
            )
        return sources


class OpenAlexClient:
    """Small OpenAlex adapter for optional paper and article references."""

    endpoint = "https://api.openalex.org/works"

    def search(self, query: str, limit: int = 2, timeout: float = 6.0) -> list[Source]:
        search_text = knowledge_query(query)
        params = {
            "search": search_text,
            "per-page": str(limit),
        }
        request = Request(
            f"{self.endpoint}?{urlencode(params)}",
            headers={
                "Accept": "application/json",
                "User-Agent": "fictional-clinic-learning-agent/0.1",
            },
        )

        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        sources: list[Source] = []
        for item in payload.get("results", [])[:limit]:
            title = item.get("title") or "OpenAlex work"
            year = item.get("publication_year")
            cited_by = item.get("cited_by_count")
            url = item.get("doi") or item.get("id") or ""
            details = []
            if year:
                details.append(f"published {year}")
            if cited_by is not None:
                details.append(f"cited by {cited_by} works")
            source_name = _primary_source_name(item)
            if source_name:
                details.append(f"source: {source_name}")
            snippet = "; ".join(details) or "Scholarly work indexed by OpenAlex."
            sources.append(
                Source(
                    title=f"OpenAlex: {title}",
                    path=url,
                    snippet=snippet,
                    score=0.0,
                )
            )
        return sources


class DbpediaClient:
    """Small DBpedia SPARQL adapter for optional encyclopedia context."""

    endpoint = "https://dbpedia.org/sparql"
    lookup_endpoint = "https://lookup.dbpedia.org/api/search"

    def search(self, query: str, limit: int = 2, timeout: float = 6.0) -> list[Source]:
        search_text = knowledge_query(query)
        lookup_sources = self._search_lookup(search_text, limit, timeout)
        if lookup_sources:
            return lookup_sources
        sources = self._search_sparql(search_text, limit, timeout)
        if sources:
            return sources
        resource_source = self._search_resource_sparql(search_text, timeout)
        if resource_source:
            return [resource_source]
        linked_data_source = self._search_linked_data(search_text, timeout)
        return [linked_data_source] if linked_data_source else []

    def _search_lookup(self, search_text: str, limit: int, timeout: float) -> list[Source]:
        params = {
            "format": "JSON",
            "query": search_text,
            "maxResults": str(limit),
        }
        request = Request(
            f"{self.lookup_endpoint}?{urlencode(params)}",
            headers={
                "Accept": "application/json",
                "User-Agent": "fictional-clinic-learning-agent/0.1",
            },
        )

        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        sources: list[Source] = []
        for item in payload.get("docs", [])[:limit]:
            label = _clean_lookup_text(_first_value(item.get("label"))) or "DBpedia result"
            resource = _first_value(item.get("resource")) or ""
            comment = (
                _clean_lookup_text(_first_value(item.get("comment")))
                or "No DBpedia summary available."
            )
            sources.append(
                Source(
                    title=f"DBpedia: {label}",
                    path=resource,
                    snippet=_trim(comment),
                    score=0.0,
                )
            )
            if label.lower() == search_text.lower():
                return sources
        return sources

    def _search_sparql(self, search_text: str, limit: int, timeout: float) -> list[Source]:
        sparql = f"""
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?resource ?label ?abstract WHERE {{
  ?resource rdfs:label ?label ;
            dbo:abstract ?abstract .
  FILTER (lang(?label) = "en")
  FILTER (lang(?abstract) = "en")
  FILTER CONTAINS(LCASE(STR(?label)), LCASE("{_sparql_string(search_text)}"))
}}
LIMIT {limit}
"""
        params = {
            "query": sparql,
            "format": "application/sparql-results+json",
        }
        request = Request(
            f"{self.endpoint}?{urlencode(params)}",
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "fictional-clinic-learning-agent/0.1",
            },
        )

        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        sources: list[Source] = []
        for item in payload.get("results", {}).get("bindings", []):
            label = item.get("label", {}).get("value", "DBpedia result")
            resource = item.get("resource", {}).get("value", "")
            abstract = item.get("abstract", {}).get("value", "")
            sources.append(
                Source(
                    title=f"DBpedia: {label}",
                    path=resource,
                    snippet=_trim(abstract),
                    score=0.0,
                )
            )
        return sources

    def _search_resource_sparql(self, search_text: str, timeout: float) -> Source | None:
        resource_name = _dbpedia_resource_name(search_text)
        sparql = f"""
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?label ?abstract WHERE {{
  dbr:{resource_name} rdfs:label ?label ;
                      dbo:abstract ?abstract .
  FILTER (lang(?label) = "en")
  FILTER (lang(?abstract) = "en")
}}
LIMIT 1
"""
        params = {
            "query": sparql,
            "format": "application/sparql-results+json",
        }
        request = Request(
            f"{self.endpoint}?{urlencode(params)}",
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": "fictional-clinic-learning-agent/0.1",
            },
        )

        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        bindings = payload.get("results", {}).get("bindings", [])
        if not bindings:
            return None

        item = bindings[0]
        label = item.get("label", {}).get("value", search_text.title())
        abstract = item.get("abstract", {}).get("value", "")
        if not abstract:
            return None

        return Source(
            title=f"DBpedia: {label}",
            path=f"http://dbpedia.org/resource/{resource_name}",
            snippet=_trim(abstract),
            score=0.0,
        )

    def _search_linked_data(self, search_text: str, timeout: float) -> Source | None:
        resource_name = _dbpedia_resource_name(search_text)
        resource_uri = f"http://dbpedia.org/resource/{resource_name}"
        request = Request(
            f"https://dbpedia.org/data/{resource_name}.json",
            headers={
                "Accept": "application/json",
                "User-Agent": "fictional-clinic-learning-agent/0.1",
            },
        )

        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))

        resource = payload.get(resource_uri) or payload.get(resource_uri.replace("http:", "https:"))
        if not resource:
            return None

        label = _literal_value(resource, "http://www.w3.org/2000/01/rdf-schema#label")
        abstract = _literal_value(resource, "http://dbpedia.org/ontology/abstract")
        if not abstract:
            return None

        return Source(
            title=f"DBpedia: {label or search_text.title()}",
            path=resource_uri,
            snippet=_trim(abstract),
            score=0.0,
        )


def knowledge_query(query: str) -> str:
    terms: list[str] = []
    for token in tokenize(query):
        if token in STOPWORDS or len(token) <= 2 or token in terms:
            continue
        terms.append(token)
    return " ".join(terms) or query.strip()


def _dbpedia_resource_name(search_text: str) -> str:
    return "_".join(search_text.split()).capitalize()


def _sparql_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _trim(value: str, max_chars: int = 420) -> str:
    compact = " ".join(value.split())
    return compact[:max_chars].rstrip() + ("..." if len(compact) > max_chars else "")


def _literal_value(resource: dict[str, list[dict[str, str]]], predicate: str) -> str:
    for item in resource.get(predicate, []):
        if item.get("lang") == "en" and item.get("value"):
            return item["value"]
    for item in resource.get(predicate, []):
        if item.get("value"):
            return item["value"]
    return ""


def _first_value(value: object) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if isinstance(value, str):
        return value
    return ""


def _primary_source_name(work: dict[str, object]) -> str:
    primary_location = work.get("primary_location")
    if not isinstance(primary_location, dict):
        return ""
    source = primary_location.get("source")
    if not isinstance(source, dict):
        return ""
    display_name = source.get("display_name")
    return str(display_name) if display_name else ""


def _clean_lookup_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", "", value)
    return html.unescape(without_tags).replace("â", "-").strip()
