import pytest

from hybridsearch import Document, InMemoryStore

CORPUS = [
    Document(
        "doc-114",
        "An appeal against an administrative decision must be filed within sixty days of"
        " notification.",
    ),
    Document(
        "doc-322",
        "The appellant may withdraw the appeal at any time before the hearing begins.",
    ),
    Document(
        "doc-455",
        "Where the appeal is withdrawn, the appellant bears the court fees in full.",
    ),
    Document(
        "doc-771",
        "Procedural deadlines are suspended from the first to the thirty-first of August each"
        " year.",
    ),
    Document(
        "doc-208",
        "A notice of appeal shall be in writing and signed by counsel admitted to practise.",
    ),
    Document(
        "doc-640",
        "Filings submitted through the certified electronic portal are accepted as of the filing"
        " date.",
    ),
    Document(
        "doc-512",
        "The court shall issue its decision within ninety days of the hearing.",
    ),
    Document(
        "doc-901",
        "Where the appellant fails to appear, the appeal is declared inadmissible.",
    ),
    Document(
        "doc-118",
        "Deadlines run from the day following notification.",
    ),
    Document(
        "doc-777",
        "Legal costs are ordinarily recoverable by the winning party.",
    ),
]


@pytest.fixture
def store():
    store = InMemoryStore()
    store.index(CORPUS)
    return store
