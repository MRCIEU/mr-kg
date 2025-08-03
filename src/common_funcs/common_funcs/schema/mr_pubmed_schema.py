from typing import TypedDict

class RawMrPubmedRecord(TypedDict):
    """Raw PubMed record"""
    # PubMed ID
    pmid: str
    # abstract
    ab: str
    # publication date
    pub_date: str
    # title
    title: str
    # Journal ISSN
    journal_issn: str
    # Journal name
    journal: str
    # Author affiliation
    author_affil: str

RawMrData = list[RawMrPubmedRecord]
