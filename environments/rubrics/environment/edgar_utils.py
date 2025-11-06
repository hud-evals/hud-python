import asyncio
import logging
import os
from typing import Any, Callable, Dict
from urllib.parse import urlparse

import httpx
from edgar import Company, Filing

logger = logging.getLogger(__name__)


async def get_content_with_fallback(filing: Filing, filing_url: str) -> str:
    content = ""
    try:
        content = filing.text()
    except Exception:
        try:
            content = filing.text
        except Exception:
            content = ""

    # Fallback to HTML
    if not content:
        try:
            content = filing.html()
        except Exception:
            try:
                content = filing.html
            except Exception:
                content = ""

    if not content:
        try:
            edgar_identity = os.getenv("EDGAR_IDENTITY")
            if not edgar_identity:
                raise ValueError("EDGAR_IDENTITY environment variable not set")

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(filing_url, headers={"User-Agent": edgar_identity})
                resp.raise_for_status()
                content = resp.text
        except Exception:
            content = ""

    return content


def populate_financal_data(financials_fn: Callable[[], Any]) -> Dict[str, Any]:
    fn_name = getattr(financials_fn, "__name__", "unknown")
    try:
        data = financials_fn()
        if data is not None:
            return {
                fn_name: {
                    "data": data.to_dict(orient="index")
                    if hasattr(data, "to_dict")
                    else str(data)[:5000],
                    "columns": list(data.columns) if hasattr(data, "columns") else None,
                }
            }
    except Exception as e:
        logger.warning(f"Could not extract {fn_name}: {e}")
    return {}


def get_filing_from_url(filing_url: str) -> Filing:
    parsed = urlparse(filing_url)
    path_parts = parsed.path.split("/")

    accession_no_dashes = None
    for part in path_parts:
        if len(part) >= 18 and part.isdigit():
            accession_no_dashes = part
            break

    if not accession_no_dashes:
        raise ValueError(f"Could not extract accession number from URL: {filing_url}")

    accession = (
        f"{accession_no_dashes[:10]}-{accession_no_dashes[10:12]}-{accession_no_dashes[12:]}"
    )

    filing = None
    try:
        cik = None
        parts = [p for p in path_parts if p]
        if "data" in parts:
            idx = parts.index("data")
            if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                cik = parts[idx + 1]

        if cik:
            comp = Company(cik)
            for f in comp.get_filings():
                if getattr(f, "accession_number", "").replace("-", "") == accession_no_dashes:
                    filing = f
                    break
    except Exception:
        filing = None

    if filing is None:
        try:
            filing = Filing(accession)
        except TypeError:
            filing = None

    if filing is None:
        raise LookupError(f"Filing not found for accession {accession}")

    return filing


def get_filing_by_accession(identifier: str, accession_number: str) -> Filing:
    company = Company(identifier)
    filing = None

    for f in company.get_filings():
        if f.accession_number.replace("-", "") == accession_number.replace("-", ""):
            filing = f
            break

    if filing is None:
        raise LookupError(f"Filing {accession_number} not found for {identifier}")

    return filing
