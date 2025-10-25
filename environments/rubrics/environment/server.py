"""
FastAPI server for Rubrics environment with SEC EDGAR integration.
Manages SEC filing data access and state.
"""

import asyncio
import logging
import os
import traceback
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar

import httpx
import uvicorn
from edgar import Company, set_identity, get_filings as edgar_get_filings
from edgar.financials import Financials
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rubric import Rubric

from .edgar_utils import (
    get_content_with_fallback,
    get_filing_by_accession,
    get_filing_from_url,
    populate_financal_data,
)

from .exa_utils import execute_fetch, execute_search

T = TypeVar("T")


# Set up logging
logger = logging.getLogger(__name__)


async def call_with_exponential_backoff(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    **kwargs: Any,
) -> T:
    """
    Call an async function with exponential backoff on rate limit errors.

    Args:
        func: The async function to call
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts (default: 5)
        initial_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay in seconds (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        **kwargs: Keyword arguments for the function

    Returns:
        The result of the function call

    Raises:
        The last exception if all retries fail
    """
    last_exception: Optional[Exception] = None
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                last_exception = e
                if attempt < max_retries:
                    # Log the retry attempt
                    logger.warning(
                        "Rate limit hit (429), retrying in %s seconds... (attempt %s/%s)",
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(delay)
                    # Calculate next delay with exponential backoff
                    delay = min(delay * exponential_base, max_delay)
                else:
                    # All retries exhausted
                    raise
            else:
                # Not a rate limit error, raise immediately
                raise
        except Exception:
            # Not an HTTP error, raise immediately
            raise

    # This should never be reached, but just in case
    if last_exception:
        raise last_exception
    raise RuntimeError("Unexpected error in exponential backoff")


class _EnvState:
    """In-memory environment state for tracking usage and agent answer."""

    def __init__(self) -> None:
        self.search_count: int = 0
        self.fetch_count: int = 0
        self.submitted_answer: Optional[str] = None

    def reset(self) -> None:
        self.search_count = 0
        self.fetch_count = 0
        self.submitted_answer = None


state = _EnvState()


class SearchCompanyRequest(BaseModel):
    query: str


class GetFilingsRequest(BaseModel):
    ticker: str | None = None
    form_type: str | None = None
    limit: int = 10
    cutoff_date: str | None = None


class GetFilingContentRequest(BaseModel):
    filing_url: str


class AnswerRequest(BaseModel):
    final_answer: str


class EvaluateRequest(BaseModel):
    rubric: list[dict[str, str | float]]


class FilingByAccessionRequest(BaseModel):
    ticker: str
    accession_number: str


class FetchRequest(BaseModel):
    url: str


class SearchRequest(BaseModel):
    query: str


app = FastAPI(title="SEC EDGAR Environment API", version="0.1.0")


set_identity(os.getenv("EDGAR_IDENTITY"))


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "healthy"}


@app.post("/setup")
async def setup() -> Dict[str, Any]:
    state.reset()
    return {"ok": True}


@app.post("/search_company")
async def search_company(req: SearchCompanyRequest) -> List[Dict[str, str]]:
    """Search for a company by ticker or name."""
    try:
        company = Company(req.query)

        ticker = company.tickers[0] if company.tickers else ""

        results = [
            {
                "ticker": ticker,
                "name": company.name,
                "cik": str(company.cik),
                "message": f"Found company: {company.name} ({ticker})",
            }
        ]

        state.search_count += 1
        return results

    except Exception as e:
        logger.error(f"Company search failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Company search failed: {type(e).__name__}: {e}"
        )


@app.post("/get_filings")
async def get_filings(req: GetFilingsRequest) -> List[Dict[str, Any]]:
    """Get filings for a company (by ticker/CIK) or globally.

    Args:
        ticker: Optional ticker or CIK. If omitted, returns global recent filings
        form_type: Optional form filter (e.g., "10-K", "8-K", ["3","4","5"])
        limit: Max number of results
        cutoff_date: Optional date string (YYYY-MM-DD). Only filings on or after this date will be returned
    """
    try:
        results: list[dict[str, Any]] = []

        if req.cutoff_date:
            try:
                datetime.strptime(req.cutoff_date, "%Y-%m-%d")
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid cutoff_date format. Expected YYYY-MM-DD, got: {req.cutoff_date}. Error: {e}",
                )

        if req.ticker:
            company = Company(req.ticker)
            filings = (
                company.get_filings(form=req.form_type) if req.form_type else company.get_filings()
            )
        else:
            filings = edgar_get_filings(form=req.form_type)

        for i, filing in enumerate(filings):
            if i >= req.limit:
                break

            if req.cutoff_date and filing.filing_date:
                filing_date_str = (
                    filing.filing_date
                    if isinstance(filing.filing_date, str)
                    else filing.filing_date.strftime("%Y-%m-%d")
                )
                if filing_date_str < req.cutoff_date:
                    continue

            results.append(
                {
                    "filing_date": filing.filing_date
                    if isinstance(filing.filing_date, str)
                    else filing.filing_date.strftime("%Y-%m-%d")
                    if filing.filing_date
                    else "",
                    "form_type": filing.form,
                    "company": getattr(filing, "company", None),
                    "cik": getattr(filing, "cik", None),
                    "file_number": getattr(filing, "file_number", None),
                    "acceptance_datetime": getattr(filing, "acceptance_datetime", None),
                    "period_of_report": getattr(filing, "period_of_report", None),
                    "filing_url": getattr(filing, "filing_url", getattr(filing, "url", None)),
                    "accession_number": filing.accession_number,
                    "description": getattr(filing, "primary_doc_description", ""),
                }
            )

        state.search_count += 1
        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get filings failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Get filings failed: {type(e).__name__}: {e}")


@app.post("/get_filing_content")
async def get_filing_content(req: GetFilingContentRequest) -> Dict[str, str]:
    """Get the content of a specific filing."""
    try:
        filing = get_filing_from_url(req.filing_url)

        content = await get_content_with_fallback(filing, req.filing_url)

        max_length = 50000
        if len(content) > max_length:
            content = content[:max_length] + "\n\n...[truncated]"

        state.fetch_count += 1
        return {"content": content}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get filing content failed: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Get filing content failed: {type(e).__name__}: {e}"
        )


@app.post("/get_financial_data")
async def get_financial_data(req: FilingByAccessionRequest) -> Dict[str, Any]:
    """Extract financial statements and key metrics from a 10-K or 10-Q filing."""
    try:
        filing = get_filing_by_accession(req.ticker, req.accession_number)
        company = Company(req.ticker)

        result = {
            "accession_number": filing.accession_number,
            "form_type": filing.form,
            "filing_date": filing.filing_date.isoformat()
            if hasattr(filing.filing_date, "isoformat")
            else str(filing.filing_date),
            "has_financials": False,
            "financial_data": None,
        }

        try:
            financials = Financials.extract(filing)

            if financials:
                result["has_financials"] = True
                result["cik"] = str(company.cik)
                result["name"] = company.name
                financial_data = {}

                financial_sections = [
                    financials.income_statement,
                    financials.balance_sheet,
                    financials.cashflow_statement,
                    financials.statement_of_equity,
                    financials.comprehensive_income,
                ]

                for section in financial_sections:
                    financial_data.update(populate_financal_data(section))

                result["financial_data"] = financial_data
        except Exception as e:
            logger.warning(f"Could not extract financials: {e}")

        return {"success": True, **result}
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_financials failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"get_financials failed: {type(e).__name__}: {e}"
        )


@app.post("/get_segment_data")
async def get_segment_data(req: FilingByAccessionRequest) -> Dict[str, Any]:
    """Extract segment-level financial data from a 10-K or 10-Q filing."""
    try:
        filing = get_filing_by_accession(req.ticker, req.accession_number)
        company = Company(req.ticker)

        result = {
            "success": True,
            "accession_number": filing.accession_number,
            "form_type": filing.form,
            "filing_date": filing.filing_date.isoformat()
            if hasattr(filing.filing_date, "isoformat")
            else str(filing.filing_date),
            "cik": str(company.cik),
            "name": company.name,
            "has_segment_data": False,
            "segment_data": None,
        }

        try:
            filing_obj = filing.obj()

            if hasattr(filing_obj, "segments"):
                result["has_segment_data"] = True
                result["segment_data"] = str(filing_obj.segments)[:10000]
            elif hasattr(filing_obj, "notes") and hasattr(filing_obj.notes, "segments"):
                result["has_segment_data"] = True
                result["segment_data"] = str(filing_obj.notes.segments)[:10000]
        except Exception as e:
            logger.warning(f"Could not extract segment data: {e}")

        return result
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_segment_data failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"get_segment_data failed: {type(e).__name__}: {e}"
        )


@app.post("/answer")
async def answer(req: AnswerRequest) -> Dict[str, Any]:
    state.submitted_answer = req.final_answer
    return {"ok": True, "message": "Answer submitted"}


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest) -> Dict[str, Any]:
    submitted = state.submitted_answer
    if submitted is None:
        return {
            "reward": 0.0,
            "content": f"No answer submitted. Searches: {state.search_count}, Fetches: {state.fetch_count}",
            "done": False,
        }

    logger.info(f"Evaluating answer (length: {len(submitted)} chars)")
    logger.info(f"Answer preview: {submitted}")

    try:
        rubric = Rubric.from_dict(req.rubric)
        evaluation = await rubric.grade(submitted)
        reward = evaluation.score
        info = {"report": [r.model_dump() for r in evaluation.report] if evaluation.report else []}

        logger.info(f"Rubric evaluation completed. Score: {reward}")
        logger.info(f"Evaluation report: {info}")
        return {"reward": reward / 100, "info": info, "done": True}
    except Exception as e:
        logger.error(f"Rubric evaluation failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        return {
            "reward": 0.0,
            "content": f"Evaluation failed: {type(e).__name__}: {e}",
            "done": True,
        }


@app.post("/get_filing_sections")
async def get_filing_sections(req: FilingByAccessionRequest) -> Dict[str, Any]:
    """Get specific sections from a 10-K or 10-Q filing."""
    try:
        filing = get_filing_by_accession(req.ticker, req.accession_number)
        form_type = filing.form

        sections = {"form_type": form_type, "has_structure": False}

        try:
            filing_obj = filing.obj()
            sections["has_structure"] = True

            # Extract sections based on form type
            if form_type in ["10-K", "10-Q"]:
                if hasattr(filing_obj, "business"):
                    sections["business"] = str(filing_obj.business)[:5000]
                if hasattr(filing_obj, "risk_factors"):
                    sections["risk_factors"] = str(filing_obj.risk_factors)[:5000]
                if hasattr(filing_obj, "mda"):
                    sections["mda"] = str(filing_obj.mda)[:5000]
                if hasattr(filing_obj, "financials"):
                    sections["has_financials"] = True
        except Exception as e:
            logger.warning(f"Could not get structured sections: {e}")

        return {"success": True, "sections": sections}
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_filing_sections failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"get_filing_sections failed: {type(e).__name__}: {e}"
        )


@app.post("/web_search")
async def web_search(req: SearchRequest) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    max_results: int = 1

    exa_api_key: Optional[str] = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        raise HTTPException(status_code=400, detail="EXA_API_KEY not set on environment")

    try:
        # Use exponential backoff for the API call
        data = await call_with_exponential_backoff(
            execute_search,
            req.query,
            exa_api_key,
            max_results,
        )

        for item in data.get("results", []):
            title = item.get("title", "")
            url = item.get("url", "")
            if title and url:
                results.append({"title": title, "url": url})

        if not results:
            autoprompt = data.get("autopromptString", req.query)
            return [
                {
                    "message": "No results found",
                    "query": req.query,
                    "autopromptString": autoprompt,
                }
            ]

    except (
        httpx.HTTPStatusError
    ) as e:  # pragma: no cover - network errors are environment dependent
        status_code = e.response.status_code
        if status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid EXA_API_KEY")
        if status_code == 429:
            # This should be handled by exponential backoff, but if all retries fail
            raise HTTPException(status_code=429, detail="Exa API rate limit exceeded after retries")
        raise HTTPException(status_code=502, detail=f"Exa API error: {status_code}")
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Search failed: {type(e).__name__}: {e}")

    state.search_count += 1
    return results


@app.post("/web_fetch")
async def web_fetch(req: FetchRequest) -> Dict[str, str]:
    from urllib.parse import urlparse

    max_length: int = 2500
    parsed = urlparse(req.url)
    if not parsed.scheme or not parsed.netloc:
        raise HTTPException(status_code=400, detail=f"Invalid URL: {req.url}")

    exa_api_key: Optional[str] = os.getenv("EXA_API_KEY")
    if not exa_api_key:
        raise HTTPException(status_code=400, detail="EXA_API_KEY not set on environment")

    try:
        # Use exponential backoff for the API call
        data = await call_with_exponential_backoff(
            execute_fetch,
            req.url,
            exa_api_key,
            max_length,
        )

        results = data.get("results", [])
        if results:
            result = results[0]
            text = result.get("text", "")
            summary = result.get("summary", "")
            highlights = result.get("highlights", [])

            parts: List[str] = []
            if summary:
                parts.append("=== SUMMARY (Main Takeaways) ===")
                parts.append(summary)
                parts.append("")
            if highlights:
                parts.append("=== KEY HIGHLIGHTS ===")
                for idx, hl in enumerate(highlights[:3], 1):
                    parts.append(f"\nHighlight {idx}:")
                    parts.append(str(hl))
                parts.append("")
            if text:
                parts.append("=== FULL CONTENT ===")
                if len(text) > max_length:
                    text = text[:max_length] + "...[truncated]"
                parts.append(text)

            content = "\n".join(parts) if parts else "No content available"
        else:
            content = "No content available for this URL"

    except httpx.HTTPStatusError as e:  # pragma: no cover
        status_code = e.response.status_code
        if status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid EXA_API_KEY")
        if status_code == 429:
            # This should be handled by exponential backoff, but if all retries fail
            raise HTTPException(status_code=429, detail="Exa API rate limit exceeded after retries")
        raise HTTPException(status_code=502, detail=f"Exa API error: {status_code}")
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Fetch failed: {type(e).__name__}: {e}")

    state.fetch_count += 1
    return {"content": content}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
