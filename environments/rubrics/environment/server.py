"""
FastAPI server for Rubrics environment with SEC EDGAR integration.
Manages SEC filing data access and state.
"""

import asyncio
import logging
import os
import socket
import traceback
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar
from urllib.parse import urlparse

import httpx
import uvicorn
from edgar import Company, Filing, set_identity, get_filings as edgar_get_filings
from edgar.financials import Financials
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rubric import Rubric

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
    ticker: str
    form_type: Optional[str] = None
    limit: int = 10


class GetFilingContentRequest(BaseModel):
    filing_url: str


class AnswerRequest(BaseModel):
    final_answer: str


class EvaluateRequest(BaseModel):
    rubric: list[dict[str, str | float]]


class RecentFilingsRequest(BaseModel):
    identifier: str | None = None  # ticker or CIK; if None, global recent
    form_type: str | None = None
    limit: int = 50
    days: int | None = None  # reserved, not used directly


class FilingByAccessionRequest(BaseModel):
    identifier: str  # ticker or CIK
    accession_number: str


app = FastAPI(title="SEC EDGAR Environment API", version="0.1.0")


# Require SEC EDGAR identity via EDGAR_IDENTITY (format: "Your Name your.email@domain.com")
set_identity(os.environ["EDGAR_IDENTITY"])


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "healthy"}


async def _is_port_open(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.15)
    try:
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except Exception:
        return False


@app.post("/setup")
async def setup() -> Dict[str, Any]:
    state.reset()
    return {"ok": True}


@app.post("/search_company")
async def search_company(req: SearchCompanyRequest) -> List[Dict[str, str]]:
    """Search for a company by ticker or name."""
    try:
        # Use edgartools to search for company
        company = Company(req.query)

        # edgartools Company has tickers (plural) not ticker
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
    """Get recent filings for a company."""
    try:
        company = Company(req.ticker)

        # Get filings (no limit parameter, apply limit after fetching)
        if req.form_type:
            filings = company.get_filings(form=req.form_type)
        else:
            filings = company.get_filings()

        results = []
        for i, filing in enumerate(filings):
            if i >= req.limit:
                break
            results.append(
                {
                    "filing_date": filing.filing_date.strftime("%Y-%m-%d")
                    if filing.filing_date
                    else "",
                    "form_type": filing.form,
                    "description": filing.primary_doc_description or "",
                    "filing_url": filing.filing_url,
                    "accession_number": filing.accession_number,
                }
            )

        state.search_count += 1
        return results

    except Exception as e:
        logger.error(f"Get filings failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Get filings failed: {type(e).__name__}: {e}")


@app.post("/get_filing_content")
async def get_filing_content(req: GetFilingContentRequest) -> Dict[str, str]:
    """Get the content of a specific filing."""
    try:
        # Parse the filing URL to extract accession number
        # URL format: https://www.sec.gov/Archives/edgar/data/{CIK}/{ACCESSION_NO_DASHES}/{filename}
        parsed = urlparse(req.filing_url)
        path_parts = parsed.path.split("/")

        # Find the accession number without dashes
        accession_no_dashes = None
        for part in path_parts:
            if len(part) >= 18 and part.isdigit():  # Accession numbers are 18 digits
                accession_no_dashes = part
                break

        if not accession_no_dashes:
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract accession number from URL: {req.filing_url}",
            )

        # Convert to accession format with dashes: 0001104659-25-042659
        accession = (
            f"{accession_no_dashes[:10]}-{accession_no_dashes[10:12]}-{accession_no_dashes[12:]}"
        )

        # Prefer locating via Company to satisfy Filing ctor requirements
        filing = None
        try:
            # Try to infer CIK from the URL path (segment after 'data')
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

        # Fallback: try direct Filing by accession for versions that support it
        if filing is None:
            try:
                filing = Filing(accession)
            except TypeError:
                filing = None

        if filing is None:
            raise HTTPException(
                status_code=404, detail=f"Filing not found for accession {accession}"
            )

        # Get the filing content with fallbacks
        content = ""
        try:
            # Prefer full text submission
            content = filing.text()
        except Exception:
            try:
                content = filing.text  # property in some versions
            except Exception:
                content = ""

        # Fallback to HTML
        if not content:
            try:
                content = filing.html()
            except Exception:
                try:
                    content = filing.html  # property fallback
                except Exception:
                    content = ""

        # Final fallback: fetch the URL directly
        if not content:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(
                        req.filing_url, headers={"User-Agent": os.environ["EDGAR_IDENTITY"]}
                    )
                    resp.raise_for_status()
                    content = resp.text
            except Exception:
                content = ""

        # Truncate if too long
        max_length = 50000
        if len(content) > max_length:
            content = content[:max_length] + "\n\n...[truncated]"

        state.fetch_count += 1
        return {"content": content}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get filing content failed: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Get filing content failed: {type(e).__name__}: {e}"
        )


@app.post("/get_recent_filings")
async def get_recent_filings(req: RecentFilingsRequest) -> List[Dict[str, Any]]:
    """Get recent filings for a company (by ticker/CIK) or globally.

    Args:
        identifier: Optional ticker or CIK. If omitted, returns global recent filings
        form_type: Optional form filter (e.g., "10-K", "8-K", ["3","4","5"])
        limit: Max number of results
    """
    try:
        results: list[dict[str, Any]] = []

        if req.identifier:
            company = Company(req.identifier)
            filings = (
                company.get_filings(form=req.form_type) if req.form_type else company.get_filings()
            )
        else:
            # Global feed via edgar.get_filings
            filings = edgar_get_filings(form=req.form_type, count=req.limit)

        for i, filing in enumerate(filings):
            if i >= req.limit:
                break
            results.append(
                {
                    "filing_date": filing.filing_date.strftime("%Y-%m-%d")
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

        return results
    except Exception as e:
        logger.error(f"get_recent_filings failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"get_recent_filings failed: {type(e).__name__}: {e}"
        )


@app.post("/get_filing_content_by_accession")
async def get_filing_content_by_accession(req: FilingByAccessionRequest) -> Dict[str, Any]:
    """Get filing content and structured info via identifier + accession."""
    try:
        company = Company(req.identifier)

        filing = None
        clean_req = req.accession_number.replace("-", "")
        for f in company.get_filings():
            if getattr(f, "accession_number", "").replace("-", "") == clean_req:
                filing = f
                break

        if filing is None:
            raise HTTPException(
                status_code=404,
                detail=f"Filing {req.accession_number} not found for {req.identifier}",
            )

        # Content
        content = ""
        try:
            content = filing.text()
        except Exception:
            try:
                content = filing.text
            except Exception:
                content = ""

        if not content:
            try:
                content = filing.html()
            except Exception:
                try:
                    content = filing.html
                except Exception:
                    content = ""

        # Truncate
        max_length = 50000
        if len(content) > max_length:
            content = content[:max_length] + "\n\n...[truncated]"

        # Optional: minimal structured hints
        filing_data: dict[str, Any] = {}
        try:
            obj = filing.obj()
            if obj:
                if filing.form == "8-K" and hasattr(obj, "items"):
                    filing_data["items"] = getattr(obj, "items", [])
                    filing_data["has_press_release"] = getattr(obj, "has_press_release", False)
                elif filing.form in ["10-K", "10-Q"]:
                    filing_data["has_financials"] = True
        except Exception:
            pass

        return {
            "accession_number": filing.accession_number,
            "form_type": filing.form,
            "filing_date": filing.filing_date.isoformat()
            if hasattr(filing.filing_date, "isoformat")
            else str(filing.filing_date),
            "content": content,
            "url": getattr(filing, "url", getattr(filing, "filing_url", None)),
            "filing_data": filing_data,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_filing_content_by_accession failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"get_filing_content_by_accession failed: {type(e).__name__}: {e}",
        )


@app.post("/analyze_8k")
async def analyze_8k(req: FilingByAccessionRequest) -> Dict[str, Any]:
    """Analyze an 8-K filing for specific events and items."""
    try:
        company = Company(req.identifier)
        filing = None
        for f in company.get_filings(form="8-K"):
            if f.accession_number.replace("-", "") == req.accession_number.replace("-", ""):
                filing = f
                break

        if filing is None:
            raise HTTPException(
                status_code=404,
                detail=f"8-K filing {req.accession_number} not found for {req.identifier}",
            )

        # Try to get structured 8-K object
        analysis = {
            "accession_number": filing.accession_number,
            "form_type": filing.form,
            "filing_date": filing.filing_date.isoformat()
            if hasattr(filing.filing_date, "isoformat")
            else str(filing.filing_date),
            "has_structure": False,
        }

        try:
            eightk = filing.obj()
            analysis["has_structure"] = True
            analysis["items"] = getattr(eightk, "items", [])
            analysis["has_press_release"] = getattr(eightk, "has_press_release", False)
        except Exception:
            pass

        return {"success": True, "analysis": analysis}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"analyze_8k failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"analyze_8k failed: {type(e).__name__}: {e}")


@app.post("/get_filing_sections")
async def get_filing_sections(req: FilingByAccessionRequest) -> Dict[str, Any]:
    """Get specific sections from a 10-K or 10-Q filing."""
    try:
        company = Company(req.identifier)
        filing = None
        form_type = None

        # Try to find filing
        for f in company.get_filings():
            if f.accession_number.replace("-", "") == req.accession_number.replace("-", ""):
                filing = f
                form_type = f.form
                break

        if filing is None:
            raise HTTPException(
                status_code=404,
                detail=f"Filing {req.accession_number} not found for {req.identifier}",
            )

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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"get_filing_sections failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"get_filing_sections failed: {type(e).__name__}: {e}"
        )


@app.post("/get_financials")
async def get_financials(req: FilingByAccessionRequest) -> Dict[str, Any]:
    """Extract financial statements and key metrics from a 10-K or 10-Q filing."""
    try:
        company = Company(req.identifier)
        filing = None

        # Try to find filing
        for f in company.get_filings():
            if f.accession_number.replace("-", "") == req.accession_number.replace("-", ""):
                filing = f
                break

        if filing is None:
            raise HTTPException(
                status_code=404,
                detail=f"Filing {req.accession_number} not found for {req.identifier}",
            )

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

                # Extract income statement
                try:
                    income = financials.income_statement()
                    if income is not None:
                        financial_data["income_statement"] = {
                            "data": income.to_dict(orient="index")
                            if hasattr(income, "to_dict")
                            else str(income)[:5000],
                            "columns": list(income.columns) if hasattr(income, "columns") else None,
                        }
                except Exception as e:
                    logger.warning(f"Could not extract income statement: {e}")

                # Extract balance sheet
                try:
                    balance = financials.balance_sheet()
                    if balance is not None:
                        financial_data["balance_sheet"] = {
                            "data": balance.to_dict(orient="index")
                            if hasattr(balance, "to_dict")
                            else str(balance)[:5000],
                            "columns": list(balance.columns)
                            if hasattr(balance, "columns")
                            else None,
                        }
                except Exception as e:
                    logger.warning(f"Could not extract balance sheet: {e}")

                # Extract cash flow
                try:
                    cashflow = financials.cashflow_statement()
                    if cashflow is not None:
                        financial_data["cash_flow"] = {
                            "data": cashflow.to_dict(orient="index")
                            if hasattr(cashflow, "to_dict")
                            else str(cashflow)[:5000],
                            "columns": list(cashflow.columns)
                            if hasattr(cashflow, "columns")
                            else None,
                        }
                except Exception as e:
                    logger.warning(f"Could not extract cash flow: {e}")

                result["financial_data"] = financial_data
        except Exception as e:
            logger.warning(f"Could not extract financials: {e}")

        return {"success": True, **result}
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
        company = Company(req.identifier)
        filing = None

        # Try to find filing
        for f in company.get_filings():
            if f.accession_number.replace("-", "") == req.accession_number.replace("-", ""):
                filing = f
                break

        if filing is None:
            raise HTTPException(
                status_code=404,
                detail=f"Filing {req.accession_number} not found for {req.identifier}",
            )

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

            # Try to extract segment data
            if hasattr(filing_obj, "segments"):
                result["has_segment_data"] = True
                result["segment_data"] = str(filing_obj.segments)[:10000]
            elif hasattr(filing_obj, "notes") and hasattr(filing_obj.notes, "segments"):
                result["has_segment_data"] = True
                result["segment_data"] = str(filing_obj.notes.segments)[:10000]
        except Exception as e:
            logger.warning(f"Could not extract segment data: {e}")

        return result
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
        return {"reward": reward, "info": info, "done": True}
    except Exception as e:
        logger.error(f"Rubric evaluation failed: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        return {
            "reward": 0.0,
            "content": f"Evaluation failed: {type(e).__name__}: {e}",
            "done": True,
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)
