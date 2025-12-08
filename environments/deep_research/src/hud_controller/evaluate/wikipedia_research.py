"""Wikipedia-specific evaluation functions for research tasks."""

import logging
from typing import Dict, Any, List
from ..evaluate import evaluator

logger = logging.getLogger(__name__)


@evaluator("wikipedia_article_sections", description="Check if Wikipedia article has expected sections")
def wikipedia_article_sections(
    context: Dict[str, Any], 
    expected_sections: List[str], 
    partial_match: bool = True
) -> Dict[str, Any]:
    """Check if a Wikipedia article contains expected sections.
    
    Args:
        context: Browser context containing page content
        expected_sections: List of section names to look for
        partial_match: If True, only some sections need to be found
        
    Returns:
        Evaluation result with success status and details
    """
    try:
        page_content = context.get("page_content", "").lower()
        
        found_sections = []
        for section in expected_sections:
            if section.lower() in page_content:
                found_sections.append(section)
        
        if partial_match:
            success = len(found_sections) > 0
        else:
            success = len(found_sections) == len(expected_sections)
            
        return {
            "success": success,
            "score": len(found_sections) / len(expected_sections) if expected_sections else 0,
            "details": {
                "expected_sections": expected_sections,
                "found_sections": found_sections,
                "missing_sections": [s for s in expected_sections if s not in found_sections]
            }
        }
    except Exception as e:
        logger.error(f"Error checking Wikipedia sections: {e}")
        return {"success": False, "score": 0, "error": str(e)}


@evaluator("wikipedia_infobox_present", description="Check if Wikipedia article has an infobox")
def wikipedia_infobox_present(context: Dict[str, Any]) -> Dict[str, Any]:
    """Check if a Wikipedia article contains an infobox.
    
    Args:
        context: Browser context containing page content
        
    Returns:
        Evaluation result with success status
    """
    try:
        page_content = context.get("page_content", "").lower()
        
        # Look for common infobox indicators
        infobox_indicators = [
            "infobox", 
            "class=\"infobox\"",
            "table class=\"infobox",
            "wikitable infobox"
        ]
        
        has_infobox = any(indicator in page_content for indicator in infobox_indicators)
        
        return {
            "success": has_infobox,
            "score": 1.0 if has_infobox else 0.0,
            "details": {"infobox_found": has_infobox}
        }
    except Exception as e:
        logger.error(f"Error checking for infobox: {e}")
        return {"success": False, "score": 0, "error": str(e)}


@evaluator("wikipedia_references_count", description="Count references in a Wikipedia article")
def wikipedia_references_count(
    context: Dict[str, Any], 
    min_references: int = 5
) -> Dict[str, Any]:
    """Count the number of references in a Wikipedia article.
    
    Args:
        context: Browser context containing page content
        min_references: Minimum number of references expected
        
    Returns:
        Evaluation result with reference count
    """
    try:
        page_content = context.get("page_content", "").lower()
        
        # Count various reference indicators
        ref_patterns = [
            "<ref>", 
            "<ref ", 
            "[1]", "[2]", "[3]", "[4]", "[5]",
            "cite web", "cite book", "cite journal"
        ]
        
        total_refs = sum(page_content.count(pattern) for pattern in ref_patterns)
        
        success = total_refs >= min_references
        
        return {
            "success": success,
            "score": min(total_refs / min_references, 1.0) if min_references > 0 else 1.0,
            "details": {
                "reference_count": total_refs,
                "min_required": min_references
            }
        }
    except Exception as e:
        logger.error(f"Error counting references: {e}")
        return {"success": False, "score": 0, "error": str(e)}


@evaluator("wikipedia_external_links", description="Check for external links section")
def wikipedia_external_links(context: Dict[str, Any]) -> Dict[str, Any]:
    """Check if Wikipedia article has an external links section.
    
    Args:
        context: Browser context containing page content
        
    Returns:
        Evaluation result with external links status
    """
    try:
        page_content = context.get("page_content", "").lower()
        
        external_link_indicators = [
            "external links",
            "further reading",
            "see also",
            "official website"
        ]
        
        has_external_links = any(indicator in page_content for indicator in external_link_indicators)
        
        return {
            "success": has_external_links,
            "score": 1.0 if has_external_links else 0.0,
            "details": {"external_links_found": has_external_links}
        }
    except Exception as e:
        logger.error(f"Error checking external links: {e}")
        return {"success": False, "score": 0, "error": str(e)}