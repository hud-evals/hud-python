"""Search engine interaction problems."""

from ..problems import problem


@problem("google_search", description="Perform a Google search and verify results")
class GoogleSearchProblem:
    """Problem that performs a search and verifies results appear."""

    def get_setup(self):
        """Navigate to Google."""
        return {"name": "navigate_to_url", "arguments": {"url": "https://www.google.com"}}

    def get_evaluation(self):
        """Verify Google search page loaded."""
        return {
            "name": "page_contains",
            "arguments": {"search_terms": ["Google", "Search"], "partial_rewarding": True},
        }
