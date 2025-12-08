"""Wikipedia search and research interaction problems."""

from ..problems import problem


@problem("wikipedia_search", description="Search Wikipedia for a topic and navigate to article")
class WikipediaSearchProblem:
    """Problem that searches Wikipedia for a topic and navigates to an article."""

    def get_setup(self):
        """Navigate to Wikipedia."""
        return {"name": "navigate_to_url", "arguments": {"url": "https://en.wikipedia.org"}}

    def get_evaluation(self):
        """Verify Wikipedia main page loaded."""
        return {
            "name": "page_contains",
            "arguments": {"search_terms": ["Wikipedia", "encyclopedia"], "partial_rewarding": True},
        }


@problem("wikipedia_article_research", description="Research a Wikipedia article and extract key information")
class WikipediaArticleResearchProblem:
    """Problem that navigates to a specific Wikipedia article and verifies content."""

    def get_setup(self):
        """Navigate to a sample Wikipedia article."""
        return {"name": "navigate_to_url", "arguments": {"url": "https://en.wikipedia.org/wiki/Python_(programming_language)"}}

    def get_evaluation(self):
        """Verify article content loaded."""
        return {
            "name": "page_contains",
            "arguments": {"search_terms": ["Python", "programming", "language"], "partial_rewarding": True},
        }


@problem("wikipedia_category_exploration", description="Navigate Wikipedia categories and explore related articles")
class WikipediaCategoryExplorationProblem:
    """Problem that explores Wikipedia categories and related articles."""

    def get_setup(self):
        """Navigate to a Wikipedia category page."""
        return {"name": "navigate_to_url", "arguments": {"url": "https://en.wikipedia.org/wiki/Category:Computer_science"}}

    def get_evaluation(self):
        """Verify category page loaded."""
        return {
            "name": "page_contains",
            "arguments": {"search_terms": ["Computer science", "Category"], "partial_rewarding": True},
        }


@problem("wikipedia_random_article", description="Use Wikipedia's random article feature for discovery")
class WikipediaRandomArticleProblem:
    """Problem that uses Wikipedia's random article feature."""

    def get_setup(self):
        """Navigate to Wikipedia random article."""
        return {"name": "navigate_to_url", "arguments": {"url": "https://en.wikipedia.org/wiki/Special:Random"}}

    def get_evaluation(self):
        """Verify a Wikipedia article loaded (any article is valid)."""
        return {
            "name": "page_contains",
            "arguments": {"search_terms": ["Wikipedia"], "partial_rewarding": True},
        }
