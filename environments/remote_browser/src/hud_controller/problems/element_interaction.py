"""Element interaction problems for testing UI components."""

from ..problems import problem


@problem("button_click_test", description="Test button clicking and verification")
class ButtonClickTestProblem:
    """Problem that tests clicking buttons and verifying state changes."""

    def get_setup(self):
        """Load a page with interactive elements."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Button Test</title>
            <style>
                button { padding: 10px 20px; margin: 10px; font-size: 16px; }
                #result { margin-top: 20px; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>Button Click Test</h1>
            <button id="test-btn" onclick="document.getElementById('result').innerText='Button clicked!'">
                Click Me
            </button>
            <div id="result"></div>
        </body>
        </html>
        """
        return {"name": "load_html_content", "arguments": {"html": html_content}}

    def get_evaluation(self):
        """Verify the button is present."""
        return {
            "name": "page_contains",
            "arguments": {
                "search_terms": ["Button Click Test", "Click Me"],
                "partial_rewarding": True,
            },
        }
