"""Todo app setup tools."""

import logging
from typing import Dict, Any, List
from hud.tools import BaseSetup, SetupResult
from ..setup import setup

logger = logging.getLogger(__name__)


@setup("todo_seed", description="Seed database with default test todos")
class TodoSeedSetup(BaseSetup):
    """Setup tool that seeds the database with default test data."""

    async def __call__(self, context, num_items: int = 5) -> SetupResult:
        """Seed the database with test todo items.

        Args:
            num_items: Number of test items to create (default: 5)

        Returns:
            Setup result with seeded items info
        """
        try:
            # Call the app's seed API
            result = await context.call_app_api("todo", "/api/eval/seed", method="POST")

            return SetupResult(
                status="success",
                message=f"Seeded database with test data",
                items_added=result.get("items_added", num_items),
            )
        except Exception as e:
            logger.error(f"TodoSeedSetup failed: {e}")
            return SetupResult(
                status="error",
                message=f"Failed to seed database: {str(e)}",
            )


@setup("todo_reset", description="Reset database to empty state")
class TodoResetSetup(BaseSetup):
    """Setup tool that resets the database to empty state."""

    async def __call__(self, context) -> SetupResult:
        """Reset the database to empty state.

        Returns:
            Setup result with reset confirmation
        """
        try:
            # Call the app's reset API
            result = await context.call_app_api("todo", "/api/eval/reset", method="DELETE")

            return SetupResult(
                status="success",
                message="Database reset to empty state",
            )
        except Exception as e:
            logger.error(f"TodoResetSetup failed: {e}")
            return SetupResult(
                status="error",
                message=f"Failed to reset database: {str(e)}",
            )


@setup("todo_custom_seed", description="Seed database with custom todo items")
class TodoCustomSeedSetup(BaseSetup):
    """Setup tool that seeds the database with custom todo items."""

    async def __call__(self, context, items: List[Dict[str, Any]]) -> SetupResult:
        """Seed the database with custom todo items.

        Args:
            items: List of todo items to create

        Returns:
            Setup result with seeded items info
        """
        try:
            # Ensure all items have required fields
            formatted_items = []
            for item in items:
                formatted_item = {
                    "title": item.get("title", ""),
                    "description": item.get(
                        "description", ""
                    ),  # Add empty description if not provided
                    "completed": item.get("completed", False),
                }
                formatted_items.append(formatted_item)

            # Call the app's custom seed API (send list directly, not wrapped in dict)
            result = await context.call_app_api(
                "todo", "/api/eval/seed_custom", method="POST", json=formatted_items
            )

            return SetupResult(
                status="success",
                message=f"Seeded database with {len(items)} custom items",
                items_added=result.get("items_added", len(items)),
            )
        except Exception as e:
            logger.error(f"TodoCustomSeedSetup failed: {e}")
            return SetupResult(
                status="error",
                message=f"Failed to seed custom items: {str(e)}",
            )


@setup("todo_navigate", description="Navigate to the Todo app")
class TodoNavigateSetup(BaseSetup):
    """Setup tool that navigates to the Todo app."""

    async def __call__(self, context, url: str = None) -> SetupResult:
        """Navigate to the Todo app.

        Args:
            url: Optional custom URL to navigate to

        Returns:
            Setup result with navigation confirmation
        """
        try:
            # Get the default URL if not provided
            if not url:
                url = context.get_app_url("todo")

            # Use Playwright to navigate
            if context.playwright:
                nav_result = await context.playwright.navigate(url)

                return SetupResult(
                    status="success",
                    message=f"Navigated to Todo app at {url}",
                    url=url,
                )
            else:
                return SetupResult(
                    status="error",
                    message="Playwright tool not available for navigation",
                )
        except Exception as e:
            logger.error(f"TodoNavigateSetup failed: {e}")
            return SetupResult(
                status="error",
                message=f"Failed to navigate to Todo app: {str(e)}",
            )
