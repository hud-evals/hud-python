"""Every data source the environment serves, keyed by the id its template arguments declare."""

from data_sources.base import DataSource
from data_sources.sources.calendars import Calendar
from data_sources.sources.jira import Jira
from data_sources.sources.slack import Slack

SOURCES: dict[str, type[DataSource]] = {source.id: source for source in (Slack, Jira, Calendar)}
