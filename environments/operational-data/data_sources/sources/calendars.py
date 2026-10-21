"""Calendars, rebuilt from iCalendar exports (Google Calendar, Outlook, Apple Calendar)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime, time, timedelta, tzinfo
from pathlib import Path
from typing import Annotated, Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import icalendar
import recurring_ical_events
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

from data_sources.base import (
    DataSource,
    ExportError,
    Limit,
    Page,
    TaskContext,
    contains_all,
    counted,
    paginate,
    search_terms,
    tool,
)

People = Annotated[list[str], Field(description="Email addresses or names")]


class Attendee(BaseModel):
    name: str | None = None
    email: str
    response: str


class CalendarEvent(BaseModel):
    id: str
    calendar: str
    title: str
    start: datetime | date
    end: datetime | date
    all_day: bool
    location: str | None = None
    organizer: str | None = None
    attendees: list[Attendee] = []
    description: str | None = None
    recurring: bool = False
    busy: bool = True


class CalendarInfo(BaseModel):
    name: str
    timezone: str
    events: int


class FreeSlot(BaseModel):
    start: datetime
    end: datetime
    minutes: int


class CreatedEvent(BaseModel):
    event: CalendarEvent
    conflicts: list[str]


class Calendar(DataSource):
    id = "calendar"
    title = "Calendar"
    summary = "Calendar export (.ics)."

    def __init__(self, context: TaskContext) -> None:
        super().__init__(context)
        self._calendars: list[tuple[str, icalendar.Calendar]] = []
        self._owners: dict[str, list[str]] = {}
        self._recurring: set[str] = set()
        self._created: list[CalendarEvent] = []
        self._tz: tzinfo = UTC

    @classmethod
    def load(cls, files: Sequence[Path], context: TaskContext) -> Self:
        source = cls(context)
        zones: list[str] = []
        for path in files:
            if path.suffix.lower() not in {".ics", ".ical", ".ifb"}:
                raise ExportError(f"{path.name}: expected an iCalendar (.ics) export")
            try:
                calendar = icalendar.Calendar.from_ical(path.read_bytes())
            except ValueError as error:
                raise ExportError(f"{path.name}: not a valid iCalendar file ({error})") from error
            calendar.subcomponents = [
                component
                for component in calendar.subcomponents
                if component.name != "VEVENT" or "DTSTART" in component
            ]
            name = str(calendar.get("X-WR-CALNAME") or path.stem)
            source._calendars.append((name, calendar))
            if zone := calendar.get("X-WR-TIMEZONE"):
                zones.append(str(zone))
            for event in calendar.events:
                if event.get("RRULE") or event.get("RDATE"):
                    source._recurring.add(str(event.get("UID")))
        for zone in zones:
            try:
                source._tz = ZoneInfo(zone)
                break
            except (ZoneInfoNotFoundError, ValueError):
                continue
        source._owners = source._calendar_owners()
        return source

    def _calendar_owners(self) -> dict[str, list[str]]:
        """Each calendar's names for its owner: the calendar name, plus the person's name
        when the calendar is named by an address that appears on invitations."""
        names: dict[str, str] = {}
        for _, calendar in self._calendars:
            for component in calendar.events:
                for person in [component.get("ORGANIZER"), *component.attendees]:
                    if person is not None and person.params.get("CN"):
                        names.setdefault(_address(person).lower(), str(person.params["CN"]))
        return {
            name: [name.lower(), *([names[name.lower()].lower()] if name.lower() in names else [])]
            for name, _ in self._calendars
        }

    def describe(self) -> str:
        events = sum(len(calendar.events) for _, calendar in self._calendars)
        return f"{counted(len(self._calendars), 'calendar')}, {counted(events, 'event')}"

    def _local(self, moment: datetime) -> datetime:
        return moment.astimezone(self._tz) if moment.tzinfo else moment.replace(tzinfo=self._tz)

    def _bounds(self, value: datetime | date) -> datetime:
        if isinstance(value, datetime):
            return self._local(value)
        return datetime.combine(value, time(), self._tz)

    def _events(self, start: datetime, end: datetime) -> Iterator[CalendarEvent]:
        for name, calendar in self._calendars:
            for component in recurring_ical_events.of(calendar).between(start, end):
                if isinstance(component, icalendar.Event):
                    yield self._event(name, component)
        for event in self._created:
            if self._bounds(event.start) < end and self._bounds(event.end) > start:
                yield event

    def _event(self, calendar: str, component: icalendar.Event) -> CalendarEvent:
        uid = str(component.get("UID", ""))
        start: datetime | date = component.start
        end: datetime | date = component.end
        all_day = not isinstance(start, datetime)
        if isinstance(start, datetime):
            start = self._local(start)
        if isinstance(end, datetime):
            end = self._local(end)
        recurring = uid in self._recurring
        organizer = component.get("ORGANIZER")
        return CalendarEvent(
            id=f"{uid}@{start:%Y%m%dT%H%M}" if recurring else uid,
            calendar=calendar,
            title=str(component.get("SUMMARY", "(untitled)")),
            start=start,
            end=end,
            all_day=all_day,
            location=str(component["LOCATION"]) if component.get("LOCATION") else None,
            organizer=_person(organizer) if organizer else None,
            attendees=[
                Attendee(
                    name=str(attendee.params.get("CN")) if attendee.params.get("CN") else None,
                    email=_address(attendee),
                    response=str(attendee.params.get("PARTSTAT", "NEEDS-ACTION")).lower(),
                )
                for attendee in component.attendees
            ],
            description=str(component["DESCRIPTION"]) if component.get("DESCRIPTION") else None,
            recurring=recurring,
            busy=str(component.get("TRANSP", "OPAQUE")).upper() != "TRANSPARENT"
            and str(component.get("STATUS", "")).upper() != "CANCELLED",
        )

    def _involves(self, event: CalendarEvent, person: str) -> bool:
        """Whether ``person`` is expected at ``event``.

        Their own response decides when they are invited, since a declined invitation
        stays on their calendar; otherwise owning the calendar or organizing counts.
        """
        wanted = person.lower()
        responses = [
            attendee.response
            for attendee in event.attendees
            if wanted in attendee.email.lower() or wanted in (attendee.name or "").lower()
        ]
        if responses:
            return any(response != "declined" for response in responses)
        owners = self._owners.get(event.calendar, [event.calendar.lower()])
        if any(wanted in owner for owner in owners):
            return True
        return bool(event.organizer and wanted in event.organizer.lower())

    def _window(self, start: date, end: date) -> tuple[datetime, datetime]:
        if end < start:
            raise ToolError("end is before start")
        return datetime.combine(start, time(), self._tz), datetime.combine(
            end + timedelta(days=1), time(), self._tz
        )

    @tool
    def list_calendars(self) -> list[CalendarInfo]:
        """List the calendars that were loaded and the timezone times are shown in."""
        return [
            CalendarInfo(name=name, timezone=str(self._tz), events=len(calendar.events))
            for name, calendar in self._calendars
        ]

    @tool
    def list_events(
        self,
        start: Annotated[date, Field(description="First day to include")],
        end: Annotated[date, Field(description="Last day to include")],
        calendar: Annotated[str | None, Field(description="Only this calendar")] = None,
        attendee: Annotated[
            str | None, Field(description="Only events this person organizes or attends")
        ] = None,
        query: Annotated[str | None, Field(description="Words in the title or description")] = None,
        limit: Limit = 100,
    ) -> Page[CalendarEvent]:
        """List events between two days, recurring meetings expanded, in time order."""
        window_start, window_end = self._window(start, end)
        terms = search_terms(query)
        events = [
            event
            for event in self._events(window_start, window_end)
            if (calendar is None or calendar.lower() in event.calendar.lower())
            and (attendee is None or self._involves(event, attendee))
            and contains_all(terms, event.title, event.description, event.location)
        ]
        events.sort(key=lambda event: self._bounds(event.start))
        return paginate(events, limit)

    @tool
    def find_free_time(
        self,
        attendees: People,
        start: Annotated[date, Field(description="First day to search")],
        end: Annotated[date, Field(description="Last day to search")],
        duration_minutes: Annotated[int, Field(ge=5, le=480)] = 30,
        day_starts_at: Annotated[int, Field(ge=0, le=23, description="Working day start hour")] = 9,
        day_ends_at: Annotated[int, Field(ge=1, le=24, description="Working day end hour")] = 17,
        include_weekends: bool = False,
        limit: Limit = 20,
    ) -> list[FreeSlot]:
        """Find times when every attendee is free, within working hours.

        A person is busy during events on their own calendar, events they
        organize, and events they were invited to and did not decline.
        """
        if day_ends_at <= day_starts_at:
            raise ToolError("day_ends_at must be after day_starts_at")
        window_start, window_end = self._window(start, end)
        events = [event for event in self._events(window_start, window_end) if event.busy]
        busy: list[tuple[datetime, datetime]] = []
        for person in attendees:
            mine = [event for event in events if self._involves(event, person)]
            if not mine and not self._known(person):
                raise ToolError(f"No calendar or invitations mention {person!r}")
            busy.extend((self._bounds(event.start), self._bounds(event.end)) for event in mine)
        busy.sort()
        slots: list[FreeSlot] = []
        day = start
        while day <= end and len(slots) < limit:
            if include_weekends or day.weekday() < 5:
                cursor = datetime.combine(day, time(day_starts_at), self._tz)
                closing = datetime.combine(day, time(), self._tz) + timedelta(hours=day_ends_at)
                for busy_start, busy_end in busy:
                    if busy_end <= cursor or busy_start >= closing:
                        continue
                    if busy_start - cursor >= timedelta(minutes=duration_minutes):
                        slots.append(_slot(cursor, busy_start))
                    cursor = max(cursor, busy_end)
                if closing - cursor >= timedelta(minutes=duration_minutes):
                    slots.append(_slot(cursor, closing))
            day += timedelta(days=1)
        return slots[:limit]

    def _known(self, person: str) -> bool:
        wanted = person.lower()
        for name, calendar in self._calendars:
            if any(wanted in owner for owner in self._owners.get(name, [])):
                return True
            for component in calendar.events:
                people = [component.get("ORGANIZER"), *component.attendees]
                if any(p is not None and wanted in _person(p).lower() for p in people):
                    return True
        return False

    @tool
    def create_event(
        self,
        title: str,
        start: datetime,
        end: datetime,
        attendees: People | None = None,
        description: str | None = None,
        location: str | None = None,
    ) -> CreatedEvent:
        """Create an event and invite attendees; reports attendees who are already busy then."""
        start, end = self._local(start), self._local(end)
        if end <= start:
            raise ToolError("end must be after start")
        invited = attendees or []
        clashes = [
            f"{person} has {event.title!r}"
            for person in invited
            for event in self._events(start, end)
            if event.busy and self._involves(event, person)
        ]
        event = CalendarEvent(
            id=f"created-{len(self._created) + 1}",
            calendar=self._calendars[0][0] if self._calendars else "Calendar",
            title=title,
            start=start,
            end=end,
            all_day=False,
            location=location,
            organizer="assistant",
            attendees=[Attendee(email=person, response="needs-action") for person in invited],
            description=description,
        )
        self._created.append(event)
        outcome = f"created {title!r} {start:%Y-%m-%d %H:%M}-{end:%H:%M} {start.tzname()}"
        self.record(
            "create_event",
            {
                "title": title,
                "start": start,
                "end": end,
                "attendees": invited,
                "description": description,
            },
            outcome + (f"; conflicts: {'; '.join(clashes)}" if clashes else ""),
        )
        return CreatedEvent(event=event, conflicts=clashes)


def _address(value: icalendar.vCalAddress) -> str:
    return str(value).removeprefix("mailto:").removeprefix("MAILTO:")


def _person(value: icalendar.vCalAddress) -> str:
    name = value.params.get("CN")
    return f"{name} <{_address(value)}>" if name else _address(value)


def _slot(start: datetime, end: datetime) -> FreeSlot:
    return FreeSlot(start=start, end=end, minutes=int((end - start).total_seconds() // 60))
