//! Items related to the `Pattern` implementation for slices.

use crate::{span::Span, Event, Pattern};

/// An iterator produced by a query to a rendered slice of events.
pub struct SliceEvents<'a, T> {
    span: Span,
    /// Thes tart of the known, consistently intersecting range.
    consistently_intersecting_start: usize,
    /// Enumerated events, excluding non-intersecting tail events.
    events: std::iter::Enumerate<std::slice::Iter<'a, Event<T>>>,
}

impl<'a, T> Pattern for &'a [Event<T>] {
    type Value = &'a T;
    type Events = SliceEvents<'a, T>;
    /// Assumes all events are ordered (at least by their spans).
    fn query(&self, span: Span) -> Self::Events {
        let range = range_consistently_intersecting(self, span);
        let consistently_intersecting_start = range.start;
        let events = self[..range.end].iter().enumerate();
        SliceEvents {
            span,
            consistently_intersecting_start,
            events,
        }
    }
}

impl<'a, T> Iterator for SliceEvents<'a, T> {
    type Item = Event<&'a T>;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some((ix, ev)) = self.events.next() {
            // If we're still before the region of known intersecting
            // evs, we know the event start precedes the span start so
            // just check the end.
            let mut active = ev.span.whole_or_active();
            if ix < self.consistently_intersecting_start {
                if self.span.start < active.end {
                    active.start = self.span.start;
                    return Some(Event::new(&ev.value, active, ev.span.whole));
                }
            // Otherwise, assume the `start` is already in range to
            // reduce expensive `Rational` comparisons.
            } else {
                active.end = std::cmp::min(active.end, self.span.end);
                return Some(Event::new(&ev.value, active, ev.span.whole));
            };
        }
        None
    }
}

/// Retain only those events that intersect the given span.
///
/// Assumes the given `events` are ordered by their `span`.
///
/// Uses a binary search to find the start and end bounds of the known
/// intersecting events to avoid iterating through the entire slice.
///
/// The aim is to achieve O(log n) complexity as opposed to O(n).
pub fn retain_intersecting<T>(events: &mut Vec<Event<T>>, span: Span) {
    let range = range_consistently_intersecting(events, span);
    // Remove all events that start at or after the end of our
    // span as we know they do not intersect.
    events.truncate(range.end);
    // Separate and keep all remaining events starting after the
    // start of `new_span`, as we know they must intersect.
    let mut end = events.split_off(range.start);
    // The remaining start before our new span. Only keep them if
    // their `end`s are greater than our span's `start`.
    events.retain(|ev| span.start < ev.span.whole_or_active().end);
    // Re-append our known intersecting events.
    events.append(&mut end);
}

/// Find the range of events that are known to consistently intersect
/// the given span. This function is used to optimise `SliceEvents`
/// iteration and the `retain_intersecting` implementation.
///
/// Assumes the given `events` are ordered by their `span`.
///
/// - The range `start` is the first event that starts at or after the
///   given span `start`. Note that any of the preceding events may
///   also intersect as their span `end`s may succeed the given span
///   start.
/// - The range `end` is the first event that starts at or after the
///   given span `end`. All events following this index are known to
///   not intersect.
pub fn range_consistently_intersecting<T>(
    events: &[Event<T>],
    span: Span,
) -> std::ops::Range<usize> {
    let first_event_with_start_gte_span_start = events
        .binary_search_by(|ev| {
            if ev.span.whole_or_active().start < span.start {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        })
        .unwrap_or_else(|ix| ix);
    let first_event_with_start_gte_span_end = events
        .binary_search_by(|ev| {
            if ev.span.whole_or_active().start < span.end {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        })
        .unwrap_or_else(|ix| ix);
    first_event_with_start_gte_span_start..first_event_with_start_gte_span_end
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::span;

    #[test]
    fn test_empty() {
        let events: [Event<()>; 0] = [];
        let pattern = &events[..];
        let mut result = pattern.query(Span::new(0.into(), 1.into()));
        assert_eq!(result.next(), None);
    }

    #[test]
    fn test_single_event() {
        let span = span!(0 / 1, 1 / 1);
        let events = [Event::new((), span, None)];
        let pattern = &events[..];
        let mut result = pattern.query(span);
        assert_eq!(result.next().unwrap().value, &());
        assert_eq!(result.next(), None);
    }

    #[test]
    fn test_multiple_events() {
        let events = [
            Event::new(0, span!(0 / 1, 1 / 2), None),
            Event::new(1, span!(1 / 2, 1 / 1), None),
        ];
        let pattern = &events[..];
        let mut result = pattern.query(span!(0 / 1, 1 / 1));
        assert_eq!(result.next().unwrap().value, &0);
        assert_eq!(result.next().unwrap().value, &1);
        assert_eq!(result.next(), None);
    }

    #[test]
    fn test_query_span_outside_events() {
        let events = [
            Event::new(0, span!(0 / 1, 1 / 2), None),
            Event::new(1, span!(1 / 2, 1 / 1), None),
        ];
        let pattern = &events[..];
        let mut result = pattern.query(span!(2 / 1, 3 / 1));
        assert_eq!(result.next(), None);
    }

    #[test]
    fn test_query_span_partial_overlap() {
        let events = [
            Event::new(0, span!(0 / 1, 1 / 2), None),
            Event::new(1, span!(1 / 2, 1 / 1), None),
            Event::new(2, span!(1 / 1, 3 / 2), None),
        ];
        let pattern = &events[..];
        let mut result = pattern.query(span!(1 / 4, 5 / 4));
        assert_eq!(result.next().unwrap().value, &0);
        assert_eq!(result.next().unwrap().value, &1);
        assert_eq!(result.next().unwrap().value, &2);
        assert_eq!(result.next(), None);
    }

    #[test]
    fn test_active_span_within_query() {
        let events = [
            Event::new(0, span!(0 / 1, 1 / 2), None),
            Event::new(1, span!(1 / 2, 1 / 1), None),
        ];
        let pattern = &events[..];
        let mut evs = pattern.query(span!(1 / 4, 3 / 4));
        let ev = evs.next().unwrap();
        assert_eq!(ev.value, &0);
        assert_eq!(ev.span.active, span!(1 / 4, 1 / 2));
        let ev = evs.next().unwrap();
        assert_eq!(ev.value, &1);
        assert_eq!(ev.span.active, span!(1 / 2, 3 / 4));
        assert!(evs.next().is_none());
    }

    #[test]
    fn test_active_span_partial_overlap_start() {
        let events = [
            Event::new(0, span!(0 / 1, 1 / 2), None),
            Event::new(1, span!(1 / 2, 1 / 1), None),
        ];
        let pattern = &events[..];
        let mut evs = pattern.query(span!(1 / 8, 3 / 4));
        let ev = evs.next().unwrap();
        assert_eq!(ev.value, &0);
        assert_eq!(ev.span.active, span!(1 / 8, 1 / 2));
        let ev = evs.next().unwrap();
        assert_eq!(ev.value, &1);
        assert_eq!(ev.span.active, span!(1 / 2, 3 / 4));
        assert!(evs.next().is_none());
    }

    #[test]
    fn test_active_span_partial_overlap_end() {
        let events = [
            Event::new(0, span!(0 / 1, 1 / 2), None),
            Event::new(1, span!(1 / 2, 1 / 1), None),
            Event::new(2, span!(1 / 1, 3 / 2), None),
        ];
        let pattern = &events[..];
        let mut evs = pattern.query(span!(1 / 4, 5 / 4));
        let ev = evs.next().unwrap();
        assert_eq!(ev.value, &0);
        assert_eq!(ev.span.active, span!(1 / 4, 1 / 2));
        let ev = evs.next().unwrap();
        assert_eq!(ev.value, &1);
        assert_eq!(ev.span.active, span!(1 / 2, 1 / 1));
        let ev = evs.next().unwrap();
        assert_eq!(ev.value, &2);
        assert_eq!(ev.span.active, span!(1 / 1, 5 / 4));
        assert!(evs.next().is_none());
    }
}
