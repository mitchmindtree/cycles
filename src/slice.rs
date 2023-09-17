//! Items related to the `Pattern` implementation for slices.

use crate::{span::Span, Event, Pattern};

/// An iterator produced by a query to a rendered slice of events.
pub struct SliceEvents<'a, T> {
    span: Span,
    events: std::slice::Iter<'a, Event<T>>,
}

impl<'a, T> Pattern for &'a [Event<T>] {
    type Value = &'a T;
    type Events = SliceEvents<'a, T>;
    fn query(&self, span: Span) -> Self::Events {
        let events = self.iter();
        SliceEvents { span, events }
    }
}

impl<'a, T> Iterator for SliceEvents<'a, T> {
    type Item = Event<&'a T>;
    fn next(&mut self) -> Option<Self::Item> {
        while let Some(ev) = self.events.next() {
            if let Some(active) = self.span.intersect(ev.span.whole_or_active()) {
                return Some(Event::new(&ev.value, active, ev.span.whole));
            }
        }
        None
    }
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
