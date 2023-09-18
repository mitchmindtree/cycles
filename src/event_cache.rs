use crate::{span::Span, Event, Pattern};

#[derive(Debug)]
pub struct EventCache<T> {
    /// The span of time over which events are currently cached.
    span: Span,
    /// The cached events, all of which have active spans that intersect `span`.
    events: Vec<Event<T>>,
}

impl<T: std::fmt::Debug> EventCache<T> {
    /// Cached span.
    pub fn span(&self) -> &Span {
        &self.span
    }

    /// Cached events.
    pub fn events(&self) -> &[Event<T>] {
        &self.events
    }

    /// To be called in the case that the cache is invalidated due to
    /// the pattern changing in some way.
    pub fn reset(&mut self) {
        self.span = Span::new(0.into(), 0.into());
        self.events.clear();
    }

    /// Efficiently update the cache to contain all events within the given `new_span`.
    ///
    /// This method will only query the difference between `self.span`
    /// and `new_span` to reduce the load on calls to `query`.
    ///
    /// Returns whether or not the cache was mutated in some way.
    pub fn update(&mut self, new_span: Span, pattern: impl Pattern<Value = T>) {
        // First, remove events that no longer intersect the new span.
        crate::slice::retain_intersecting(&mut self.events, new_span);
        // Find the new spans and query events for them.
        let (pre, post) = self.span.difference(new_span);
        let old_evs = std::mem::replace(&mut self.events, vec![]);
        let mut pre_evs: Vec<_> = pre
            .into_iter()
            .flat_map(|pre| {
                pattern.query(pre).filter(move |ev| {
                    let ev_span = ev.span.whole_or_active();
                    pre.contains(ev_span.end)
                })
            })
            .collect();
        let mut post_evs: Vec<_> = post
            .into_iter()
            .flat_map(|post| {
                pattern.query(post).filter(move |ev| {
                    let ev_span = ev.span.whole_or_active();
                    post.contains(ev_span.start)
                })
            })
            .collect();
        pre_evs.sort_by_key(|ev| ev.span);
        post_evs.sort_by_key(|ev| ev.span);
        self.events.extend(pre_evs);
        self.events.extend(old_evs);
        self.events.extend(post_evs);
        self.span = new_span;
    }
}

impl<T> Default for EventCache<T> {
    fn default() -> Self {
        let span = Span::instant(0.into());
        let events = Default::default();
        Self { span, events }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::span;

    #[test]
    fn test_update_no_overlap() {
        let pattern = &[
            Event::new((), span!(0 / 1, 1 / 1), None),
            Event::new((), span!(1 / 1, 2 / 1), None),
            Event::new((), span!(2 / 1, 3 / 1), None),
        ][..];
        let span = span!(0 / 1, 3 / 1);
        let events = pattern.query(span).collect();
        let mut cache = EventCache { span, events };
        let new_span = span!(5 / 1, 8 / 1);
        cache.update(new_span, pattern);
        assert_eq!(cache.span, new_span);
        assert!(cache.events.is_empty());
    }

    #[test]
    fn test_update_partial_overlap() {
        let pattern = &[
            Event::new((), span!(0 / 1, 1 / 1), None),
            Event::new((), span!(1 / 1, 2 / 1), None),
            Event::new((), span!(2 / 1, 3 / 1), None),
            Event::new((), span!(3 / 1, 4 / 1), None),
            Event::new((), span!(4 / 1, 5 / 1), None),
        ][..];
        let span = span!(0 / 1, 3 / 1);
        let events = pattern.query(span).collect();
        let mut cache = EventCache { span, events };
        dbg!(&cache);
        let new_span = span!(2 / 1, 5 / 1);
        cache.update(new_span, pattern);
        assert_eq!(cache.span, new_span);
        assert_eq!(
            &cache.events[..],
            &[
                Event::new(&(), span!(2 / 1, 3 / 1), None),
                Event::new(&(), span!(3 / 1, 4 / 1), None),
                Event::new(&(), span!(4 / 1, 5 / 1), None),
            ][..],
        );
    }

    #[test]
    fn test_update_full_overlap() {
        let pattern = &[
            Event::new((), span!(0 / 1, 1 / 1), None),
            Event::new((), span!(1 / 1, 2 / 1), None),
            Event::new((), span!(2 / 1, 3 / 1), None),
        ][..];
        let span = span!(0 / 1, 3 / 1);
        let events = pattern.query(span).collect();
        let mut cache = EventCache { span, events };
        let new_span = span!(1 / 1, 2 / 1);
        cache.update(new_span, pattern);
        assert_eq!(cache.span, new_span);
        assert_eq!(
            &cache.events[..],
            &[Event::new(&(), span!(1 / 1, 2 / 1), None),],
        );
    }

    #[test]
    fn test_update_same_span() {
        let pattern = &[
            Event::new((), Span::instant(0.into()), None),
            Event::new((), Span::instant(1.into()), None),
            Event::new((), Span::instant(2.into()), None),
        ][..];
        let span = span!(0 / 1, 3 / 1);
        let events = pattern.query(span).collect();
        let mut cache = EventCache { span, events };
        let evs_before = cache.events.clone();
        cache.update(span, pattern);
        let evs_after = cache.events.clone();
        assert_eq!(evs_before, evs_after);
    }
}
