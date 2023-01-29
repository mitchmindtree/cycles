#![doc = include_str!("../README.md")]

use num_rational::Rational64;
pub use span::Span;
use std::{
    fmt,
    ops::{Add, Mul, Sub},
    sync::Arc,
};

mod span;

/// A composable abstraction for 1-dimensional patterns.
///
/// A [`Pattern`] is any type that may be [queried][`Pattern::query`] with a
/// [`Span`] to produce a sequence of [`Event<Self::Value>`]s.
//
// TODO: When returning `impl Trait` in trait methods is supported (see
// https://github.com/rust-lang/rust/issues/91611), review the following
// methods and return `impl Pattern<Value = Self::Value>` where suitable.
// Particularly, methods returning `DynPattern` (besides `into_dyn`) should
// be replaced.
pub trait Pattern {
    /// The type of the values emitted in the pattern's events.
    type Value;
    /// An iterator yielding the events occuring within a query's span.
    type Events: Iterator<Item = Event<Self::Value>>;

    /// Query the pattern for events within the given span.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cycles::{atom, saw, span, Pattern, Rational};
    ///
    /// let pattern = atom("hello");
    /// let mut events = pattern.query(span!(0/1, 1/1));
    /// assert_eq!(events.next().unwrap().value, "hello");
    /// assert_eq!(events.next(), None);
    ///
    /// let pattern = saw();
    /// assert_eq!(pattern.query(span!(0/1)).next().unwrap().value, Rational::new(0, 1));
    /// assert_eq!(pattern.query(span!(1/2)).next().unwrap().value, Rational::new(1, 2));
    /// ```
    fn query(&self, span: Span) -> Self::Events;

    /// Convert the pattern to a trait object behind an [`Arc`] and dynamically
    /// box queries in order to represent the pattern with a known, sized type.
    ///
    /// This is useful for storing multiple patterns within a single
    /// collection, or passing patterns between threads, etc.
    fn into_dyn(self) -> DynPattern<Self::Value>
    where
        Self: 'static + Sized,
    {
        DynPattern::new(self)
    }

    /// Map the values produced by pattern queries with the given function.
    fn map<T, F>(self, map: F) -> MapValues<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Value) -> T,
    {
        let pattern = self;
        let map = Arc::new(map);
        MapValues { pattern, map }
    }

    /// Map the start and end points of the pattern's query spans.
    fn map_query_points<F>(self, map: F) -> MapQueryPoints<Self, F>
    where
        Self: Sized,
        F: Fn(Rational) -> Rational,
    {
        let pattern = self;
        MapQueryPoints { pattern, map }
    }

    /// Map the active and whole span start and end points of events produced by pattern
    /// queries with the given function. Useful for mapping time.
    fn map_event_points<F>(self, map: F) -> MapEventPoints<Self, F>
    where
        Self: Sized,
        F: Fn(Rational) -> Rational,
    {
        let pattern = self;
        let map = Arc::new(map);
        MapEventPoints { pattern, map }
    }

    /// Map the events produced by pattern queries with the given function.
    fn map_events<F, T>(self, map: F) -> MapEvents<Self, F>
    where
        Self: Sized,
        F: Fn(Event<Self::Value>) -> Event<T>,
    {
        let pattern = self;
        let map = Arc::new(map);
        MapEvents { pattern, map }
    }

    /// Map the events iterator produced by the pattern queries with the given function.
    fn map_events_iter<E, F, T>(self, map: F) -> MapEventsIter<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Events) -> E,
        E: Iterator<Item = Event<T>>,
    {
        let pattern = self;
        MapEventsIter { pattern, map }
    }

    /// Increase or decrease the rate of event emission by the given value.
    fn rate(self, rate: Rational) -> Rate<Self>
    where
        Self: Sized,
    {
        let pattern = self;
        Rate { pattern, rate }
    }

    /// Apply the given pattern of functions to `self`.
    ///
    /// The resulting pattern yields an event at each the intersection of each
    /// of the active spans. The function must return the value along with the
    /// `whole` span, which should either come from one of the intersecting
    /// events, or the intersection of the two.
    fn apply_with<P, F, B>(self, apply: P) -> DynPattern<B>
    where
        Self: 'static + Sized,
        Self::Value: Clone,
        P: 'static + Pattern<Value = F>,
        F: Fn(ApplyEvent<Self::Value>) -> (B, Option<Span>),
    {
        let apply = Arc::new(apply);
        let applied = move |span: Span| {
            let apply = apply.clone();
            self.query(span).flat_map(move |ev| {
                apply.query(span).flat_map(move |ef| {
                    let ev = ev.clone();
                    ev.span.active.intersect(ef.span.active).map(|active| {
                        let new = ApplyEvent {
                            value: ev.value,
                            active,
                            left: EventSpan::new(ev.span.active, ev.span.whole),
                            right: EventSpan::new(ef.span.active, ef.span.whole),
                        };
                        let (value, whole) = (ef.value)(new);
                        Event::new(value, active, whole)
                    })
                })
            })
        };
        applied.into_dyn()
    }

    /// Apply the given pattern of functions to `self`.
    ///
    /// Yields an event at each intersection between the active spans of `self` and `apply`.
    ///
    /// The resulting structure is determined by the given function `structure`
    /// which provides the `whole` spans of the intersecting events produced by
    /// `self` and `apply` respectively.
    fn apply<P, F, G, B>(self, apply: P, structure: G) -> DynPattern<B>
    where
        Self: 'static + Sized,
        Self::Value: Clone,
        P: 'static + Pattern<Value = F>,
        F: Fn(Self::Value) -> B,
        G: 'static + Fn(Span, Span) -> Span,
    {
        let structure = Arc::new(structure);
        let apply = apply.map(move |f| {
            let structure = structure.clone();
            move |e: ApplyEvent<_>| {
                let value = f(e.value);
                let whole = e
                    .left
                    .whole
                    .and_then(|lw| e.right.whole.map(|rw| (*structure)(lw, rw)));
                (value, whole)
            }
        });
        self.apply_with(apply)
    }

    /// Apply the given pattern of functions to `self`.
    ///
    /// Yields an event at each intersection between the active spans of `self` and `apply`.
    ///
    /// The resulting structure is the intersection of `self` and `apply`.
    fn app<P, F, B>(self, apply: P) -> DynPattern<B>
    where
        Self: 'static + Sized,
        Self::Value: Clone,
        P: 'static + Pattern<Value = F>,
        F: Fn(Self::Value) -> B,
    {
        self.apply(apply, |l, r| {
            l.intersect(r)
                .expect("if `active` spans intersect, `whole` must too")
        })
    }

    /// Apply the given pattern of functions to `self`.
    ///
    /// Yields an event at each intersection between the active spans of `self` and `apply`.
    ///
    /// The resulting structure is carried from the left (i.e. `self`).
    fn appl<P, F, B>(self, apply: P) -> DynPattern<B>
    where
        Self: 'static + Sized,
        Self::Value: Clone,
        P: 'static + Pattern<Value = F>,
        F: Fn(Self::Value) -> B,
    {
        self.apply(apply, |l, _| l)
    }

    /// Apply the given pattern of functions to `self`.
    ///
    /// Yields an event at each intersection between the active spans of `self` and `apply`.
    ///
    /// The resulting structure is carried from the right (i.e. the `apply` pattern).
    fn appr<P, F, B>(self, apply: P) -> DynPattern<B>
    where
        Self: 'static + Sized,
        Self::Value: Clone,
        P: 'static + Pattern<Value = F>,
        F: Fn(Self::Value) -> B,
    {
        self.apply(apply, |_, r| r)
    }

    /// Merge the given pattern by calling the given function for each value at
    /// each active span intersection.
    fn merge_with<P, F, T>(self, other: P, merge: F) -> DynPattern<T>
    where
        Self: 'static + Sized,
        Self::Value: Clone,
        P: 'static + Pattern,
        P::Value: Clone,
        F: 'static + Fn(Self::Value, P::Value) -> T,
    {
        let merge = Arc::new(merge);
        let apply = other.map(move |o: P::Value| {
            let f = merge.clone();
            move |s: Self::Value| (*f)(s, o.clone())
        });
        self.app(apply)
    }

    /// Merge the given pattern by calling `Extend<P::Value>` for each value at
    /// intersections of active spans.
    ///
    /// Useful for applying one control pattern to another and producing the
    /// union between values.
    fn merge_extend<P>(self, other: P) -> DynPattern<Self::Value>
    where
        Self: 'static + Sized,
        Self::Value: Clone + Extend<<P::Value as IntoIterator>::Item>,
        P: 'static + Pattern,
        P::Value: Clone + IntoIterator,
    {
        self.merge_with(other, |mut s, o| {
            s.extend(o);
            s
        })
    }

    /// Assuming a pattern of values in the range 0 to 1, produces a pattern in the range -1 to 1.
    fn polar(self) -> MapValues<Self, fn(Self::Value) -> Self::Value>
    where
        Self: Sized,
        Self::Value: Polar,
    {
        self.map(Polar::polar)
    }

    /// Return a wrapper providing a `fmt::Debug` implementation for the pattern.
    ///
    /// Formats events resulting from a query to the given span.
    fn debug_span(&self, span: Span) -> PatternDebug<Self::Value, Self::Events>
    where
        Self: Sized,
    {
        let pattern = self;
        PatternDebug { pattern, span }
    }

    /// Return a wrapper providing a `fmt::Debug` implementation for the pattern.
    ///
    /// Formats events resulting from a query for a single cycle.
    fn debug(&self) -> PatternDebug<Self::Value, Self::Events>
    where
        Self: Sized,
    {
        self.debug_span(span!(0 / 1, 1 / 1))
    }
}

/// Types that can be sampled with a rational to produce a value.
///
/// Useful for representing continuous functions.
pub trait Sample {
    /// The type of value returned when sampled.
    type Value;
    /// Sample `self` with `rational` to produce a value.
    fn sample(&self, rational: Rational) -> Self::Value;
}

/// Types that can represent a polar value.
pub trait Polar:
    Sized + One + Add<Output = Self> + Mul<Output = Self> + Sub<Output = Self>
{
    /// Assuming `self` is a value in the range `0..=1`, produces the polar
    /// representation (`-1..=1`).
    fn polar(self) -> Self {
        self * (Self::ONE + Self::ONE) - Self::ONE
    }
}

/// Types that can represent the value `1`.
pub trait One {
    const ONE: Self;
}

// ----------------------------------------------------------------------------

/// The rational value type used throughout the library to represent a point
/// along a single dimension.
pub type Rational = Rational64;

/// A pattern value type that allows for representing a set of labelled controls.
pub type Controls = std::collections::BTreeMap<String, Value>;

const SOUND: &str = "sound";
const NOTE: &str = "note";

/// The set of possible control value types.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    String(String),
    F64(f64),
    Rational(Rational),
}

/// A dynamic representation of a [`Pattern`].
///
/// Useful for storing or sending patterns, at the cost of boxing queried
/// events and allocating the inner [`Pattern`] behind an ARC.
#[derive(Clone)]
pub struct DynPattern<T>(Arc<dyn Pattern<Value = T, Events = BoxEvents<T>>>);

/// A dynamic representation of a pattern's associated events iterator.
pub struct BoxEvents<T>(Box<dyn Iterator<Item = Event<T>>>);

/// A type providing a [`std::fmt::Debug`] implementation for types implementing [`Pattern`].
pub struct PatternDebug<'p, V, E> {
    pattern: &'p dyn Pattern<Value = V, Events = E>,
    span: Span,
}

/// An event yielded by a pattern query, see [`Pattern::query`].
#[derive(Clone, Copy, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct Event<T> {
    /// The span of the event (both "active" and "whole" parts).
    pub span: EventSpan,
    /// The value associated with the event.
    pub value: T,
}

/// Context given to a `Pattern::apply_with` function that can be used to
/// produce the applied event along with its associated structure.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct ApplyEvent<T> {
    /// The span of the left event (normally provided by `self`).
    pub left: EventSpan,
    /// The span of the right event (normally provided by the pattern of functions).
    pub right: EventSpan,
    /// The intersection of each event's active span.
    pub active: Span,
    /// The value from the "left" event (normally provided by `self`) that is to be mapped.
    pub value: T,
}

/// The span associated with a single event.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct EventSpan {
    /// The span within which the active part is relevant.
    /// Also referred to as the event "structure".
    /// This is only relevant to patterns of discrete events.
    /// Patterns of continuous values (i.e. signals) will always have a `whole` of `None`.
    pub whole: Option<Span>,
    /// The span over which the event's value is active.
    pub active: Span,
}

/// See the [`signal`] pattern constructor.
#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
pub struct Signal<S>(S);

/// See [`Pattern::map_events`].
#[derive(Debug)]
pub struct MapEvents<P, F> {
    pattern: P,
    map: Arc<F>,
}

/// See [`Pattern::map`].
#[derive(Debug)]
pub struct MapValues<P, F> {
    pattern: P,
    map: Arc<F>,
}

/// See [`Pattern::map_query_points`].
#[derive(Debug)]
pub struct MapQueryPoints<P, F> {
    pattern: P,
    map: F,
}

/// See [`Pattern::map_event_points`].
#[derive(Debug)]
pub struct MapEventPoints<P, F> {
    pattern: P,
    map: Arc<F>,
}

/// See [`Pattern::map_events_iter`].
#[derive(Debug)]
pub struct MapEventsIter<P, F> {
    pattern: P,
    map: F,
}

/// See [`Pattern::rate`].
#[derive(Debug)]
pub struct Rate<P> {
    pattern: P,
    rate: Rational,
}

/// The [`Pattern::Events`] type for [`MapEvents`].
#[derive(Debug)]
pub struct EventsMap<I, F> {
    events: I,
    map: Arc<F>,
}

/// The [`Pattern::Events`] type for [`MapValues`].
#[derive(Debug)]
pub struct EventsMapValues<I, F> {
    events: I,
    map: Arc<F>,
}

/// The [`Pattern::Events`] type for [`MapEventPoints`].
#[derive(Debug)]
pub struct EventsMapPoints<I, F> {
    events: I,
    map: Arc<F>,
}

/// The [`Pattern::Events`] type for [`Rate`].
#[derive(Debug)]
pub struct EventsRate<I> {
    events: I,
    rate: Rational,
}

// ----------------------------------------------------------------------------

impl<T> Event<T> {
    fn new(value: T, active: Span, whole: Option<Span>) -> Self {
        let span = EventSpan::new(active, whole);
        Self { span, value }
    }

    fn map<U>(self, map: impl FnOnce(T) -> U) -> Event<U> {
        let Event { span, value } = self;
        let value = map(value);
        Event::new(value, span.active, span.whole)
    }

    fn map_spans(self, map: impl Fn(Span) -> Span) -> Self {
        let active = map(self.span.active);
        let whole = self.span.whole.map(&map);
        let value = self.value;
        Self::new(value, active, whole)
    }

    fn map_points(self, map: impl Fn(Rational) -> Rational) -> Self {
        self.map_spans(|span| span.map(&map))
    }
}

impl EventSpan {
    pub fn new(active: Span, whole: Option<Span>) -> Self {
        EventSpan { active, whole }
    }
}

impl<T> BoxEvents<T> {
    fn new<E>(es: E) -> Self
    where
        E: 'static + Iterator<Item = Event<T>>,
    {
        Self(Box::new(es) as Box<_>)
    }
}

impl<T> DynPattern<T> {
    fn new<P>(pattern: P) -> Self
    where
        P: 'static + Pattern<Value = T>,
        P::Events: 'static,
        T: 'static,
    {
        let arc = Arc::new(pattern.map_events_iter(BoxEvents::new))
            as Arc<dyn Pattern<Value = T, Events = BoxEvents<T>>>;
        DynPattern(arc)
    }
}

// ----------------------------------------------------------------------------

impl<F, I, T> Pattern for F
where
    F: Fn(Span) -> I,
    I: Iterator<Item = Event<T>>,
{
    type Value = T;
    type Events = I;
    fn query(&self, span: Span) -> Self::Events {
        (*self)(span)
    }
}

impl<T> Pattern for DynPattern<T> {
    type Value = T;
    type Events = BoxEvents<T>;
    fn query(&self, span: Span) -> Self::Events {
        self.0.query(span)
    }
}

impl<S: Sample> Pattern for Signal<S> {
    type Value = S::Value;
    type Events = std::iter::Once<Event<Self::Value>>;
    fn query(&self, active @ Span { start, end }: Span) -> Self::Events {
        let Signal(s) = self;
        let value = s.sample(start + ((end - start) / 2));
        let whole = None;
        let event = Event::new(value, active, whole);
        std::iter::once(event)
    }
}

impl<P, F, T> Pattern for MapValues<P, F>
where
    P: Pattern,
    F: Fn(P::Value) -> T,
{
    type Value = T;
    type Events = EventsMapValues<P::Events, F>;
    fn query(&self, span: Span) -> Self::Events {
        let Self { pattern, map } = self;
        let events = pattern.query(span);
        let map = map.clone();
        EventsMapValues { events, map }
    }
}

impl<P, F> Pattern for MapQueryPoints<P, F>
where
    P: Pattern,
    F: Fn(Rational) -> Rational,
{
    type Value = P::Value;
    type Events = P::Events;
    fn query(&self, span: Span) -> Self::Events {
        let span = span.map(&self.map);
        self.pattern.query(span)
    }
}

impl<P, F> Pattern for MapEventPoints<P, F>
where
    P: Pattern,
    F: Fn(Rational) -> Rational,
{
    type Value = P::Value;
    type Events = EventsMapPoints<P::Events, F>;
    fn query(&self, span: Span) -> Self::Events {
        let Self { pattern, map } = self;
        let events = pattern.query(span);
        let map = map.clone();
        EventsMapPoints { events, map }
    }
}

impl<P, F, T> Pattern for MapEvents<P, F>
where
    P: Pattern,
    F: Fn(Event<P::Value>) -> Event<T>,
{
    type Value = T;
    type Events = EventsMap<P::Events, F>;
    fn query(&self, span: Span) -> Self::Events {
        let events = self.pattern.query(span);
        let map = self.map.clone();
        EventsMap { events, map }
    }
}

impl<P, F, E, T> Pattern for MapEventsIter<P, F>
where
    P: Pattern,
    F: Fn(P::Events) -> E,
    E: Iterator<Item = Event<T>>,
{
    type Value = T;
    type Events = E;
    fn query(&self, span: Span) -> Self::Events {
        let Self { pattern, map } = self;
        let events = pattern.query(span);
        map(events)
    }
}

impl<P> Pattern for Rate<P>
where
    P: Pattern,
{
    type Value = P::Value;
    type Events = EventsRate<P::Events>;
    fn query(&self, span: Span) -> Self::Events {
        let Self { ref pattern, rate } = *self;
        let span = span.map(|p| p * rate);
        let events = pattern.query(span);
        EventsRate { events, rate }
    }
}

impl<I, F, T, U> Iterator for EventsMap<I, F>
where
    I: Iterator<Item = Event<T>>,
    F: Fn(Event<T>) -> Event<U>,
{
    type Item = Event<U>;
    fn next(&mut self) -> Option<Self::Item> {
        self.events.next().map(&*self.map)
    }
}

impl<I, F, T, U> Iterator for EventsMapValues<I, F>
where
    I: Iterator<Item = Event<T>>,
    F: Fn(T) -> U,
{
    type Item = Event<U>;
    fn next(&mut self) -> Option<Self::Item> {
        self.events.next().map(|ev| ev.map(&*self.map))
    }
}

impl<I, F, T> Iterator for EventsMapPoints<I, F>
where
    I: Iterator<Item = Event<T>>,
    F: Fn(Rational) -> Rational,
{
    type Item = Event<T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.events.next().map(|ev| ev.map_points(&*self.map))
    }
}

impl<I, T> Iterator for EventsRate<I>
where
    I: Iterator<Item = Event<T>>,
{
    type Item = Event<T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.events
            .next()
            .map(|ev| ev.map_points(|p| p / self.rate))
    }
}

impl<T> Iterator for BoxEvents<T> {
    type Item = Event<T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<F, T> Sample for F
where
    F: Fn(Rational) -> T,
{
    type Value = T;
    fn sample(&self, r: Rational) -> Self::Value {
        (*self)(r)
    }
}

impl<T> Polar for T where T: One + Add<Output = Self> + Mul<Output = Self> + Sub<Output = Self> {}

impl One for Rational {
    const ONE: Self = Rational::new_raw(1, 1);
}

impl<T> fmt::Debug for Event<T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut d = f.debug_struct("Event");
        if let Some(whole) = self.span.whole {
            d.field("whole", &whole);
        }
        d.field("active", &self.span.active)
            .field("value", &self.value)
            .finish()
    }
}

impl<'p, V, E> fmt::Debug for PatternDebug<'p, V, E>
where
    E: Iterator<Item = Event<V>>,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let events = self.pattern.query(self.span);
        f.debug_list().entries(events).finish()
    }
}

// ----------------------------------------------------------------------------

/// A pattern that, when queried, always produces a single event sampled from the given function.
pub fn signal<S: Sample>(sample: S) -> impl Pattern<Value = S::Value> {
    Signal(sample)
}

/// When queried, always returns a single event with a clone of the given value.
// TODO: Better name = clone?
pub fn steady<T: Clone>(t: T) -> impl Pattern<Value = T> {
    signal(move |_| t.clone())
}

/// When queried, always produces an empty event iterator.
// TODO: Better name = empty?
pub fn silence<T>() -> impl Pattern<Value = T> {
    |_| std::iter::empty()
}

/// Repeats a given discrete value once per cycle.
// TODO: Better name = cycle?
pub fn atom<T: Clone>(t: T) -> impl Pattern<Value = T> {
    move |span: Span| {
        let t = t.clone();
        span.cycles().map(move |active| {
            let start = active.start.floor();
            let end = start + 1;
            let whole = Some(Span { start, end });
            let value = t.clone();
            Event::new(value, active, whole)
        })
    }
}

/// A signal pattern that produces a saw wave in the range 0..=1.
pub fn saw() -> impl Pattern<Value = Rational> {
    signal(|r| r % 1)
}

/// A signal pattern that produces a saw wave in the range -1..=1.
pub fn saw2() -> impl Pattern<Value = Rational> {
    saw().polar()
}

/// Concatenate the given sequence of patterns into a single pattern whose
/// total unique span covers a number of cycles equal to the number of patterns
/// in the sequence.
pub fn slowcat<I>(patterns: I) -> impl Pattern<Value = <I::Item as Pattern>::Value>
where
    I: IntoIterator,
    I::Item: Pattern,
{
    let patterns: Arc<[I::Item]> = patterns.into_iter().collect();
    move |span: Span| {
        let ps = patterns.clone();
        span.cycles().flat_map(move |cycle| {
            let sam = usize::try_from(cycle.start.floor().to_integer())
                .expect("failed to cast span start to usize");
            let ix = sam % ps.len();
            let p = &ps[ix];
            p.query(cycle)
        })
    }
}

/// Concatenate the given sequence of patterns into a single pattern so that
/// all patterns fit to a single cycle.
pub fn fastcat<I>(patterns: I) -> impl Pattern<Value = <I::Item as Pattern>::Value>
where
    I: IntoIterator,
    I::Item: Pattern,
    I::IntoIter: ExactSizeIterator,
{
    let patterns = patterns.into_iter();
    let n = i64::try_from(patterns.len()).expect("pattern count out of range");
    let rate = Rational::from_integer(n);
    slowcat(patterns).rate(rate)
}

/// Combine the patterns into a single "stacked" pattern, where each query
/// is equivalent to querying each of the inner patterns and concatenating their
/// produced events.
pub fn stack<I>(patterns: I) -> impl Pattern<Value = <I::Item as Pattern>::Value>
where
    I: IntoIterator,
    I::Item: Pattern,
{
    let patterns: Arc<[I::Item]> = patterns.into_iter().collect();
    move |span: Span| {
        let ps = patterns.clone();
        (0..ps.len()).flat_map(move |ix| ps[ix].query(span))
    }
}

/// Given a pattern of sound names, produce a control pattern of `"sound"` events.
pub fn sound<P>(pattern: P) -> impl Pattern<Value = Controls>
where
    P: 'static + Pattern,
    P::Value: Clone + Into<String>,
{
    let f = |s: P::Value| std::iter::once((SOUND.to_string(), Value::String(s.into()))).collect();
    pattern.app(atom(f))
}

/// Given a pattern of note values, produce a control pattern of `"note"` events.
pub fn note<P>(pattern: P) -> impl Pattern<Value = Controls>
where
    P: 'static + Pattern<Value = f64>,
{
    let f = |n| std::iter::once((NOTE.to_string(), Value::F64(n))).collect();
    pattern.app(atom(f))
}

// ----------------------------------------------------------------------------

#[test]
fn test_merge_extend() {
    let p = sound(atom("hello")).merge_extend(note(atom(4.0)));
    dbg!(p.debug_span(span!(0 / 1, 4 / 1)));
    let mut cycle = p.query(span!(0 / 1, 1 / 1));
    let mut expected = std::collections::BTreeMap::new();
    expected.insert(SOUND.to_string(), Value::String("hello".into()));
    expected.insert(NOTE.to_string(), Value::F64(4.0));
    assert_eq!(cycle.next().unwrap().value, expected);
    assert_eq!(cycle.next(), None);
}

#[test]
fn test_apply() {
    let a = atom(1.0).rate(2.into());
    let b = atom(|v| v + 2.0).rate(3.into());
    let p = a.app(b);
    let v: Vec<_> = p.query(span!(0 / 1, 1 / 1)).collect();
    // cycle1                        cycle2
    // a              a
    // b         b         b
    // 0/1       1/3  1/2  2/3       1/1
    // 1         2    3    4
    // |         |    |    |         |
    let s0 = span!(0 / 1, 1 / 3);
    let s1 = span!(1 / 3, 1 / 2);
    let s2 = span!(1 / 2, 2 / 3);
    let s3 = span!(2 / 3, 1 / 1);
    assert_eq!(v[0], Event::new(3.0, s0, Some(s0)));
    assert_eq!(v[1], Event::new(3.0, s1, Some(s1)));
    assert_eq!(v[2], Event::new(3.0, s2, Some(s2)));
    assert_eq!(v[3], Event::new(3.0, s3, Some(s3)));
    assert_eq!(v.len(), 4);
}

#[test]
fn test_rate() {
    let p = atom("hello");
    // Only one event per cycle by default.
    let mut q = p.query(span!(0 / 1, 1 / 1));
    assert!(q.next().is_some());
    assert!(q.next().is_none());
    // At double rate, should get 2 events per cycle.
    let p = p.rate(Rational::new(2, 1));
    let mut q = p.query(span!(0 / 1, 1 / 1));
    assert!(q.next().is_some());
    assert!(q.next().is_some());
    assert!(q.next().is_none());
    // If we now divide by 4, we should get half an event per cycle, or 1 per 2 cycles.
    let p = p.rate(Rational::new(1, 4));
    let mut q = p.query(span!(0 / 1, 2 / 1));
    assert!(q.next().is_some());
    assert!(q.next().is_none());
}

#[test]
fn test_slowcat() {
    let a = atom("a");
    let b = atom("b");
    let cat = slowcat([a.into_dyn(), b.into_dyn()]);
    let span = span!(0 / 1, 5 / 2);
    let mut es = cat
        .query(span)
        .map(|ev| (ev.value, ev.span.active, ev.span.whole));
    assert_eq!(
        Some(("a", span!(0 / 1, 1 / 1), Some(span!(0 / 1, 1 / 1)))),
        es.next()
    );
    assert_eq!(
        Some(("b", span!(1 / 1, 2 / 1), Some(span!(1 / 1, 2 / 1)))),
        es.next()
    );
    assert_eq!(
        Some(("a", span!(2 / 1, 5 / 2), Some(span!(2 / 1, 3 / 1)))),
        es.next()
    );
    assert_eq!(None, es.next());
}

#[test]
fn test_fastcat() {
    let a = atom("a");
    let b = atom("b");
    let cat = fastcat([a.into_dyn(), b.into_dyn()]);
    let span = span!(0 / 1, 5 / 4);
    let mut es = cat
        .query(span)
        .map(|ev| (ev.value, ev.span.active, ev.span.whole));
    assert_eq!(
        Some(("a", span!(0 / 1, 1 / 2), Some(span!(0 / 1, 1 / 2)))),
        es.next()
    );
    assert_eq!(
        Some(("b", span!(1 / 2, 1 / 1), Some(span!(1 / 2, 1 / 1)))),
        es.next()
    );
    assert_eq!(
        Some(("a", span!(1 / 1, 5 / 4), Some(span!(1 / 1, 3 / 2)))),
        es.next()
    );
    assert_eq!(None, es.next());
}

#[test]
fn test_span_cycles() {
    let span = span!(0 / 1, 3 / 1);
    assert_eq!(span.cycles().count(), 3);
}

#[test]
fn test_saw() {
    let max = 10;
    for n in 0..=max {
        let r = Rational::new(n, max);
        let i = span!(r);
        let v1 = saw().query(i).map(|ev| ev.value).next().unwrap();
        let v2 = saw2().query(i).map(|ev| ev.value).next().unwrap();
        println!("{}: v1={}, v2={}", r, v1, v2);
    }
}

#[test]
fn test_dyn_pattern() {
    let _patterns: Vec<DynPattern<_>> = vec![
        saw().into_dyn(),
        saw2().into_dyn(),
        silence().into_dyn(),
        steady(Rational::new(1, 1)).into_dyn(),
        atom(Rational::new(0, 1)).into_dyn(),
    ];
}

#[test]
fn test_steady() {
    let max = 10;
    for n in 0..=max {
        let i = span!(Rational::new(n, max));
        let v = steady("hello").query(i).map(|ev| ev.value).next().unwrap();
        assert_eq!(v, "hello");
    }
}

#[test]
fn test_silence() {
    let max = 10;
    for n in 0..=max {
        let i = span!(Rational::new(n, max));
        assert!(silence::<Rational>().query(i).next().is_none());
    }
}

#[test]
fn test_pattern_reuse() {
    let saw_ = saw();
    let max = 10;
    for n in 0..=max {
        let i = span!(Rational::new(n, max));
        let ev1 = saw_.query(i).next().unwrap();
        let ev2 = saw().query(i).next().unwrap();
        assert_eq!(ev1, ev2);
    }
}

#[test]
fn test_atom() {
    let span = span!(0 / 1, 3 / 1);
    let pattern = atom("hello");
    let mut values = pattern.query(span).map(|ev| ev.value);
    assert_eq!(Some("hello"), values.next());
    assert_eq!(Some("hello"), values.next());
    assert_eq!(Some("hello"), values.next());
    assert_eq!(None, values.next());
}

#[test]
fn test_atom_whole() {
    let span = span!(0 / 1, 7 / 2);
    let pattern = atom("hello");
    let mut events = pattern.query(span);
    {
        let mut values = events.by_ref().map(|ev| ev.value);
        assert_eq!(Some("hello"), values.next());
        assert_eq!(Some("hello"), values.next());
        assert_eq!(Some("hello"), values.next());
    }
    let event = events.next().unwrap();
    let active = span!(3 / 1, 7 / 2);
    let whole = Some(span!(3 / 1, 4 / 1));
    assert_eq!(active, event.span.active);
    assert_eq!(whole, event.span.whole);
    assert_eq!(None, events.next());
}

#[test]
fn test_debug() {
    let p = atom("hello");
    println!("{:?}", p.debug());
    println!("{:?}", p.debug_span(span!(2 / 1, 7 / 2)));
}
