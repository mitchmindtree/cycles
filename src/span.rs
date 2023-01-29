//! The `Span` type and related items.

use crate::Rational;
use std::fmt;

/// A shorthand macro for constructing spans from rationals, e.g. `span!(0/1, 3/1)`.
#[macro_export]
macro_rules! span {
    ($n1:literal/$d1:literal, $n2:literal/$d2:literal) => {{
        span!($n1 / $d1, $crate::Rational::new_raw($n2, $d2))
    }};
    ($n1:literal/$d1:literal, $r2:expr) => {{
        span!($crate::Rational::new_raw($n1, $d1), $r2)
    }};
    ($r1:expr, $n2:literal/$d2:literal) => {{
        span!($r1, $crate::Rational::new_raw($n2, $d2))
    }};
    ($r1:expr, $r2:expr) => {{
        $crate::Span::new($r1, $r2)
    }};
    ($n:literal / $d:literal) => {{
        span!($crate::Rational::new_raw($n, $d))
    }};
    ($r:expr) => {{
        $crate::Span::instant($r)
    }};
}

/// A rational range over a single dimension, represented with a start and end.
#[derive(Clone, Copy, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct Span {
    pub start: Rational,
    pub end: Rational,
}

impl Span {
    pub fn new(start: Rational, end: Rational) -> Self {
        Span { start, end }
    }

    pub fn instant(start @ end: Rational) -> Self {
        Span { start, end }
    }

    pub fn cycles(self) -> impl Iterator<Item = Self> {
        let Span { mut start, end } = self;
        std::iter::from_fn(move || {
            if start >= end {
                None
            } else if start >= end.floor() {
                let span = Span { start, end };
                start = end;
                Some(span)
            } else {
                let this_end = start.floor() + 1;
                let span = Span {
                    start,
                    end: this_end,
                };
                start = this_end;
                Some(span)
            }
        })
    }

    pub fn map(self, f: impl Fn(Rational) -> Rational) -> Self {
        span!(f(self.start), f(self.end))
    }

    pub fn intersect(self, other: Self) -> Option<Self> {
        let start = std::cmp::max(self.start, other.start);
        let end = std::cmp::min(self.end, other.end);
        if end <= start {
            None
        } else {
            Some(span!(start, end))
        }
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Span({}, {})", self.start, self.end)
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.start, self.end)
    }
}

#[test]
fn test_span_macro() {
    assert_eq!(
        span!(0 / 1, 1 / 1),
        Span::new(Rational::new(0, 1), Rational::new(1, 1))
    );
    assert_eq!(
        span!(Rational::new(1, 1), 4 / 1),
        span!(1 / 1, Rational::new(4, 1)),
    );
}

#[test]
fn test_span_fmt() {
    for n in 0..10 {
        let a = Rational::new(n, 10);
        let b = Rational::new(n + 1, 10);
        let span = span!(a, b);
        println!("{:?} | {}", span, span);
    }
}

#[test]
fn test_span_intersect() {
    assert_eq!(
        span!(0 / 1, 3 / 4).intersect(span!(1 / 4, 1 / 1)),
        Some(span!(1 / 4, 3 / 4))
    );
    assert_eq!(span!(0 / 1, 1 / 4).intersect(span!(3 / 4, 1 / 1)), None);
}
