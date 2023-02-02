//! A mini-notation implementation inspired by tidalcycles'.

#[cfg(test)]
use crate::Pattern;

/// A mini-notation implementation inspired by tidalcycles'.
#[macro_export]
macro_rules! m {
    (atom $i:ident) => {{ $crate::atom(stringify!($i)) }};
    (atom $l:literal) => {{ $crate::atom($l) }};
    (atom ~) => {{ $crate::silence() }};
    (atom ($t:tt) $elongate:literal $($ts:tt)*) => {{
        let rate = $crate::Rational::new(1, $elongate);
        let p = $crate::Pattern::into_dyn($crate::Pattern::rate(m!(atom $t), rate));
        let ps = m!(group $($ts)*);
        let silence = (0..$elongate-1).map(|_| $crate::Pattern::into_dyn($crate::silence()));
        let elongated = $crate::mini::Chain(std::iter::once(p).chain(silence));
        $crate::mini::Chain(elongated.chain(ps))
    }};

    (group $i:ident($a:literal,$b:literal) $($ts:tt)*) => {{
        compile_error!("Euclidean patterns not yet supported")
    }};
    (group $l:literal($a:literal,$b:literal) $($ts:tt)*) => {{
        compile_error!("Euclidean patterns not yet supported")
    }};
    (group {{$e:expr}} $($ts:tt)*) => {{
        compile_error!("Expressions unhandled")
    }};
    (group $i:ident _ _ _ _ _ _ _ $($ts:tt)*) => {{ m!(atom ($i) 8 $($ts)*) }};
    (group $i:ident _ _ _ _ _ _ $($ts:tt)*) => {{ m!(atom ($i) 7 $($ts)*) }};
    (group $i:ident _ _ _ _ _ $($ts:tt)*) => {{ m!(atom ($i) 6 $($ts)*) }};
    (group $i:ident _ _ _ _ $($ts:tt)*) => {{ m!(atom ($i) 5 $($ts)*) }};
    (group $i:ident _ _ _ $($ts:tt)*) => {{ m!(atom ($i) 4 $($ts)*) }};
    (group $i:ident _ _ $($ts:tt)*) => {{ m!(atom ($i) 3 $($ts)*) }};
    (group $i:ident _ $($ts:tt)*) => {{ m!(atom ($i) 2 $($ts)*) }};
    (group $i:ident $($ts:tt)*) => {{ m!(atom ($i) 1 $($ts)*) }};
    (group $l:literal _ _ _ _ _ _ _ $($ts:tt)*) => {{ m!(atom ($l) 8 $($ts)*) }};
    (group $l:literal _ _ _ _ _ _ $($ts:tt)*) => {{ m!(atom ($l) 7 $($ts)*) }};
    (group $l:literal _ _ _ _ _ $($ts:tt)*) => {{ m!(atom ($l) 6 $($ts)*) }};
    (group $l:literal _ _ _ _ $($ts:tt)*) => {{ m!(atom ($l) 5 $($ts)*) }};
    (group $l:literal _ _ _ $($ts:tt)*) => {{ m!(atom ($l) 4 $($ts)*) }};
    (group $l:literal _ _ $($ts:tt)*) => {{ m!(atom ($l) 3 $($ts)*) }};
    (group $l:literal _ $($ts:tt)*) => {{ m!(atom ($l) 2 $($ts)*) }};
    (group $l:literal $($ts:tt)*) => {{ m!(atom ($l) 1 $($ts)*) }};
    (group ~ _ _ _ _ _ _ _ $($ts:tt)*) => {{ m!(atom (~) 8 $($ts)*) }};
    (group ~ _ _ _ _ _ _ $($ts:tt)*) => {{ m!(atom (~) 7 $($ts)*) }};
    (group ~ _ _ _ _ _ $($ts:tt)*) => {{ m!(atom (~) 6 $($ts)*) }};
    (group ~ _ _ _ _ $($ts:tt)*) => {{ m!(atom (~) 5 $($ts)*) }};
    (group ~ _ _ _ $($ts:tt)*) => {{ m!(atom (~) 4 $($ts)*) }};
    (group ~ _ _ $($ts:tt)*) => {{ m!(atom (~) 3 $($ts)*) }};
    (group ~ _ $($ts:tt)*) => {{ m!(atom (~) 2 $($ts)*) }};
    (group ~ $($ts:tt)*) => {{ m!(atom (~) 1 $($ts)*) }};
    (group $l:literal $($ts:tt)*) => {{
        let p = $crate::Pattern::into_dyn($crate::atom($l));
        let ps = m!(group $($ts)*);
        $crate::mini::Chain(std::iter::once(p).chain(ps))
    }};
    (group [$($a:tt)*], [$($b:tt)*]) => {{
        let stack = $crate::stack([
            $crate::fastcat(m!(group $($a)*)).into_dyn(),
            $crate::fastcat(m!(group $($b)*)).into_dyn(),
        ]).into_dyn();
        std::iter::once(stack)
    }};
    (group [$($fastcat:tt)*] $($ts:tt)*) => {{
        let p = $crate::Pattern::into_dyn(m!($($fastcat)*));
        let ps = m!(group $($ts)*);
        $crate::mini::Chain(std::iter::once(p).chain(ps))
    }};
    (group <($($slowcat:tt)*)> $($ts:tt)*) => {{
        let p = $crate::Pattern::into_dyn($crate::slowcat(m!(group $($slowcat)*)));
        let ps = m!(group $($ts)*);
        $crate::mini::Chain(std::iter::once(p).chain(ps))
    }};
    (group $($ts:tt)*) => {{
        []
    }};

    ($($ts:tt)*) => {{
        $crate::fastcat(m!(group $($ts)*))
    }};
}

// Provides an `ExactSizeIterator` implementation `for std::iter::Chain`
// The only reason it's omitted from the std library is that the sum of
// the two lengths might overflow `usize`... To account for this, we check
// for the upper bound on the size_hint explicitly.
pub struct Chain<A, B>(pub std::iter::Chain<A, B>);

impl<A, B> Iterator for Chain<A, B>
where
    A: Iterator,
    B: Iterator<Item = A::Item>,
{
    type Item = A::Item;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<A, B> ExactSizeIterator for Chain<A, B>
where
    A: Iterator,
    B: Iterator<Item = A::Item>,
{
    fn len(&self) -> usize {
        let (lower, upper) = self.size_hint();
        let upper = upper.expect("no upperbound on iterator");
        assert_eq!(lower, upper);
        upper
    }
}

#[test]
fn test_ident_atom() {
    dbg!(m![bd].debug());
}

#[test]
fn test_ident_seq() {
    dbg!(m![bd sn cp td].debug());
}

#[test]
fn test_literal_seq() {
    dbg!(m![0 1 2 3 4].debug());
}

#[test]
fn test_literal_fastcat() {
    dbg!(m![0 [1 2] 3 [4 5]].debug());
}

#[test]
fn test_fastcat() {
    let a = m![bd sn].query_cycle().collect::<Vec<_>>();
    let b = m![[[[bd sn]]]].query_cycle().collect::<Vec<_>>();
    assert_eq!(a, b);
}

#[test]
fn test_rest() {
    dbg!(m![0 ~ 2 ~ ~ 6 ~ 8].debug());
}

#[test]
fn test_elongate() {
    dbg!(m![bd _ _ _ sn _].debug());
    dbg!(m![0 _ _ _ 1 _].debug());
}

#[test]
fn test_slowcat() {
    use crate::span;
    dbg!(m![a b <(c d)>].debug_span(span!(0 / 1, 4 / 1)));
}

#[test]
fn test_stack() {
    dbg!(m![[bd bd], [sn sn sn]].debug());
}
