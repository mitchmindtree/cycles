//! An implementation of TidalCycles' mini notation.

#[cfg(test)]
use crate::Pattern;

/// The mini-notation language implementation.
#[macro_export]
macro_rules! m {
    // Match groups.
    (slowcat <$($elem:tt),*>) => {{
        $crate::slowcat([$( $crate::Pattern::into_dyn(m!(elem $elem)) ),*])
    }};
    (fastcat [$($elem:tt),*]) => {{
        $crate::fastcat([$( $crate::Pattern::into_dyn(m!(elem $elem)) ),*])
    }};

    // Match elements.
    (elem $l:literal) => {{
        $crate::atom($l)
    }};
    (elem ~) => {{
        $crate::silence()
    }};
    (elem $i:ident) => {{
        $crate::atom(stringify!($i))
    }};
    (elem <$($elem:tt)*>) => {{
        m![slowcat <$($elem),*>]
    }};
    (elem [$($elem:tt)*]) => {{
        m![fastcat [$($elem),*]]
    }};
    (elem $(ts:tt)*) => {{
        $crate::silence()
    }};

    // Entry points.
    (<$($slow_ts:tt)*>) => {{
        m![slowcat <$($slow_ts),*>]
    }};
    ($($fast_ts:tt)*) => {{
        m![fastcat [$($fast_ts),*]]
    }};
}

#[test]
fn test_nested_array() {
    dbg!(m![0 [1 2 [3 4 5]] 6].debug());
}

#[test]
fn test_ident() {
    dbg!(m![bd bd sn bd].debug());
}

#[test]
fn test_int_array() {
    dbg!(m![0 1 2 3 4].debug());
}

#[test]
fn test_rest() {
    dbg!(m![0 ~ 2 ~ ~ 6 ~ 8].debug());
}

#[test]
fn test_alternate() {
    dbg!(m![bd <sn bd> sn].debug());
}
