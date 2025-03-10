use std::{
    iter::{Chain, Flatten},
    vec::IntoIter,
};

/// The iterator type returned by [`bjorklund`].
///
/// Wraps the flattened iterator in order to provide `ExactSizeIterator`.
#[derive(Clone, Debug)]
pub struct Bjorklund {
    iter: Flatten<Chain<IntoIter<Vec<bool>>, IntoIter<Vec<bool>>>>,
    remaining: usize,
}

/// The iterator type returned by [`distances`] and [`Bjorklund::distances`].
///
/// Yields the distance until the next onset inclusive of the current event.
///
/// If there are no onsets, immediately returns `None`.
#[derive(Clone, Debug)]
pub struct BjorklundDistances {
    ix: usize,
    pattern: Vec<bool>,
}

impl Bjorklund {
    pub fn distances(self) -> BjorklundDistances {
        distances(self.collect())
    }
}

impl Iterator for Bjorklund {
    type Item = bool;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|b| {
            self.remaining -= 1;
            b
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl Iterator for BjorklundDistances {
    type Item = usize;
    fn next(&mut self) -> Option<Self::Item> {
        next_onset_distance(self.ix, &self.pattern).map(|dist| {
            self.ix += 1;
            dist
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for Bjorklund {
    fn len(&self) -> usize {
        self.remaining
    }
}

impl ExactSizeIterator for BjorklundDistances {
    fn len(&self) -> usize {
        if next_onset_distance(self.ix, &self.pattern).is_some() {
            self.pattern.len() - self.ix
        } else {
            0
        }
    }
}

/// Returns the cyclic distance until the next `true` onset, where the distance
/// is inclusive of the current event.
fn next_onset_distance(mut ix: usize, pattern: &[bool]) -> Option<usize> {
    if ix >= pattern.len() {
        return None;
    }
    let mut dist = 1;
    while dist <= pattern.len() {
        ix = (ix + 1) % pattern.len();
        if pattern[ix] {
            return Some(dist);
        }
        dist += 1;
    }
    None
}

/// When len(xs) > len(ys), we split xs at position len(ys),
/// then merge each of the first len(ys) items of xs with the entire ys,
/// and whatever is left of xs remains as the new ys.
fn left(xs: &mut Vec<Vec<bool>>, ys: &mut Vec<Vec<bool>>) {
    let xs_p = xs.split_off(ys.len());
    xs.iter_mut().zip(&mut *ys).for_each(|(a, b)| a.append(b));
    *ys = xs_p;
}

/// When len(ys) > len(xs), we split ys at position len(xs),
/// then merge each of the first len(xs) items of ys with the entire xs,
/// and whatever is left of ys remains as the new ys.
fn right(xs: &mut Vec<Vec<bool>>, ys: &mut Vec<Vec<bool>>) {
    let i = xs.len();
    let ys_i = ys.drain(..i);
    xs.iter_mut().zip(ys_i).for_each(|(a, b)| a.extend(b));
}

fn bjorklund_iter(xs: &mut Vec<Vec<bool>>, ys: &mut Vec<Vec<bool>>) {
    // Repeatedly apply left or right until one of len(xs) or len(ys) is <= 1.
    loop {
        if xs.len().min(ys.len()) <= 1 {
            break;
        }
        if xs.len() > ys.len() {
            left(xs, ys);
        } else {
            right(xs, ys);
        }
    }
}

/// The bjorklund pattern for euclidean rhythms, with `true`s as onsets.
pub fn bjorklund(k: usize, remaining @ n: usize) -> Bjorklund {
    let mut xs = vec![vec![true]; k];
    let mut ys = vec![vec![false]; n.saturating_sub(k)];
    bjorklund_iter(&mut xs, &mut ys);
    // Concatenate (flatten) all sublists in the resulting xs ++ ys
    let iter = xs.into_iter().chain(ys).flatten();
    Bjorklund { iter, remaining }
}

/// Given a bjorklund pattern, produces an iterator yielding the duration
/// until the next onset, inclusive of the current event.
pub fn distances(pattern: Vec<bool>) -> BjorklundDistances {
    BjorklundDistances { ix: 0, pattern }
}

/// Offset the given bjorklund pattern.
pub fn offset<I>(pattern: I, off: usize) -> impl Iterator<Item = bool>
where
    I: IntoIterator<Item = bool>,
    I::IntoIter: Clone + ExactSizeIterator,
{
    let iter = pattern.into_iter();
    let len = iter.len();
    iter.cycle().skip(off).take(len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bjorklund_2_3() {
        let expected = vec![true, true, false];
        let actual: Vec<_> = bjorklund(2, 3).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_2_5() {
        let expected = vec![true, false, true, false, false];
        let actual: Vec<_> = bjorklund(2, 5).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_3_4() {
        let expected = vec![true, true, true, false];
        let actual: Vec<_> = bjorklund(3, 4).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_3_5() {
        let expected = vec![true, false, true, false, true];
        let actual: Vec<_> = bjorklund(3, 5).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_3_8() {
        let expected = vec![true, false, false, true, false, false, true, false];
        let actual: Vec<_> = bjorklund(3, 8).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_4_7() {
        let expected = vec![true, false, true, false, true, false, true];
        let actual: Vec<_> = bjorklund(4, 7).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_4_9() {
        let expected = vec![true, false, true, false, true, false, true, false, false];
        let actual: Vec<_> = bjorklund(4, 9).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_4_12() {
        let expected = vec![
            true, false, false, true, false, false, true, false, false, true, false, false,
        ];
        let actual: Vec<_> = bjorklund(4, 12).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_4_15() {
        let expected = vec![
            true, false, false, false, true, false, false, false, true, false, false, false, true,
            false, false,
        ];
        let actual: Vec<_> = bjorklund(4, 15).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_5_6() {
        let expected = vec![true, true, true, true, true, false];
        let actual: Vec<_> = bjorklund(5, 6).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_5_7() {
        let expected = vec![true, false, true, true, false, true, true];
        let actual: Vec<_> = bjorklund(5, 7).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_5_8() {
        let expected = vec![true, false, true, true, false, true, true, false];
        let actual: Vec<_> = bjorklund(5, 8).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_5_9() {
        let expected = vec![true, false, true, false, true, false, true, false, true];
        let actual: Vec<_> = bjorklund(5, 9).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_5_11() {
        let expected = vec![
            true, false, true, false, true, false, true, false, true, false, false,
        ];
        let actual: Vec<_> = bjorklund(5, 11).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_5_12() {
        let expected = vec![
            true, false, false, true, false, true, false, false, true, false, true, false,
        ];
        let actual: Vec<_> = bjorklund(5, 12).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_5_13() {
        let expected = vec![
            true, false, false, true, false, true, false, false, true, false, true, false, false,
        ];
        let actual: Vec<_> = bjorklund(5, 13).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_5_16() {
        let expected = vec![
            true, false, false, true, false, false, true, false, false, true, false, false, true,
            false, false, false,
        ];
        let actual: Vec<_> = bjorklund(5, 16).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_6_7() {
        let expected = vec![true, true, true, true, true, true, false];
        let actual: Vec<_> = bjorklund(6, 7).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_6_13() {
        let expected = vec![
            true, false, true, false, true, false, true, false, true, false, true, false, false,
        ];
        let actual: Vec<_> = bjorklund(6, 13).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_7_8() {
        let expected = vec![true, true, true, true, true, true, true, false];
        let actual: Vec<_> = bjorklund(7, 8).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_7_9() {
        let expected = vec![true, false, true, true, true, false, true, true, true];
        let actual: Vec<_> = bjorklund(7, 9).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_7_10() {
        let expected = vec![
            true, false, true, true, false, true, true, false, true, true,
        ];
        let actual: Vec<_> = bjorklund(7, 10).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_7_12() {
        let expected = vec![
            true, false, true, true, false, true, false, true, true, false, true, false,
        ];
        let actual: Vec<_> = bjorklund(7, 12).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_7_15() {
        let expected = vec![
            true, false, true, false, true, false, true, false, true, false, true, false, true,
            false, false,
        ];
        let actual: Vec<_> = bjorklund(7, 15).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_7_16() {
        let expected = vec![
            true, false, false, true, false, true, false, true, false, false, true, false, true,
            false, true, false,
        ];
        let actual: Vec<_> = bjorklund(7, 16).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_7_17() {
        let expected = vec![
            true, false, false, true, false, true, false, false, true, false, true, false, false,
            true, false, true, false,
        ];
        let actual: Vec<_> = bjorklund(7, 17).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_7_18() {
        let expected = vec![
            true, false, false, true, false, true, false, false, true, false, true, false, false,
            true, false, true, false, false,
        ];
        let actual: Vec<_> = bjorklund(7, 18).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_8_17() {
        let expected = vec![
            true, false, true, false, true, false, true, false, true, false, true, false, true,
            false, true, false, false,
        ];
        let actual: Vec<_> = bjorklund(8, 17).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_8_19() {
        let expected = vec![
            true, false, false, true, false, true, false, true, false, false, true, false, true,
            false, true, false, false, true, false,
        ];
        let actual: Vec<_> = bjorklund(8, 19).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_9_14() {
        let expected = vec![
            true, false, true, true, false, true, true, false, true, true, false, true, true, false,
        ];
        let actual: Vec<_> = bjorklund(9, 14).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_9_16() {
        let expected = vec![
            true, false, true, true, false, true, false, true, false, true, true, false, true,
            false, true, false,
        ];
        let actual: Vec<_> = bjorklund(9, 16).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_9_22() {
        let expected = vec![
            true, false, false, true, false, true, false, false, true, false, true, false, false,
            true, false, true, false, false, true, false, true, false,
        ];
        let actual: Vec<_> = bjorklund(9, 22).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_9_23() {
        let expected = vec![
            true, false, false, true, false, true, false, false, true, false, true, false, false,
            true, false, true, false, false, true, false, true, false, false,
        ];
        let actual: Vec<_> = bjorklund(9, 23).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_11_12() {
        let expected = vec![
            true, true, true, true, true, true, true, true, true, true, true, false,
        ];
        let actual: Vec<_> = bjorklund(11, 12).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_11_24() {
        let expected = vec![
            true, false, false, true, false, true, false, true, false, true, false, true, false,
            false, true, false, true, false, true, false, true, false, true, false,
        ];
        let actual: Vec<_> = bjorklund(11, 24).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_13_24() {
        let expected = vec![
            true, false, true, true, false, true, false, true, false, true, false, true, false,
            true, true, false, true, false, true, false, true, false, true, false,
        ];
        let actual: Vec<_> = bjorklund(13, 24).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_15_34() {
        let expected = vec![
            true, false, false, true, false, true, false, true, false, true, false, false, true,
            false, true, false, true, false, true, false, false, true, false, true, false, true,
            false, true, false, false, true, false, true, false,
        ];
        let actual: Vec<_> = bjorklund(15, 34).collect();
        assert_eq!(actual, expected);
    }
}
