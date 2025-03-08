/// When i > j, we split xs at position j,
/// then merge each of the first j items of xs with the entire ys,
/// and whatever is left of xs remains as the new ys.
fn left(xs: &mut Vec<Vec<bool>>, ys: &mut Vec<Vec<bool>>) {
    let xs_p = xs.split_off(ys.len());
    xs.iter_mut().zip(&mut *ys).for_each(|(a, b)| a.append(b));
    *ys = xs_p;
}

/// When j > i, we split ys at position i,
/// then merge each of the first i items of ys with the entire xs,
/// and whatever is left of ys remains as the new ys.
fn right(xs: &mut Vec<Vec<bool>>, ys: &mut Vec<Vec<bool>>) {
    let ys_p = ys.split_off(xs.len());
    xs.iter_mut().zip(&mut *ys).for_each(|(a, b)| a.append(b));
    *ys = ys_p;
}

fn bjorklund_iter(xs: &mut Vec<Vec<bool>>, ys: &mut Vec<Vec<bool>>) {
    // Repeatedly apply left or right until one of i or j is <= 1.
    loop {
        println!("{:?}", (&xs, &ys));
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

/// This is the main bjorklund function that returns a Vec<bool>
/// of length total = k + (n - k).
pub fn bjorklund(k: usize, n: usize) -> Vec<bool> {
    let mut xs = vec![vec![true]; k];
    let mut ys = vec![vec![false]; n - k];
    bjorklund_iter(&mut xs, &mut ys);
    // Concatenate (flatten) all sublists in the resulting xs ++ ys
    xs.into_iter().chain(ys).flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bjorklund_2_3() {
        let expected = vec![true, true, false];
        let actual = bjorklund(2, 3);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_2_5() {
        let expected = vec![true, false, true, false, false];
        let actual = bjorklund(2, 5);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_3_4() {
        let expected = vec![true, true, true, false];
        let actual = bjorklund(3, 4);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_3_5() {
        let expected = vec![true, false, true, false, true];
        let actual = bjorklund(3, 5);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_3_8() {
        let expected = vec![true, false, false, true, false, false, true, false];
        let actual = bjorklund(3, 8);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_4_7() {
        let expected = vec![true, false, true, false, true, false, true];
        let actual = bjorklund(4, 7);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_4_9() {
        let expected = vec![true, false, true, false, true, false, true, false, false];
        let actual = bjorklund(4, 9);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_4_12() {
        let expected = vec![
            true, false, false, true, false, false, true, false, false, true, false, false,
        ];
        let actual = bjorklund(4, 12);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_4_15() {
        let expected = vec![
            true, false, false, false, true, false, false, false, true, false, false, false, true,
            false, false,
        ];
        let actual = bjorklund(4, 15);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_5_6() {
        let expected = vec![true, true, true, true, true, false];
        let actual = bjorklund(5, 6);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_5_7() {
        let expected = vec![true, false, true, true, false, true, true];
        let actual = bjorklund(5, 7);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_5_8() {
        let expected = vec![true, false, true, true, false, true, true, false];
        let actual = bjorklund(5, 8);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_5_9() {
        let expected = vec![true, false, true, false, true, false, true, false, true];
        let actual = bjorklund(5, 9);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_5_11() {
        let expected = vec![
            true, false, true, false, true, false, true, false, true, false, false,
        ];
        let actual = bjorklund(5, 11);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_5_12() {
        let expected = vec![
            true, false, false, true, false, true, false, false, true, false, true, false,
        ];
        let actual = bjorklund(5, 12);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_5_13() {
        let expected = vec![
            true, false, false, true, false, true, false, false, true, false, true, false, false,
        ];
        let actual = bjorklund(5, 13);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_5_16() {
        let expected = vec![
            true, false, false, true, false, false, true, false, false, true, false, false, true,
            false, false, false,
        ];
        let actual = bjorklund(5, 16);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_6_7() {
        let expected = vec![true, true, true, true, true, true, false];
        let actual = bjorklund(6, 7);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_6_13() {
        let expected = vec![
            true, false, true, false, true, false, true, false, true, false, true, false, false,
        ];
        let actual = bjorklund(6, 13);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_7_8() {
        let expected = vec![true, true, true, true, true, true, true, false];
        let actual = bjorklund(7, 8);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_7_9() {
        let expected = vec![true, false, true, true, true, false, true, true, true];
        let actual = bjorklund(7, 9);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_7_10() {
        let expected = vec![
            true, false, true, true, false, true, true, false, true, true,
        ];
        let actual = bjorklund(7, 10);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_7_12() {
        let expected = vec![
            true, false, true, true, false, true, false, true, true, false, true, false,
        ];
        let actual = bjorklund(7, 12);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_7_15() {
        let expected = vec![
            true, false, true, false, true, false, true, false, true, false, true, false, true,
            false, false,
        ];
        let actual = bjorklund(7, 15);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_7_16() {
        let expected = vec![
            true, false, false, true, false, true, false, true, false, false, true, false, true,
            false, true, false,
        ];
        let actual = bjorklund(7, 16);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_7_17() {
        let expected = vec![
            true, false, false, true, false, true, false, false, true, false, true, false, false,
            true, false, true, false,
        ];
        let actual = bjorklund(7, 17);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_7_18() {
        let expected = vec![
            true, false, false, true, false, true, false, false, true, false, true, false, false,
            true, false, true, false, false,
        ];
        let actual = bjorklund(7, 18);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_8_17() {
        let expected = vec![
            true, false, true, false, true, false, true, false, true, false, true, false, true,
            false, true, false, false,
        ];
        let actual = bjorklund(8, 17);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_8_19() {
        let expected = vec![
            true, false, false, true, false, true, false, true, false, false, true, false, true,
            false, true, false, false, true, false,
        ];
        let actual = bjorklund(8, 19);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_9_14() {
        let expected = vec![
            true, false, true, true, false, true, true, false, true, true, false, true, true, false,
        ];
        let actual = bjorklund(9, 14);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_9_16() {
        let expected = vec![
            true, false, true, true, false, true, false, true, false, true, true, false, true,
            false, true, false,
        ];
        let actual = bjorklund(9, 16);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_9_22() {
        let expected = vec![
            true, false, false, true, false, true, false, false, true, false, true, false, false,
            true, false, true, false, false, true, false, true, false,
        ];
        let actual = bjorklund(9, 22);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_9_23() {
        let expected = vec![
            true, false, false, true, false, true, false, false, true, false, true, false, false,
            true, false, true, false, false, true, false, true, false, false,
        ];
        let actual = bjorklund(9, 23);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_11_12() {
        let expected = vec![
            true, true, true, true, true, true, true, true, true, true, true, false,
        ];
        let actual = bjorklund(11, 12);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_11_24() {
        let expected = vec![
            true, false, false, true, false, true, false, true, false, true, false, true, false,
            false, true, false, true, false, true, false, true, false, true, false,
        ];
        let actual = bjorklund(11, 24);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_13_24() {
        let expected = vec![
            true, false, true, true, false, true, false, true, false, true, false, true, false,
            true, true, false, true, false, true, false, true, false, true, false,
        ];
        let actual = bjorklund(13, 24);
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_bjorklund_15_34() {
        let expected = vec![
            true, false, false, true, false, true, false, true, false, true, false, false, true,
            false, true, false, true, false, true, false, false, true, false, true, false, true,
            false, true, false, false, true, false, true, false,
        ];
        let actual = bjorklund(15, 34);
        assert_eq!(actual, expected);
    }
}
