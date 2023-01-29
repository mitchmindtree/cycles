//! Control patterns and related items.

use crate::{atom, Pattern, Rational};

/// A pattern value type that allows for representing a set of labelled controls.
pub type Controls = std::collections::BTreeMap<String, Value>;

pub const SOUND: &str = "sound";
pub const NOTE: &str = "note";

/// The set of possible control value types.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    String(String),
    F64(f64),
    Rational(Rational),
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
