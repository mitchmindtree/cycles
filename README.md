# cycles [![crates.io](https://img.shields.io/crates/v/cycles.svg)][crates-io] [![docs.rs](https://docs.rs/cycles/badge.svg)][docs-rs]

**A cyclic pattern abstraction, heavily inspired by [TidalCycles][tidalcycles].**

Started as an attempt at porting the `Pattern` abstraction and related items
from the TidalCycles' Haskell implementation, though some liberties have been
taken in order to achieve a more Rust-esque API.

The goal of this crate is to aim for a similar level of ergonomics to
TidalCycles (it's hard to compete with Haskell!), while taking advantage of
Rust's ability to provide low-to-zero-cost abstractions.

## The [`Pattern`] trait

The essence of this crate is the [`Pattern`] trait. `Pattern`s are types
that can be queried with a [`Span`] to produce a sequence of [`Event`]s. All
other items are related to constructing, applying, modifying or mapping types
implementing `Pattern`.

[crates-io]: https://crates.io/crates/tidalcycles
[docs-rs]: https://docs.rs/tidalcycles/
[tidalcycles]: https://tidalcycles.org/