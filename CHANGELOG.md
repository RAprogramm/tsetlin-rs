# Changelog

> All notable changes to this project will be documented in this file.
> Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) + [Semantic Versioning](https://semver.org/)

## [Unreleased]

---

## [0.3.0] - 2026-01-23

> [!IMPORTANT]
> Sparse clause representation for memory-efficient inference.

### Added

| Change | Author | Branch | Ref |
|--------|--------|--------|-----|
| Sparse clause representation | [@RAprogramm](https://github.com/RAprogramm) | `2` | [#23](https://github.com/RAprogramm/tsetlin-rs/pull/23) |

### Changed

| Change | Author | Branch | Ref |
|--------|--------|--------|-----|
| professional changelog with Keep a Changelog format | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`9df320e`](https://github.com/RAprogramm/tsetlin-rs/commit/9df320eedac67d21ba3617b2ceb03c326fd10901) |
| Extract release notes from CHANGELOG.md | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`dea324d`](https://github.com/RAprogramm/tsetlin-rs/commit/dea324d) |
| Use cargo metadata for version extraction | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`8317560`](https://github.com/RAprogramm/tsetlin-rs/commit/8317560) |

---

## [0.2.2] - 2026-01-22

### Changed

| Change | Author | Branch | Ref |
|--------|--------|--------|-----|
| Add on-merge workflow | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`8cd35a8`](https://github.com/RAprogramm/tsetlin-rs/commit/8cd35a8) |

---

## [0.1.0] - 2026-01-21

> [!NOTE]
> Initial release with core Tsetlin Machine implementation.

### Added

| Change | Author | Branch | Ref |
|--------|--------|--------|-----|
| Const generics for compile-time optimization | [@RAprogramm](https://github.com/RAprogramm) | `main` | [#1](https://github.com/RAprogramm/tsetlin-rs/issues/1) |
| SIMD-accelerated clause evaluation | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`3cc9a1f`](https://github.com/RAprogramm/tsetlin-rs/commit/3cc9a1f) |
| No-std support for embedded systems | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`edcf4ec`](https://github.com/RAprogramm/tsetlin-rs/commit/edcf4ec) |
| Binary, multiclass, regression models | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`7f1aa15`](https://github.com/RAprogramm/tsetlin-rs/commit/7f1aa15) |
| Parallel training with Rayon | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`c1f3010`](https://github.com/RAprogramm/tsetlin-rs/commit/c1f3010) |

<details>
<summary><strong>Fixed</strong> (11)</summary>

| Change | Author | Branch | Ref |
|--------|--------|--------|-----|
| Use official REUSE badge | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`987959e`](https://github.com/RAprogramm/tsetlin-rs/commit/987959e) |
| Changelog generation logic | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`bcd0c93`](https://github.com/RAprogramm/tsetlin-rs/commit/bcd0c93) |
| Improve crates.io version check | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`75dfee6`](https://github.com/RAprogramm/tsetlin-rs/commit/75dfee6) |
| Changelog preserves existing entries | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`2a17528`](https://github.com/RAprogramm/tsetlin-rs/commit/2a17528) |
| Avoid duplicate CI runs for PRs | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`f0d4db9`](https://github.com/RAprogramm/tsetlin-rs/commit/f0d4db9) |
| Remove comma from cache key | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`470f25d`](https://github.com/RAprogramm/tsetlin-rs/commit/470f25d) |
| Changelog uses Cargo.toml version | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`032322e`](https://github.com/RAprogramm/tsetlin-rs/commit/032322e) |
| Trigger CI on push to any branch | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`fcd5ba3`](https://github.com/RAprogramm/tsetlin-rs/commit/fcd5ba3) |
| Quote if expression in changelog job | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`059e6dd`](https://github.com/RAprogramm/tsetlin-rs/commit/059e6dd) |
| Clippy, no_std imports, doc links | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`edcf4ec`](https://github.com/RAprogramm/tsetlin-rs/commit/edcf4ec) |
| Remove --locked flag from CI check | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`fb62a69`](https://github.com/RAprogramm/tsetlin-rs/commit/fb62a69) |

</details>

<details>
<summary><strong>Changed</strong> (14)</summary>

| Change | Author | Branch | Ref |
|--------|--------|--------|-----|
| Auto-create GitHub release | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`a7bfeda`](https://github.com/RAprogramm/tsetlin-rs/commit/a7bfeda) |
| Add Codecov token and test results | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`497931d`](https://github.com/RAprogramm/tsetlin-rs/commit/497931d) |
| Use CRATESIO secret name | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`cc2f365`](https://github.com/RAprogramm/tsetlin-rs/commit/cc2f365) |
| Modern badge styling with logos | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`dbc9d65`](https://github.com/RAprogramm/tsetlin-rs/commit/dbc9d65) |
| Improve README with TOC and badges | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`3eb323c`](https://github.com/RAprogramm/tsetlin-rs/commit/3eb323c) |
| Add Michael Tsetlin tribute section | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`c0e96b2`](https://github.com/RAprogramm/tsetlin-rs/commit/c0e96b2) |
| Add references to original paper | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`73f7f57`](https://github.com/RAprogramm/tsetlin-rs/commit/73f7f57) |
| Update lib.rs with clause types table | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`c1f3010`](https://github.com/RAprogramm/tsetlin-rs/commit/c1f3010) |
| Detailed clause documentation | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`7f1aa15`](https://github.com/RAprogramm/tsetlin-rs/commit/7f1aa15) |
| Modular CI with reusable workflows | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`497931d`](https://github.com/RAprogramm/tsetlin-rs/commit/497931d) |
| Coverage reporting with Codecov | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`497931d`](https://github.com/RAprogramm/tsetlin-rs/commit/497931d) |
| Cleanup stale workflow runs | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`a7bfeda`](https://github.com/RAprogramm/tsetlin-rs/commit/a7bfeda) |
| Rust caching for faster CI | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`cc2f365`](https://github.com/RAprogramm/tsetlin-rs/commit/cc2f365) |
| Auto-publish to crates.io on release | [@RAprogramm](https://github.com/RAprogramm) | `main` | [`cc2f365`](https://github.com/RAprogramm/tsetlin-rs/commit/cc2f365) |

</details>

---

[Unreleased]: https://github.com/RAprogramm/tsetlin-rs/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/RAprogramm/tsetlin-rs/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/RAprogramm/tsetlin-rs/compare/v0.1.0...v0.2.2
[0.1.0]: https://github.com/RAprogramm/tsetlin-rs/releases/tag/v0.1.0
