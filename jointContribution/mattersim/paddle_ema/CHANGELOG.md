# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Most recent change on the bottom.

## [Unreleased]

## [0.3.0] - 2021-11-17
### Added
- `torch_ema.__version__`

### Changed
- Parameters without `requires_grad = True` are no longer partially ignored, resolving #9; now *all* parameters passed to the EMA object have EMA run on them, regardless of whether they are trainable or not. 

## [0.2.0] - 2021-07-07