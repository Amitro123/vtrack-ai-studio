# VTrackAI Studio - Test Suite

## Test Structure

```
backend/
├── conftest.py                    # Pytest configuration
├── test_video_preprocessor.py     # Unit tests for video preprocessing
├── test_api_integration.py        # Integration tests for API endpoints
├── test_e2e_pipeline.py           # End-to-end pipeline tests
└── requirements-test.txt          # Test dependencies
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r backend/requirements-test.txt
```

### Run All Tests

```bash
# From project root
python -m pytest backend/ -v

# With coverage report
python -m pytest backend/ -v --cov=backend --cov-report=html
```

### Run Specific Test Files

```bash
# Unit tests only
python -m pytest backend/test_video_preprocessor.py -v

# Integration tests only
python -m pytest backend/test_api_integration.py -v

# E2E tests only
python -m pytest backend/test_e2e_pipeline.py -v
```

### Run Tests by Marker

```bash
# Integration tests
python -m pytest backend/ -v -m integration

# E2E tests
python -m pytest backend/ -v -m e2e

# Slow tests
python -m pytest backend/ -v -m slow
```

## Test Coverage

### Unit Tests (`test_video_preprocessor.py`)
- ✅ Keyframe selection with various frame counts
- ✅ Keyframe selection edge cases (single frame, all frames)
- ✅ Mask interpolation nearest-neighbor logic
- ✅ Processing parameter retrieval for both modes

### Integration Tests (`test_api_integration.py`)
- ✅ Health check endpoints
- ✅ `/api/track-point` endpoint (fast & accurate modes)
- ✅ `/api/text-to-video` endpoint (fast & accurate modes)
- ✅ `/api/remove-object` endpoint (fast & accurate modes)
- ✅ Mode validation (invalid mode handling)
- ✅ Parameter validation (missing parameters)
- ✅ Response format verification (mode metadata)

### E2E Tests (`test_e2e_pipeline.py`)
- ✅ Full preprocessing pipeline (fast mode)
- ✅ Full preprocessing pipeline (accurate mode)
- ✅ No downsampling when not needed
- ✅ Mask interpolation quality
- ✅ Processing mode consistency
- ✅ Edge cases (single frame, zero frames, etc.)

## Test Features

- **Mocked SAM3 Engine**: Integration tests use mocked SAM3 to avoid GPU requirements
- **Temporary Test Videos**: E2E tests create temporary videos for testing
- **Comprehensive Coverage**: Tests cover happy paths, edge cases, and error handling
- **Fast Execution**: Most tests run in milliseconds
- **CI/CD Ready**: Tests can run in CI/CD pipelines without GPU

## Expected Test Results

All tests should pass:
- Unit tests: ~10 tests
- Integration tests: ~15 tests  
- E2E tests: ~12 tests

**Total: ~37 tests**

## Continuous Integration

Add to your CI/CD pipeline:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - run: pip install -r backend/requirements.txt
      - run: pip install -r backend/requirements-test.txt
      - run: pytest backend/ -v --cov=backend
```
