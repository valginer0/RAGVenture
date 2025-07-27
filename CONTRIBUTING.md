# Contributing to RAGVenture 🚀

First off, thank you for considering contributing to RAGVenture! It's people like you that make RAGVenture such a great tool.

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/valginer0/rag_startups.git`
3. Create your feature branch: `git checkout -b feature/amazing-feature`
4. Set up development environment:
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Unix or MacOS:
   source .venv/bin/activate

   # Install dependencies with development extras
   pip install -r requirements.txt

   # Install pre-commit hooks
   pre-commit install
   ```

## Development Process

1. **Code Style**
   - We use [Black](https://github.com/psf/black) for code formatting
   - Follow PEP 8 guidelines
   - Use type hints where possible
   - Keep functions focused and modular

2. **Documentation**
   - Document all functions and classes
   - Update README.md if adding new features
   - Add examples to docs/examples.md
   - Update API documentation in docs/api.md

3. **Testing** (177 tests - all passing)
   - Run full test suite: `python -m pytest tests/ -v`
   - Docker tests: `docker-compose run --rm app-cpu python -m pytest tests/ -v`
   - All tests must pass before submitting PR
   - Add tests for new features and bug fixes
   - Current coverage: Comprehensive with Docker runtime fixes

4. **Docker Development**
   - Docker containers are production-ready and fully tested
   - Test your changes in Docker: `docker-compose up app-cpu`
   - Verify Docker compatibility before submitting PR
   - All Docker runtime issues have been resolved

5. **Original Testing Guidelines**
   - Write tests for new features
   - Ensure all tests pass: `python -m pytest`
   - Add test cases for edge cases
   - Maintain test coverage

4. **Commit Guidelines**
   - Use clear, descriptive commit messages
   - Follow conventional commits format:
     - feat: new feature
     - fix: bug fix
     - docs: documentation changes
     - style: formatting, missing semicolons, etc.
     - refactor: code restructuring
     - test: adding tests
     - chore: maintenance

## Testing

We maintain a comprehensive test suite with 31 passing tests. Before submitting your PR:

1. Run the full test suite:
   ```bash
   pytest tests/
   ```

2. Ensure test coverage:
   ```bash
   pytest --cov=rag_startups tests/
   ```

3. Key test files:
   - `tests/test_rag_chain.py`: Core RAG functionality
   - `tests/idea_generator/test_generator.py`: Idea generation
   - `tests/test_data_loader.py`: Data loading and processing

4. Performance benchmarks (tests will fail if exceeded):
   - Data Loading: < 0.1s
   - Embedding Generation: < 25s
   - Idea Generation: < 1s per idea

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the docs/ with any new documentation
3. Add tests for new functionality
4. Ensure the test suite passes
5. Update the version numbers if applicable
6. The PR will be merged once you have the sign-off of at least one maintainer

## Reporting Bugs

When reporting bugs, please include:

1. Your operating system name and version
2. Python version and virtual environment details
3. Detailed steps to reproduce the bug
4. What you expected would happen
5. What actually happens

## Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When suggesting an enhancement, please:

1. Use a clear and descriptive title
2. Provide a step-by-step description of the suggested enhancement
3. Explain why this enhancement would be useful
4. List some examples of where this enhancement would be beneficial

## Local Development

1. **Environment Setup**
   ```bash
   # Clone your fork
   git clone https://github.com/valginer0/rag_startups.git
   cd rag_startups

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Running Tests**
   ```bash
   # Run all tests
   python -m pytest

   # Run specific test file
   python -m pytest tests/test_rag_chain.py

   # Run with coverage
   python -m pytest --cov=src/
   ```

3. **Code Formatting**
   ```bash
   # Format code with Black
   black .

   # Check formatting
   black . --check
   ```

## Getting Help

If you need help, you can:
1. Open an issue with the question label
2. Check existing issues and documentation
3. Reach out to maintainers

Thank you for contributing to RAGVenture! 🚀
