# Development & Contributing

## 🛠️ Linting

This project uses `ruff` for fast python linting and formatting.

```bash
pip install ruff
ruff check .
```

## 🧪 Running Tests

Unit tests live in the `tests/` folder. The tests are designed to mock GPU availability if it is missing, so they can run in standard CI environments (like GitHub Actions).

Run them with:

```bash
pytest -q
```

## 📄 License

This project is released under the [MIT License](https://github.com/artryazanov/shorts-maker-gpu/blob/main/LICENSE).
