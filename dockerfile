FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .

RUN pip install uv
RUN uv sync --frozen

COPY . .

EXPOSE 5000

CMD ["uv", "run", "python", "app.py"]