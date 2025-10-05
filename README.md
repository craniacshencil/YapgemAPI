# YapgemAPI

```bash
conda create -n yapgem python=3.11
conda activate yapgem
```

```bash
pip install -r requirements.txt
```

- Might face issues with numpy versions or something.
- Models do get downloaded only once and are saved locally, subsequent requests then use the cached models reducing latency

```bash
uvicorn main:app --reload
```
