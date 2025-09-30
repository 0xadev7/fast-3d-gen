
#!/usr/bin/env bash
set -euo pipefail
python scripts/download_weights.py || true
python app.py --host 0.0.0.0 --port 8093
