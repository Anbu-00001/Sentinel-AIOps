PYTHON  ?= python3
PYTEST  ?= pytest
COMPOSE ?= docker compose

.PHONY: help train preprocess retrain resign drift ablation \
        test test-fast lint clean docker-build docker-up \
        docker-down docker-logs secrets-check

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; \
	         {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── ML Pipeline ──────────────────────────────────────────────────────

preprocess: ## Run feature engineering pipeline (generates artifacts)
	CI=1 TESTING=1 $(PYTHON) models/preprocess.py

train: preprocess ## Preprocess then train LightGBM classifier
	TESTING=1 $(PYTHON) models/train_v2.py

retrain: train resign ## Full retrain cycle: preprocess + train + sign
	@echo "Retrain complete. Run 'make ablation' to verify signal."

resign: ## Re-sign all .joblib artifacts with MODEL_SIGNATURE_SECRET
	@if [ -z "$$MODEL_SIGNATURE_SECRET" ]; then \
	  echo "ERROR: MODEL_SIGNATURE_SECRET is not set."; \
	  echo "       Export it or add it to your .env file."; \
	  exit 1; \
	fi
	TESTING=1 $(PYTHON) models/resign_artifacts.py

drift: ## Run PSI drift monitor against training distribution
	$(PYTHON) models/drift_monitor.py

ablation: ## Run TF-IDF ablation study and report F1 delta
	TESTING=1 $(PYTHON) scripts/ablation_runner.py

# ── Testing ───────────────────────────────────────────────────────────

test: ## Run full test suite with verbose output
	TESTING=1 $(PYTEST) tests/ -v --tb=short

test-fast: ## Run tests excluding slow ablation fixture
	TESTING=1 $(PYTEST) tests/ -v --tb=short \
	  --ignore=tests/test_ablation.py -q

lint: ## Run flake8 on all source files
	flake8 models/ mcp_server/ dashboard/ database/ tests/ \
	  scripts/ config.py \
	  --max-line-length 120 \
	  --extend-ignore E501,W503,E402 \
	  --count --statistics

# ── Docker ────────────────────────────────────────────────────────────

docker-build: ## Build all Docker images
	$(COMPOSE) build

docker-up: ## Start all services (detached)
	$(COMPOSE) up -d

docker-down: ## Stop all services and remove containers
	$(COMPOSE) down

docker-logs: ## Tail logs from all services
	$(COMPOSE) logs -f

# ── Utilities ─────────────────────────────────────────────────────────

secrets-check: ## Verify all required environment variables are set
	@echo "Checking required environment variables..."
	@missing=0; \
	for var in MODEL_SIGNATURE_SECRET GITHUB_WEBHOOK_SECRET \
	            SENTINEL_API_KEY; do \
	  if [ -z "$$(eval echo \$$$$var)" ]; then \
	    echo "  MISSING: $$var"; \
	    missing=$$((missing+1)); \
	  else \
	    echo "  OK:      $$var"; \
	  fi; \
	done; \
	if [ $$missing -gt 0 ]; then \
	  echo ""; \
	  echo "$$missing secret(s) not set."; \
	  echo "See .env.example for generation instructions."; \
	  exit 1; \
	else \
	  echo "All secrets are set."; \
	fi

clean: ## Remove Python cache files and runtime outputs
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	find . -name "*.pyo" -delete 2>/dev/null; \
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; \
	rm -f models/drift_report.json models/v2_report.json \
	      models/metrics.json models/anomaly_scores.npy \
	      models/pr_curve.png models/confusion_matrix.png \
	      models/feature_importance.png; \
	echo "Clean complete."
