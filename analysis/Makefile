.PHONY: kurulum kontrol test reproduce temiz

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

kurulum:          ## Bagimliliklari kur
	$(PYTHON) -m pip install -r requirements.txt

kontrol:          ## Ortami dogrula (OpenCV 5, calibrateMultiview)
	$(PYTHON) scripts/ortam_kontrol.py

test:             ## Testleri calistir
	$(PYTHON) -m pytest -q

reproduce:        ## Rapordaki tum sayilari yeniden uret
	$(PYTHON) -m eval.report
	$(PYTHON) scripts/deney_sentetik_kalibrasyon.py

temiz:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
