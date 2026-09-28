.PHONY: kurulum kurulum-dev kontrol test lint kapsam reproduce kilit kilit-surum canli temiz

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
# MediaPipe ayri ortamda (0008); varsayilan yollar bitirme-capture calisma agaci.
MP_PYTHON ?= ../bitirme-capture/data/mediapipe-env/bin/python
MP_MODEL ?= ../bitirme-capture/data/models/pose_landmarker_full-float16-v1.task
# El modeli (parmak eklemleri); dosya yoksa canli mod parmaksiz calisir.
MP_EL ?= data/dis/mediapipe_hand/hand_landmarker.task

kurulum:          ## Bagimliliklari kur
	$(PYTHON) -m pip install -r requirements.txt

kurulum-dev:      ## Gelistirme araclari (ruff, pytest-cov)
	$(PYTHON) -m pip install -r requirements-dev.txt

kontrol:          ## Ortami dogrula (OpenCV 5, calibrateMultiview)
	$(PYTHON) scripts/ortam_kontrol.py

test:             ## Testleri calistir; atlananlar nedeniyle listelenir (-rs)
	$(PYTHON) -m pytest -q -rs

lint:             ## Stil denetimi (ruff)
	$(PYTHON) -m ruff check .

kapsam:           ## Test kapsami (modul basina, sinanmayan satirlarla)
	$(PYTHON) -m pytest -q --cov=calib --cov=capture --cov=eval --cov=mono \
		--cov=pose3d --cov=uncertainty --cov-report=term-missing:skip-covered

reproduce:        ## Rapordaki tum sayilari yeniden uret
	$(PYTHON) -m eval.report
	$(PYTHON) scripts/deney_sentetik_kalibrasyon.py
	$(PYTHON) scripts/deney_triangulasyon_butcesi.py
	$(PYTHON) scripts/deney_hacim_haritasi.py
	$(PYTHON) scripts/deney_suruklenme.py
	$(PYTHON) scripts/deney_form_belirsizligi.py
	$(PYTHON) scripts/deney_aci_taramasi.py
	$(PYTHON) scripts/deney_oncu_hatasi.py
	$(PYTHON) scripts/deney_senkron_kaymasi.py
	$(PYTHON) scripts/deney_derinlik_duyarliligi.py
	$(PYTHON) scripts/deney_eklem_belirsizligi.py
	$(PYTHON) scripts/sayi_kilidi.py

kilit:            ## Rapordaki sayilari kilitle karsilastir (reproduce'dan sonra)
	$(PYTHON) scripts/sayi_kilidi.py

kilit-surum:      ## Calisan ortamin tam surumlerini requirements-lock.txt'e yaz
	$(PYTHON) -m pip freeze --exclude-editable | grep -v -i -E '^(ruff|pytest-cov|coverage)==' > requirements-lock.txt

canli:            ## Kameradan canli squat valgus denemesi (q: cikis)
	PYTHONPATH=. $(MP_PYTHON) -m mono.canli --model $(MP_MODEL) \
		$(if $(wildcard $(MP_EL)),--el-modeli $(MP_EL)) $(ARGS)

temiz:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
