# -*- coding: utf-8 -*-
r"""
Asama 7B - Istatistik endpointleri testi.

Calistirmadan once backend ayakta olmali:  .\mvnw.cmd spring-boot:run

NOT: 3. bolumde 65 saniye beklenir. Sebep: sure istatistigi DAKIKA bazinda
ve Java Duration.toMinutes() asagi yuvarlar; 4 saniyelik seans 0 dakika eder.
Gercek bir "1 dakika" kaniti icin gercekten 1 dakikadan fazla beklemek gerekir.
"""

import datetime
import json
import re
import sys
import time
import urllib.error
import urllib.request
from decimal import ROUND_HALF_UP, Decimal

TABAN = "http://localhost:8080"
GECEN = 0
KALAN = 0


