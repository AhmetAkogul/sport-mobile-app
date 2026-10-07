package com.sporapp.backend.statistics;

import java.math.BigDecimal;
import java.time.Instant;

/**
 * JPQL constructor expression icin ara projeksiyon: bitmis bir seansin ozet sayilari.
 * Entity'yi cekip tum set agacini yuklemek yerine sadece gereken 3 kolonu okur.
 */
public record SessionTiming(
        Instant startedAt,
        Instant finishedAt,
        BigDecimal calories
) {}
