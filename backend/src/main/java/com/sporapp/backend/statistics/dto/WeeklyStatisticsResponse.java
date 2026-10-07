package com.sporapp.backend.statistics.dto;

import java.math.BigDecimal;
import java.time.LocalDate;

/**
 * Bir haftalik kova. weekStart her zaman pazartesidir (ISO-8601).
 * Bos haftalar da 0 degerleriyle doner; grafikte bosluk olusmasin.
 */
public record WeeklyStatisticsResponse(
        LocalDate weekStart,
        long sessions,
        long durationMinutes,
        BigDecimal calories
) {}
