package com.sporapp.backend.statistics.dto;

import java.math.BigDecimal;
import java.time.Instant;

public record StatisticsOverviewResponse(
        long finishedSessions,
        long totalDurationMinutes,
        BigDecimal totalCalories,
        long totalSets,
        long totalReps,
        BigDecimal totalVolumeKg,
        Instant firstWorkoutAt,
        Instant lastWorkoutAt
) {}
