package com.sporapp.backend.statistics.dto;

import java.math.BigDecimal;

import com.sporapp.backend.exercise.MuscleGroup;

/**
 * Ayni zamanda JPQL constructor expression hedefi (bkz. StatisticsRepository).
 * Bu yuzden alanlar ilkel degil sarmalayici tip: sorgu sonucu Long/BigDecimal doner.
 */
public record MuscleGroupStatisticsResponse(
        MuscleGroup muscleGroup,
        Long sets,
        Long reps,
        BigDecimal volumeKg
) {}
