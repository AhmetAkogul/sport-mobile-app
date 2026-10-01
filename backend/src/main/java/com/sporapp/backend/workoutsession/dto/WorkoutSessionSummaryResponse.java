package com.sporapp.backend.workoutsession.dto;

import java.time.Instant;

public record WorkoutSessionSummaryResponse(
        Long id,
        Long planId,
        String planName,
        Instant startedAt,
        Instant finishedAt,
        Long exerciseCount,
        String note
) {}
