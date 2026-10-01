package com.sporapp.backend.workoutplan.dto;

import java.time.Instant;

public record WorkoutPlanSummaryResponse(
        Long id,
        String name,
        String description,
        Long exerciseCount,
        Instant createdAt
) {
}