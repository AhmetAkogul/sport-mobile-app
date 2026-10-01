package com.sporapp.backend.workoutplan.dto;

import java.time.Instant;
import java.util.List;

import com.sporapp.backend.workoutplan.WorkoutPlan;

public record WorkoutPlanResponse(
        Long id,
        String name,
        String description,
        Instant createdAt,
        Instant updatedAt,
        List<PlanExerciseResponse> exercises
) {

    public static WorkoutPlanResponse from(WorkoutPlan plan) {
        return new WorkoutPlanResponse(
                plan.getId(),
                plan.getName(),
                plan.getDescription(),
                plan.getCreatedAt(),
                plan.getUpdatedAt(),
                plan.getExercises().stream().map(PlanExerciseResponse::from).toList()
        );
    }
}