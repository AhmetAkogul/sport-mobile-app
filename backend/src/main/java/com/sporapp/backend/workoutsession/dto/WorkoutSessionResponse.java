package com.sporapp.backend.workoutsession.dto;

import java.time.Instant;
import java.util.List;

import com.sporapp.backend.workoutplan.WorkoutPlan;
import com.sporapp.backend.workoutsession.WorkoutSession;

public record WorkoutSessionResponse(
        Long id,
        Long planId,
        String planName,
        Instant startedAt,
        Instant finishedAt,
        String note,
        List<SessionExerciseResponse> exercises
) {

    public static WorkoutSessionResponse from(WorkoutSession session) {
        WorkoutPlan plan = session.getPlan(); // null olabilir
        return new WorkoutSessionResponse(
                session.getId(),
                plan == null ? null : plan.getId(),
                plan == null ? null : plan.getName(),
                session.getStartedAt(),
                session.getFinishedAt(),
                session.getNote(),
                session.getExercises().stream().map(SessionExerciseResponse::from).toList()
        );
    }
}