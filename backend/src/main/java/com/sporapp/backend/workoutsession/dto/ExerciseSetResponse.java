package com.sporapp.backend.workoutsession.dto;

import java.math.BigDecimal;

import com.sporapp.backend.workoutsession.ExerciseSet;

public record ExerciseSetResponse(
        Long id,
        Integer setNumber,
        Integer reps,
        BigDecimal weight,
        Integer incorrectReps
) {

    public static ExerciseSetResponse from(ExerciseSet set) {
        return new ExerciseSetResponse(
                set.getId(),
                set.getSetNumber(),
                set.getReps(),
                set.getWeight(),
                set.getIncorrectReps()
        );
    }
}