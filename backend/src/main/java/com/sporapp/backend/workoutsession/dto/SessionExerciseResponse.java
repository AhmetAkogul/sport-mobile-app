package com.sporapp.backend.workoutsession.dto;

import com.sporapp.backend.exercise.Equipment;
import com.sporapp.backend.exercise.MuscleGroup;
import com.sporapp.backend.workoutsession.SessionExercise;

public record SessionExerciseResponse(
        Long id,
        Long exerciseId,
        String exerciseName,
        MuscleGroup muscleGroup,
        Equipment equipment,
        Integer position,
        Integer targetSets,
        Integer targetReps,
        Integer targetRestSeconds
) {

    public static SessionExerciseResponse from(SessionExercise sessionExercise) {
        return new SessionExerciseResponse(
                sessionExercise.getId(),
                sessionExercise.getExercise().getId(),
                sessionExercise.getExercise().getName(),
                sessionExercise.getExercise().getMuscleGroup(),
                sessionExercise.getExercise().getEquipment(),
                sessionExercise.getPosition(),
                sessionExercise.getTargetSets(),
                sessionExercise.getTargetReps(),
                sessionExercise.getTargetRestSeconds()
        );
    }
}