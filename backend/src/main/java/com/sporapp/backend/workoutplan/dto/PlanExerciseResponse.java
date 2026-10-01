package com.sporapp.backend.workoutplan.dto;

import com.sporapp.backend.exercise.Equipment;
import com.sporapp.backend.exercise.MuscleGroup;
import com.sporapp.backend.workoutplan.PlanExercise;

public record PlanExerciseResponse(
        Long id,
        Long exerciseId,
        String exerciseName,
        MuscleGroup muscleGroup,
        Equipment equipment,
        Integer position,
        Integer sets,
        Integer reps,
        Integer restSeconds
) {

    public static PlanExerciseResponse from(PlanExercise planExercise) {
        return new PlanExerciseResponse(
                planExercise.getId(),
                planExercise.getExercise().getId(),
                planExercise.getExercise().getName(),
                planExercise.getExercise().getMuscleGroup(),
                planExercise.getExercise().getEquipment(),
                planExercise.getPosition(),
                planExercise.getSets(),
                planExercise.getReps(),
                planExercise.getRestSeconds()
        );
    }
}