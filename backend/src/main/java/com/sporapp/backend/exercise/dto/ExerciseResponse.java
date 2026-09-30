package com.sporapp.backend.exercise.dto;

import com.sporapp.backend.exercise.Equipment;
import com.sporapp.backend.exercise.Exercise;
import com.sporapp.backend.exercise.MuscleGroup;

public record ExerciseResponse(
        Long id,
        String name,
        MuscleGroup muscleGroup,
        Equipment equipment,
        String description
) {

    public static ExerciseResponse from(Exercise exercise) {
        return new ExerciseResponse(
                exercise.getId(),
                exercise.getName(),
                exercise.getMuscleGroup(),
                exercise.getEquipment(),
                exercise.getDescription()
        );
    }
}