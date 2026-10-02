package com.sporapp.backend.workoutsession.dto;

import java.util.List;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;

public record ExerciseResultRequest(

        @NotNull(message = "Seans egzersizi id zorunlu")
        Long sessionExerciseId,

        @NotNull(message = "Set listesi zorunlu")
        @Valid
        List<SetResultRequest> sets
) {}