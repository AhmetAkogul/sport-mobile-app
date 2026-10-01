package com.sporapp.backend.workoutplan.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public record WorkoutPlanRequest(

        @NotBlank(message = "Plan adı boş olamaz")
        @Size(max = 100, message = "Plan adı en fazla 100 karakter olabilir")
        String name,

        @Size(max = 500, message = "Açıklama en fazla 500 karakter olabilir")
        String description
) {}