package com.sporapp.backend.workoutplan.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;

public record AddPlanExerciseRequest(

        @NotNull(message = "Egzersiz id zorunlu")
        Long exerciseId,

        @NotNull(message = "Sıra zorunlu")
        @Min(value = 1, message = "Sıra 1 veya daha büyük olmalı")
        Integer position,

        @NotNull(message = "Set sayısı zorunlu")
        @Min(value = 1, message = "Set sayısı en az 1 olmalı")
        @Max(value = 100, message = "Set sayısı en fazla 100 olabilir")
        Integer sets,

        @NotNull(message = "Tekrar sayısı zorunlu")
        @Min(value = 1, message = "Tekrar sayısı en az 1 olmalı")
        @Max(value = 1000, message = "Tekrar sayısı en fazla 1000 olabilir")
        Integer reps,

        @NotNull(message = "Dinlenme süresi zorunlu")
        @Min(value = 0, message = "Dinlenme süresi negatif olamaz")
        @Max(value = 3600, message = "Dinlenme süresi en fazla 3600 saniye olabilir")
        Integer restSeconds
) {}