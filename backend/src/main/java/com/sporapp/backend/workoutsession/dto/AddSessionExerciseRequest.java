package com.sporapp.backend.workoutsession.dto;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;

public record AddSessionExerciseRequest(

        @NotNull(message = "Egzersiz id zorunlu")
        Long exerciseId,

        @NotNull(message = "Sıra zorunlu")
        @Min(value = 1, message = "Sıra 1 veya daha büyük olmalı")
        Integer position,

        @Min(value = 1, message = "Hedef set en az 1 olmalı")
        @Max(value = 100, message = "Hedef set en fazla 100 olabilir")
        Integer targetSets,

        @Min(value = 1, message = "Hedef tekrar en az 1 olmalı")
        @Max(value = 1000, message = "Hedef tekrar en fazla 1000 olabilir")
        Integer targetReps,

        @Min(value = 0, message = "Hedef dinlenme negatif olamaz")
        @Max(value = 3600, message = "Hedef dinlenme en fazla 3600 saniye olabilir")
        Integer targetRestSeconds
) {}