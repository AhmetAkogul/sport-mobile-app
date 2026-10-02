package com.sporapp.backend.workoutsession.dto;

import java.math.BigDecimal;

import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Digits;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;

public record SetResultRequest(

        @NotNull(message = "Set numarası zorunlu")
        @Min(value = 1, message = "Set numarası 1 veya daha büyük olmalı")
        Integer setNumber,

        @NotNull(message = "Tekrar sayısı zorunlu")
        @Min(value = 0, message = "Tekrar sayısı negatif olamaz")
        @Max(value = 1000, message = "Tekrar sayısı en fazla 1000 olabilir")
        Integer reps,

        // null = vücut ağırlığı
        @DecimalMin(value = "0.0", message = "Ağırlık negatif olamaz")
        @Digits(integer = 4, fraction = 2, message = "Ağırlık en fazla 4 tam, 2 ondalık basamak olabilir")
        BigDecimal weight,

        // null = AI verisi yok
        @Min(value = 0, message = "Yanlış tekrar sayısı negatif olamaz")
        @Max(value = 1000, message = "Yanlış tekrar sayısı en fazla 1000 olabilir")
        Integer incorrectReps
) {}