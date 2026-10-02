package com.sporapp.backend.workoutsession.dto;

import java.util.List;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

public record FinishSessionRequest(

        @Size(max = 500, message = "Not en fazla 500 karakter olabilir")
        String note,

        @NotNull(message = "Egzersiz listesi zorunlu")
        @Valid
        List<ExerciseResultRequest> exercises
) {}