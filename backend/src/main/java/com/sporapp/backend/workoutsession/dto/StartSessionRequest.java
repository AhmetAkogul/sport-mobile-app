package com.sporapp.backend.workoutsession.dto;

public record StartSessionRequest(
        Long planId // null = plansiz antrenman
) {}