package com.sporapp.backend.user.dto;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;

public record LoginRequest(

        @NotBlank(message = "E-posta boş olamaz")
        @Email(message = "Geçerli bir e-posta giriniz")
        String email,

        @NotBlank(message = "Şifre boş olamaz")
        String password
) {}