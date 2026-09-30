package com.sporapp.backend.user.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public record ChangePasswordRequest(

        @NotBlank(message = "Mevcut şifre boş olamaz")
        String currentPassword,

        @NotBlank(message = "Yeni şifre boş olamaz")
        @Size(min = 8, max = 20, message = "Şifre 8-20 karakter arasında olmalı")
        String newPassword
) {}