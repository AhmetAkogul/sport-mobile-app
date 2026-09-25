package com.sporapp.backend.user.dto;

import java.math.BigDecimal;

import com.sporapp.backend.user.Gender;

import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Digits;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public record RegisterRequest(

        @NotBlank(message = "Ad boş olamaz")
        @Size(max = 100, message = "Ad en fazla 100 karakter olabilir")
        String name,

        @NotBlank(message = "E-posta boş olamaz") // not nulldan farkı null+boşluk kontrolu yapar
        @Email(message = "Geçerli bir e-posta giriniz")
        @Size(max = 255, message = "E-posta en fazla 255 karakter olabilir")
        String email,

        @NotBlank(message = "Şifre boş olamaz")
        @Size(min = 8, max = 72, message = "Şifre 8-72 karakter arasında olmalı") // bcrypt algoritmasında max 72 karakter oluyor
        String password,

        @Min(value = 1, message = "Yaş 1'den büyük olmalı")
        @Max(value = 119, message = "Yaş 119'dan küçük olmalı")
        Integer age,

        Gender gender,

        @DecimalMin(value = "0.0", inclusive = false, message = "Boy 0'dan büyük olmalı")
        @Digits(integer = 3, fraction = 2, message = "Boy en fazla 3 tam, 2 ondalık basamak olabilir") // @Digits(integer = 3, fraction = 2) = DB'deki NUMERIC(5,2) ile aynı sınır. Validation'ı DB kısıtına hizala.
        BigDecimal height,

        @DecimalMin(value = "0.0", inclusive = false, message = "Kilo 0'dan büyük olmalı")
        @Digits(integer = 3, fraction = 2, message = "Kilo en fazla 3 tam, 2 ondalık basamak olabilir")
        BigDecimal weight
) {}