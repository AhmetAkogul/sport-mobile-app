package com.sporapp.backend.common.exception;

public class InvalidCredentialsException extends RuntimeException {

    public InvalidCredentialsException() {
        super("E-posta veya şifre hatalı");
    }
}

// Tek bir mesaj, iki farklı durum için: "email kayıtlı değil" ve "şifre yanlış". Bilinçli. Ayrı mesaj verseydik saldırgan hangi e-postaların kayıtlı olduğunu tek tek deneyerek öğrenirdi (user enumeration). Kullanıcıya da doğrusu bu — "şifren yanlış" demek, "bu e-posta var" bilgisini sızdırır.