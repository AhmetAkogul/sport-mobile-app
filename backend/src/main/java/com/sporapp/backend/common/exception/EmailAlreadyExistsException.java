package com.sporapp.backend.common.exception;

public class EmailAlreadyExistsException extends RuntimeException {

    public EmailAlreadyExistsException(String email) {
        super("Bu e-posta zaten kayıtlı: " + email);
    }
}