package com.sporapp.backend.common.exception;

public class InvalidPasswordException extends RuntimeException {

    public InvalidPasswordException() {
        super("Mevcut şifre hatalı");
    }
}