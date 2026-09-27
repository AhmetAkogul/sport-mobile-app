package com.sporapp.backend.user.dto;

import com.sporapp.backend.user.User;

public record AuthResponse(
        String token,
        String tokenType,
        long expiresIn,
        UserResponse user
) {

    public static AuthResponse bearer(String token, long expiresIn, User user) {
        return new AuthResponse(token, "Bearer", expiresIn, UserResponse.from(user));
    }
}