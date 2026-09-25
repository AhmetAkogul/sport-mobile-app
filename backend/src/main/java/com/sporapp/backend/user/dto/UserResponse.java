package com.sporapp.backend.user.dto;

import com.sporapp.backend.user.Gender;
import com.sporapp.backend.user.User;

import java.math.BigDecimal;
import java.time.Instant; // utc zamanı gönderir

public record UserResponse(  // record = değişemez
        Long id,
        String name,
        String email,
        Integer age,
        Gender gender,
        BigDecimal height,
        BigDecimal weight,
        Instant createdAt // utc
) {

    public static UserResponse from(User user) {
        return new UserResponse(
                user.getId(),
                user.getName(),
                user.getEmail(),
                user.getAge(),
                user.getGender(),
                user.getHeight(),
                user.getWeight(),
                user.getCreatedAt()
        );
    }
} // sızmalara karşı dtoya password alanı eklemiyorum