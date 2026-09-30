package com.sporapp.backend.user;

import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.sporapp.backend.user.dto.ChangePasswordRequest;
import com.sporapp.backend.user.dto.ProfileUpdateRequest;
import com.sporapp.backend.user.dto.UserResponse;

import jakarta.validation.Valid;

@RestController // HTTP requestleri karşılamak için
@RequestMapping("/api/users") // bu controller'ın tüm yolları /api/users ile başlar
public class UserController {

    private final UserService userService;

    public UserController(UserService userService) {
        this.userService = userService;
    }

    @GetMapping("/me") // GET /api/users/me
    public ResponseEntity<UserResponse> getProfile(@AuthenticationPrincipal Long userId) { // token'dan gelen kullanıcının id'si
        return ResponseEntity.ok(userService.getProfile(userId)); // 200 + profil
    }

    @PutMapping("/me") // PUT /api/users/me
    public ResponseEntity<UserResponse> updateProfile(@AuthenticationPrincipal Long userId,
                                                      @Valid @RequestBody ProfileUpdateRequest request) {
        return ResponseEntity.ok(userService.updateProfile(userId, request)); // 200 + güncel profil
    }

    @PutMapping("/me/password") // PUT /api/users/me/password
    public ResponseEntity<Void> changePassword(@AuthenticationPrincipal Long userId,
                                               @Valid @RequestBody ChangePasswordRequest request) {
        userService.changePassword(userId, request);
        return ResponseEntity.noContent().build(); // 204: gövde yok, işlem başarılı
    }
}