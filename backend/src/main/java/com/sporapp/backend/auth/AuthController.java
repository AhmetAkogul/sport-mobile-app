package com.sporapp.backend.auth;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.sporapp.backend.user.UserService;
import com.sporapp.backend.user.dto.AuthResponse;
import com.sporapp.backend.user.dto.LoginRequest;
import com.sporapp.backend.user.dto.RegisterRequest;
import com.sporapp.backend.user.dto.UserResponse;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;

// @SecurityRequirement YOK: bu iki uç nokta herkese açık, kilit simgesi çıkmamalı.
@Tag(name = "Kimlik Doğrulama", description = "Kayıt ve giriş. Diğer tüm uç noktalar token ister.")
@RestController // HTTP requestleri karşılamak için restcontroller yazdık
@RequestMapping("/api/auth") // endpointleri belirler
public class AuthController {

    private final UserService userService;

    public AuthController(UserService userService) { // constructer userService verilerini alır
        this.userService = userService;
    }

    @Operation(summary = "Yeni kullanıcı kaydı", description = "Başarılı olursa 201 döner. E-posta zaten kayıtlıysa 409.")
    @PostMapping("/register") // HTTP post requesti gelince karşılayacak
    public ResponseEntity<UserResponse> register(@Valid @RequestBody RegisterRequest request) { // @requestbody = Frontend'den gelen JSON'u Java nesnesine çeviriyor., @valid aşağıda
        UserResponse response = userService.register(request); // HTTP isteği kaydedilmek üzere userServiceye gönderir
        return ResponseEntity.status(HttpStatus.CREATED).body(response); // HTTP response oluştur
    }

    @Operation(summary = "Giriş yap ve JWT token al",
            description = "Dönen token'ı sağ üstteki Authorize butonuna yapıştır. 'Bearer' kelimesini yazma, otomatik eklenir.")
    @PostMapping("/login")
    public ResponseEntity<AuthResponse> login(@Valid @RequestBody LoginRequest request) {
        return ResponseEntity.ok(userService.login(request));
    }
}

/*
 @valid = registerRequestten gelen verilerin doğruluğunu kontrol ediyor

 method, şeklinde olduğundan
 public record RegisterRequest(

    @NotBlank
    String name,

    @Email
    @NotBlank
    String email,

    @Size(min = 6)
    String password

) {}


{
    "name": "",
    "email": "abc",
    "password": "123"
}

frontendden böyle gelirse request reddeder

*/
