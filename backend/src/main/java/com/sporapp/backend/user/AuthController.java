package com.sporapp.backend.user;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.sporapp.backend.user.dto.RegisterRequest;
import com.sporapp.backend.user.dto.UserResponse;

import jakarta.validation.Valid;

@RestController // HTTP requestleri karşılamak için restcontroller yazdık
@RequestMapping("/api/auth") // endpointleri belirler
public class AuthController {

    private final UserService userService;

    public AuthController(UserService userService) { // constructer userService verilerini alır
        this.userService = userService;
    }

    @PostMapping("/register") // HTTP post requesti gelince karşılayacak
    public ResponseEntity<UserResponse> register(@Valid @RequestBody RegisterRequest request) { // @requestbody = Frontend'den gelen JSON'u Java nesnesine çeviriyor., @valid aşağıda
        UserResponse response = userService.register(request); // HTTP isteği kaydedilmek üzere userServiceye gönderir
        return ResponseEntity.status(HttpStatus.CREATED).body(response); // HTTP response oluştur
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