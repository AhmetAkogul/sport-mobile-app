package com.sporapp.backend.user;

import java.util.Locale;

import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.sporapp.backend.common.exception.EmailAlreadyExistsException;
import com.sporapp.backend.user.dto.RegisterRequest;
import com.sporapp.backend.user.dto.UserResponse;

@Service
public class UserService {

    private final UserRepository userRepository; // dbdeki user işlemlerini yapmayı saglar*
    private final PasswordEncoder passwordEncoder; // pass encode (bcrypt)

    public UserService(UserRepository userRepository, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }

    @Transactional
    public UserResponse register(RegisterRequest request) { // gelen kayıt bilgilerini al → kontrol et → kullanıcı oluştur → DB'ye kaydet → response döndür
        String email = request.email().trim().toLowerCase(Locale.ROOT); // maili standarta cevir

        if (userRepository.existsByEmail(email)) { // EmailAlreadyExist kontrolü
            throw new EmailAlreadyExistsException(email);
        }

        User user = new User();
        user.setName(request.name().trim());
        user.setEmail(email);
        user.setPassword(passwordEncoder.encode(request.password()));
        user.setAge(request.age());
        user.setGender(request.gender());
        user.setHeight(request.height());
        user.setWeight(request.weight());

        User saved = userRepository.save(user); // dbye kaydet
        return UserResponse.from(saved); // userResponse.java ya return
    }
}