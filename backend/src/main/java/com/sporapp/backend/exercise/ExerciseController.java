package com.sporapp.backend.exercise;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.sporapp.backend.config.OpenApiConfig;
import com.sporapp.backend.exercise.dto.ExerciseResponse;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;

@Tag(name = "Egzersiz Kataloğu", description = "Salt okunur ortak katalog. Kullanıcıya özel değil.")
@SecurityRequirement(name = OpenApiConfig.GUVENLIK_SEMASI)
@RestController
@RequestMapping("/api/exercises") // bu controller'ın tüm yolları /api/exercises ile başlar
public class ExerciseController {

    private final ExerciseService exerciseService;

    public ExerciseController(ExerciseService exerciseService) {
        this.exerciseService = exerciseService;
    }

    @Operation(summary = "Egzersizleri listele", description = "Filtreler isteğe bağlı; ikisi birlikte de verilebilir.")
    @GetMapping // GET /api/exercises  (filtreler opsiyonel)
    public ResponseEntity<List<ExerciseResponse>> list(
            @RequestParam(required = false) MuscleGroup muscleGroup, // ?muscleGroup=CHEST
            @RequestParam(required = false) Equipment equipment) {   // ?equipment=DUMBBELL
        return ResponseEntity.ok(exerciseService.list(muscleGroup, equipment));
    }

    @Operation(summary = "Tek egzersiz getir")
    @GetMapping("/{id}") // GET /api/exercises/5
    public ResponseEntity<ExerciseResponse> getById(@PathVariable Long id) {
        return ResponseEntity.ok(exerciseService.getById(id));
    }
}
