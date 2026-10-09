package com.sporapp.backend.workoutsession;

import java.util.List;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.sporapp.backend.config.OpenApiConfig;
import com.sporapp.backend.workoutsession.dto.AddSessionExerciseRequest;
import com.sporapp.backend.workoutsession.dto.FinishSessionRequest;
import com.sporapp.backend.workoutsession.dto.StartSessionRequest;
import com.sporapp.backend.workoutsession.dto.WorkoutSessionResponse;
import com.sporapp.backend.workoutsession.dto.WorkoutSessionSummaryResponse;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;

@Tag(name = "Antrenman Seansları", description = "Seans başlatma, set sonuçlarını girme, bitirme. Bitmiş seansa egzersiz eklenemez (409).")
@SecurityRequirement(name = OpenApiConfig.GUVENLIK_SEMASI)
@RestController
@RequestMapping("/api/workout-sessions")
public class WorkoutSessionController {

    private final WorkoutSessionService workoutSessionService;

    public WorkoutSessionController(WorkoutSessionService workoutSessionService) {
        this.workoutSessionService = workoutSessionService;
    }

    @Operation(summary = "Seans başlat",
            description = "planId verilirse planın egzersizleri ve hedefleri kopyalanır (snapshot). planId null/boş = plansız antrenman.")
    @PostMapping // POST /api/workout-sessions   {"planId": 5}  veya  {}
    public ResponseEntity<WorkoutSessionResponse> start(@AuthenticationPrincipal Long userId,
                                                        @Valid @RequestBody StartSessionRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(workoutSessionService.start(userId, request));
    }

    @Operation(summary = "Geçmiş seansları listele", description = "Hafif liste: set detayı gelmez, sadece egzersiz sayısı.")
    @GetMapping // GET /api/workout-sessions  (gecmis, hafif liste)
    public ResponseEntity<List<WorkoutSessionSummaryResponse>> list(@AuthenticationPrincipal Long userId) {
        return ResponseEntity.ok(workoutSessionService.list(userId));
    }

    @Operation(summary = "Seans detayı", description = "Egzersizler ve girilen set sonuçları.")
    @GetMapping("/{sessionId}") // GET /api/workout-sessions/12
    public ResponseEntity<WorkoutSessionResponse> getDetail(@AuthenticationPrincipal Long userId,
                                                            @PathVariable Long sessionId) {
        return ResponseEntity.ok(workoutSessionService.getDetail(userId, sessionId));
    }

    @Operation(summary = "Seansı sil", description = "Gerçek silme. Setler ve seans egzersizleri de silinir (cascade).")
    @DeleteMapping("/{sessionId}") // DELETE /api/workout-sessions/12
    public ResponseEntity<Void> delete(@AuthenticationPrincipal Long userId,
                                       @PathVariable Long sessionId) {
        workoutSessionService.delete(userId, sessionId);
        return ResponseEntity.noContent().build();
    }

    @Operation(summary = "Seansa egzersiz ekle", description = "Bitmiş seansa eklenemez → 409.")
    @PostMapping("/{sessionId}/exercises") // POST /api/workout-sessions/12/exercises
    public ResponseEntity<WorkoutSessionResponse> addExercise(
            @AuthenticationPrincipal Long userId,
            @PathVariable Long sessionId,
            @Valid @RequestBody AddSessionExerciseRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(workoutSessionService.addExercise(userId, sessionId, request));
    }

    @Operation(summary = "Seanstan egzersiz çıkar", description = "Bitmiş seanstan çıkarılamaz → 409.")
    @DeleteMapping("/{sessionId}/exercises/{sessionExerciseId}")
    public ResponseEntity<Void> removeExercise(@AuthenticationPrincipal Long userId,
                                               @PathVariable Long sessionId,
                                               @PathVariable Long sessionExerciseId) {
        workoutSessionService.removeExercise(userId, sessionId, sessionExerciseId);
        return ResponseEntity.noContent().build();
    }

    @Operation(summary = "Seansı bitir: set sonuçlarını kaydet ve kaloriyi hesapla",
            description = """
                    Gönderilen seans egzersizlerinin setleri SİLİNİP yeniden yazılır; gönderilmeyenler dokunulmadan kalır.
                    incorrectReps null = AI verisi yok, 0 = kamerayla bakıldı ve hepsi doğruydu.
                    Kalori bitişte hesaplanıp saklanır; ikinci kez finish çağrılırsa finishedAt ve calories korunur (idempotent).""")
    @PutMapping("/{sessionId}/finish") // PUT /api/workout-sessions/12/finish
    public ResponseEntity<WorkoutSessionResponse> finish(
            @AuthenticationPrincipal Long userId,
            @PathVariable Long sessionId,
            @Valid @RequestBody FinishSessionRequest request) {
        return ResponseEntity.ok(workoutSessionService.finish(userId, sessionId, request));
    }
}
