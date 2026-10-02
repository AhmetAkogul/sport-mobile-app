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

import com.sporapp.backend.workoutsession.dto.AddSessionExerciseRequest;
import com.sporapp.backend.workoutsession.dto.FinishSessionRequest;
import com.sporapp.backend.workoutsession.dto.StartSessionRequest;
import com.sporapp.backend.workoutsession.dto.WorkoutSessionResponse;
import com.sporapp.backend.workoutsession.dto.WorkoutSessionSummaryResponse;

import jakarta.validation.Valid;

@RestController
@RequestMapping("/api/workout-sessions")
public class WorkoutSessionController {

    private final WorkoutSessionService workoutSessionService;

    public WorkoutSessionController(WorkoutSessionService workoutSessionService) {
        this.workoutSessionService = workoutSessionService;
    }

    @PostMapping // POST /api/workout-sessions   {"planId": 5}  veya  {}
    public ResponseEntity<WorkoutSessionResponse> start(@AuthenticationPrincipal Long userId,
                                                        @Valid @RequestBody StartSessionRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(workoutSessionService.start(userId, request));
    }

    @GetMapping // GET /api/workout-sessions  (gecmis, hafif liste)
    public ResponseEntity<List<WorkoutSessionSummaryResponse>> list(@AuthenticationPrincipal Long userId) {
        return ResponseEntity.ok(workoutSessionService.list(userId));
    }

    @GetMapping("/{sessionId}") // GET /api/workout-sessions/12
    public ResponseEntity<WorkoutSessionResponse> getDetail(@AuthenticationPrincipal Long userId,
                                                            @PathVariable Long sessionId) {
        return ResponseEntity.ok(workoutSessionService.getDetail(userId, sessionId));
    }

    @DeleteMapping("/{sessionId}") // DELETE /api/workout-sessions/12
    public ResponseEntity<Void> delete(@AuthenticationPrincipal Long userId,
                                       @PathVariable Long sessionId) {
        workoutSessionService.delete(userId, sessionId);
        return ResponseEntity.noContent().build();
    }

    @PostMapping("/{sessionId}/exercises") // POST /api/workout-sessions/12/exercises
    public ResponseEntity<WorkoutSessionResponse> addExercise(
            @AuthenticationPrincipal Long userId,
            @PathVariable Long sessionId,
            @Valid @RequestBody AddSessionExerciseRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(workoutSessionService.addExercise(userId, sessionId, request));
    }

    @DeleteMapping("/{sessionId}/exercises/{sessionExerciseId}")
    public ResponseEntity<Void> removeExercise(@AuthenticationPrincipal Long userId,
                                               @PathVariable Long sessionId,
                                               @PathVariable Long sessionExerciseId) {
        workoutSessionService.removeExercise(userId, sessionId, sessionExerciseId);
        return ResponseEntity.noContent().build();
    }

    @PutMapping("/{sessionId}/finish") // PUT /api/workout-sessions/12/finish
    public ResponseEntity<WorkoutSessionResponse> finish(
            @AuthenticationPrincipal Long userId,
            @PathVariable Long sessionId,
            @Valid @RequestBody FinishSessionRequest request) {
        return ResponseEntity.ok(workoutSessionService.finish(userId, sessionId, request));
    }
}