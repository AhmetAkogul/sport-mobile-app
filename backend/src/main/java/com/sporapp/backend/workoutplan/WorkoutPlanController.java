package com.sporapp.backend.workoutplan;

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

import com.sporapp.backend.workoutplan.dto.AddPlanExerciseRequest;
import com.sporapp.backend.workoutplan.dto.UpdatePlanExerciseRequest;
import com.sporapp.backend.workoutplan.dto.WorkoutPlanRequest;
import com.sporapp.backend.workoutplan.dto.WorkoutPlanResponse;
import com.sporapp.backend.workoutplan.dto.WorkoutPlanSummaryResponse;

import jakarta.validation.Valid;

@RestController
@RequestMapping("/api/workout-plans") // bu controller'ın tüm yolları /api/workout-plans ile başlar
public class WorkoutPlanController {

    private final WorkoutPlanService workoutPlanService;

    public WorkoutPlanController(WorkoutPlanService workoutPlanService) {
        this.workoutPlanService = workoutPlanService;
    }

    @PostMapping // POST /api/workout-plans
    public ResponseEntity<WorkoutPlanResponse> create(@AuthenticationPrincipal Long userId,
                                                      @Valid @RequestBody WorkoutPlanRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(workoutPlanService.create(userId, request));
    }

    @GetMapping // GET /api/workout-plans  (kendi planlarım, hafif liste)
    public ResponseEntity<List<WorkoutPlanSummaryResponse>> list(@AuthenticationPrincipal Long userId) {
        return ResponseEntity.ok(workoutPlanService.list(userId));
    }

    @GetMapping("/{planId}") // GET /api/workout-plans/5
    public ResponseEntity<WorkoutPlanResponse> getDetail(@AuthenticationPrincipal Long userId,
                                                         @PathVariable Long planId) {
        return ResponseEntity.ok(workoutPlanService.getDetail(userId, planId));
    }

    @PutMapping("/{planId}") // PUT /api/workout-plans/5
    public ResponseEntity<WorkoutPlanResponse> update(@AuthenticationPrincipal Long userId,
                                                      @PathVariable Long planId,
                                                      @Valid @RequestBody WorkoutPlanRequest request) {
        return ResponseEntity.ok(workoutPlanService.update(userId, planId, request));
    }

    @DeleteMapping("/{planId}") // DELETE /api/workout-plans/5
    public ResponseEntity<Void> delete(@AuthenticationPrincipal Long userId,
                                       @PathVariable Long planId) {
        workoutPlanService.delete(userId, planId);
        return ResponseEntity.noContent().build(); // 204
    }

    @PostMapping("/{planId}/exercises") // POST /api/workout-plans/5/exercises
    public ResponseEntity<WorkoutPlanResponse> addExercise(
            @AuthenticationPrincipal Long userId,
            @PathVariable Long planId,
            @Valid @RequestBody AddPlanExerciseRequest request) {
        return ResponseEntity.status(HttpStatus.CREATED)
                .body(workoutPlanService.addExercise(userId, planId, request));
    }

    @PutMapping("/{planId}/exercises/{planExerciseId}") // PUT /api/workout-plans/5/exercises/12
    public ResponseEntity<WorkoutPlanResponse> updateExercise(
            @AuthenticationPrincipal Long userId,
            @PathVariable Long planId,
            @PathVariable Long planExerciseId,
            @Valid @RequestBody UpdatePlanExerciseRequest request) {
        return ResponseEntity.ok(workoutPlanService.updateExercise(userId, planId, planExerciseId, request));
    }

    @DeleteMapping("/{planId}/exercises/{planExerciseId}") // DELETE /api/workout-plans/5/exercises/12
    public ResponseEntity<Void> removeExercise(@AuthenticationPrincipal Long userId,
                                               @PathVariable Long planId,
                                               @PathVariable Long planExerciseId) {
        workoutPlanService.removeExercise(userId, planId, planExerciseId);
        return ResponseEntity.noContent().build(); // 204
    }
}