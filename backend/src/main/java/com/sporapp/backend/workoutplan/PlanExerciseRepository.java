package com.sporapp.backend.workoutplan;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

public interface PlanExerciseRepository extends JpaRepository<PlanExercise, Long> {

    // planId de aranıyor: başka planın satırına erişilemez
    Optional<PlanExercise> findByIdAndPlanId(Long id, Long planId);
}