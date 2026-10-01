package com.sporapp.backend.workoutsession;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

public interface SessionExerciseRepository extends JpaRepository<SessionExercise, Long> {

    Optional<SessionExercise> findByIdAndSessionId(Long id, Long sessionId);
}