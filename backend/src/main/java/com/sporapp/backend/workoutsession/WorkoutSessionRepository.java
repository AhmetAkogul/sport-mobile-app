package com.sporapp.backend.workoutsession;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import com.sporapp.backend.workoutsession.dto.WorkoutSessionSummaryResponse;

public interface WorkoutSessionRepository extends JpaRepository<WorkoutSession, Long> {

    Optional<WorkoutSession> findByIdAndUserId(Long id, Long userId);

    @Query("""
            SELECT new com.sporapp.backend.workoutsession.dto.WorkoutSessionSummaryResponse(
                s.id, p.id, p.name, s.startedAt, s.finishedAt, COUNT(se), s.note, s.calories)
            FROM WorkoutSession s
            LEFT JOIN s.plan p
            LEFT JOIN s.exercises se
            WHERE s.user.id = :userId
            GROUP BY s.id, p.id, p.name, s.startedAt, s.finishedAt, s.note, s.calories
            ORDER BY s.startedAt DESC
            """)
    List<WorkoutSessionSummaryResponse> findSummariesByUserId(@Param("userId") Long userId);

    @Query("""
            SELECT DISTINCT s FROM WorkoutSession s
            LEFT JOIN FETCH s.plan
            LEFT JOIN FETCH s.exercises se
            LEFT JOIN FETCH se.exercise
            WHERE s.id = :id AND s.user.id = :userId
            """)
    Optional<WorkoutSession> findDetailByIdAndUserId(@Param("id") Long id,
                                                     @Param("userId") Long userId);
}