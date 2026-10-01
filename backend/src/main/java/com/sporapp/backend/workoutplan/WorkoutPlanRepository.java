package com.sporapp.backend.workoutplan;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import com.sporapp.backend.workoutplan.dto.WorkoutPlanSummaryResponse;

public interface WorkoutPlanRepository extends JpaRepository<WorkoutPlan, Long> {

    // sahiplik kontrolü: id VE user_id birlikte → başkasının planı "yok" sayılır
    Optional<WorkoutPlan> findByIdAndUserId(Long id, Long userId);

    // liste: plan + egzersiz SAYISI, tek sorgu (N+1 yok, egzersizler yüklenmiyor)
    @Query("""
            SELECT new com.sporapp.backend.workoutplan.dto.WorkoutPlanSummaryResponse(
                p.id, p.name, p.description, COUNT(pe), p.createdAt)
            FROM WorkoutPlan p
            LEFT JOIN p.exercises pe
            WHERE p.user.id = :userId
            GROUP BY p.id, p.name, p.description, p.createdAt
            ORDER BY p.createdAt DESC
            """)
    List<WorkoutPlanSummaryResponse> findSummariesByUserId(@Param("userId") Long userId);

    // detay: plan + egzersizler + her egzersizin bilgisi, tek sorgu
    @Query("""
            SELECT DISTINCT p FROM WorkoutPlan p
            LEFT JOIN FETCH p.exercises pe
            LEFT JOIN FETCH pe.exercise
            WHERE p.id = :id AND p.user.id = :userId
            """)
    Optional<WorkoutPlan> findDetailByIdAndUserId(@Param("id") Long id,
                                                  @Param("userId") Long userId);
}