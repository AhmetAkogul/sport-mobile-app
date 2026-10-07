package com.sporapp.backend.statistics;

import java.math.BigDecimal;
import java.util.List;

import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.Repository;
import org.springframework.data.repository.query.Param;

import com.sporapp.backend.statistics.dto.MuscleGroupStatisticsResponse;
import com.sporapp.backend.workoutsession.WorkoutSession;

/**
 * Istatistik sorgulari. JpaRepository yerine sadece Repository:
 * burada yazma metodu yok, sadece okuma. Yanlislikla save/delete cagrilamaz.
 *
 * Tum sorgular "bitmis seans" filtreler: finishedAt IS NULL olan seansin
 * suresi ve kalorisi yoktur, istatistige girmemeli.
 */
public interface StatisticsRepository extends Repository<WorkoutSession, Long> {

    @Query("""
            SELECT new com.sporapp.backend.statistics.SessionTiming(
                s.startedAt, s.finishedAt, s.calories)
            FROM WorkoutSession s
            WHERE s.user.id = :userId AND s.finishedAt IS NOT NULL
            """)
    List<SessionTiming> findFinishedTimings(@Param("userId") Long userId);

    @Query("""
            SELECT COUNT(es)
            FROM WorkoutSession s
            JOIN s.exercises se
            JOIN se.sets es
            WHERE s.user.id = :userId AND s.finishedAt IS NOT NULL
            """)
    long countSets(@Param("userId") Long userId);

    // Toplam tekrar. Hem overview hem accuracy kullanir.
    @Query("""
            SELECT COALESCE(SUM(es.reps), 0)
            FROM WorkoutSession s
            JOIN s.exercises se
            JOIN se.sets es
            WHERE s.user.id = :userId AND s.finishedAt IS NOT NULL
            """)
    long sumReps(@Param("userId") Long userId);

    // Hacim: kaldirilan toplam kg. weight NULL olan setler (vucut agirligi)
    // SQL'de kendiliginden dusar: reps * NULL = NULL, SUM NULL'lari yok sayar.
    @Query("""
            SELECT COALESCE(SUM(es.reps * es.weight), 0)
            FROM WorkoutSession s
            JOIN s.exercises se
            JOIN se.sets es
            WHERE s.user.id = :userId AND s.finishedAt IS NOT NULL
            """)
    BigDecimal sumVolumeKg(@Param("userId") Long userId);

    // Sadece AI'in baktigi setlerin tekrari: incorrect_reps NULL = AI calismadi.
    // Kamerasiz yapilan set "dogru tekrar" sayilmamali.
    @Query("""
            SELECT COALESCE(SUM(es.reps), 0)
            FROM WorkoutSession s
            JOIN s.exercises se
            JOIN se.sets es
            WHERE s.user.id = :userId AND s.finishedAt IS NOT NULL
              AND es.incorrectReps IS NOT NULL
            """)
    long sumTrackedReps(@Param("userId") Long userId);

    // SUM, NULL degerleri kendiliginden atlar; AI verisi olmayan setler katkida bulunmaz.
    @Query("""
            SELECT COALESCE(SUM(es.incorrectReps), 0)
            FROM WorkoutSession s
            JOIN s.exercises se
            JOIN se.sets es
            WHERE s.user.id = :userId AND s.finishedAt IS NOT NULL
            """)
    long sumIncorrectReps(@Param("userId") Long userId);

    @Query("""
            SELECT new com.sporapp.backend.statistics.dto.MuscleGroupStatisticsResponse(
                e.muscleGroup, COUNT(es), COALESCE(SUM(es.reps), 0), COALESCE(SUM(es.reps * es.weight), 0))
            FROM WorkoutSession s
            JOIN s.exercises se
            JOIN se.exercise e
            JOIN se.sets es
            WHERE s.user.id = :userId AND s.finishedAt IS NOT NULL
            GROUP BY e.muscleGroup
            ORDER BY COUNT(es) DESC
            """)
    List<MuscleGroupStatisticsResponse> findMuscleGroupStats(@Param("userId") Long userId);
}
