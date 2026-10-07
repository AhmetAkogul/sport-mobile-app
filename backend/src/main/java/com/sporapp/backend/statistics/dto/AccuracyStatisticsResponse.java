package com.sporapp.backend.statistics.dto;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * AI hareket dogrulugu.
 *
 * trackedReps = AI'in baktigi setlerin tekrari. Kamerasiz yapilan setler
 * buraya GIRMEZ; aksi halde "AI hic calismadi" durumu "%100 dogru" gibi gorunurdu.
 *
 * correctReps = trackedReps - incorrectReps. Cikarma guvenli: exercise_sets
 * tablosundaki chk_exercise_sets_incorrect kisiti her sette
 * incorrect_reps <= reps olmasini garanti eder, toplam negatife dusemez.
 *
 * accuracyPercent, AI hic calismadiysa null doner - "%0 dogru" demek yaniltirdi.
 */
public record AccuracyStatisticsResponse(
        long totalReps,
        long trackedReps,
        long correctReps,
        long incorrectReps,
        BigDecimal accuracyPercent
) {

    public static AccuracyStatisticsResponse of(long totalReps, long trackedReps,
                                                long incorrectReps) {
        long correctReps = trackedReps - incorrectReps;

        BigDecimal accuracyPercent = trackedReps == 0
                ? null
                : BigDecimal.valueOf(correctReps * 100.0 / trackedReps)
                        .setScale(1, RoundingMode.HALF_UP);

        return new AccuracyStatisticsResponse(
                totalReps, trackedReps, correctReps, incorrectReps, accuracyPercent);
    }
}
