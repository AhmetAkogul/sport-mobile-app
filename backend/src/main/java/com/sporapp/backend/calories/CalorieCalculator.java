package com.sporapp.backend.calories;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Duration;
import java.time.Instant;
import java.util.List;

import com.sporapp.backend.exercise.Exercise;

/**
 * kcal = ortalama MET x 3.5 x kilo(kg) / 200 x dakika
 *
 * Formul MET tanimindan gelir: 1 MET = dinlenirken kilogram basina dakikada 3.5 ml oksijen.
 * 1 litre oksijen ~5 kcal enerji verir; 3.5 ml x 5 kcal / 1000 = 0.0175 kcal/kg/dk
 * ve 0.0175 = 3.5 / 200. Yani formul uydurma degil, MET taniminin acilimi.
 */
public final class CalorieCalculator {

    private CalorieCalculator() {
    }

    public static BigDecimal calculate(List<Exercise> exercises, Instant startedAt,
                                       Instant finishedAt, BigDecimal weightKg) {
        // kilo yok / seans bitmemis / egzersiz yok -> hesaplanamaz, null don
        if (weightKg == null || startedAt == null || finishedAt == null || exercises.isEmpty()) {
            return null;
        }

        long saniye = Duration.between(startedAt, finishedAt).getSeconds();
        if (saniye <= 0) {
            return null; // sifir saniyelik seans: bolme anlamsiz
        }

        double averageMet = exercises.stream()
                .map(MetValues::of)
                .mapToDouble(met -> met.doubleValue())
                .average()
                .orElse(0);

        double minutes = saniye / 60.0;
        double kcal = averageMet * 3.5 * weightKg.doubleValue() / 200.0 * minutes;

        return BigDecimal.valueOf(kcal).setScale(1, RoundingMode.HALF_UP);
    }
}
