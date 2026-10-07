package com.sporapp.backend.calories;

import java.math.BigDecimal;
import java.util.EnumMap;
import java.util.Map;

import com.sporapp.backend.exercise.Exercise;
import com.sporapp.backend.exercise.MuscleGroup;

/**
 * Egzersizin MET degerini bulur.
 * Once exercises.met_value kolonuna bakar; kolon bos ise kas grubuna gore varsayilan dondurur.
 * Varsayilanlar Compendium of Physical Activities'teki direnc/kardiyo araliklarindan secildi.
 */
public final class MetValues {

    private static final BigDecimal FALLBACK_MET = new BigDecimal("5.0");

    private static final Map<MuscleGroup, BigDecimal> BY_MUSCLE_GROUP =
            new EnumMap<>(MuscleGroup.class);

    static {
        BY_MUSCLE_GROUP.put(MuscleGroup.CHEST, new BigDecimal("5.0"));
        BY_MUSCLE_GROUP.put(MuscleGroup.BACK, new BigDecimal("5.0"));
        BY_MUSCLE_GROUP.put(MuscleGroup.LEGS, new BigDecimal("6.0"));
        BY_MUSCLE_GROUP.put(MuscleGroup.SHOULDERS, new BigDecimal("4.5"));
        BY_MUSCLE_GROUP.put(MuscleGroup.ARMS, new BigDecimal("3.5"));
        BY_MUSCLE_GROUP.put(MuscleGroup.CORE, new BigDecimal("3.5"));
        BY_MUSCLE_GROUP.put(MuscleGroup.CARDIO, new BigDecimal("8.0"));
        BY_MUSCLE_GROUP.put(MuscleGroup.FULL_BODY, new BigDecimal("6.5"));
    }

    private MetValues() {
    }

    public static BigDecimal of(Exercise exercise) {
        if (exercise.getMetValue() != null) {
            return exercise.getMetValue();
        }
        return BY_MUSCLE_GROUP.getOrDefault(exercise.getMuscleGroup(), FALLBACK_MET);
    }
}
