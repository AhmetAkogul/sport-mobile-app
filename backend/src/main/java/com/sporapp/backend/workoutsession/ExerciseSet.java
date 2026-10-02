package com.sporapp.backend.workoutsession;

import java.math.BigDecimal;

import com.sporapp.backend.common.BaseEntity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import lombok.Getter;
import lombok.Setter;

@Entity
@Table(name = "exercise_sets")
@Getter
@Setter
public class ExerciseSet extends BaseEntity {

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "session_exercise_id", nullable = false)
    private SessionExercise sessionExercise;

    @Column(name = "set_number", nullable = false)
    private Integer setNumber;

    @Column(nullable = false)
    private Integer reps;

    // null = vücut ağırlığı (şınav, plank...)
    @Column(precision = 6, scale = 2)
    private BigDecimal weight;

    // null = AI verisi yok; 0 = kamerayla bakıldı, hepsi doğruydu
    @Column(name = "incorrect_reps")
    private Integer incorrectReps;
}