package com.sporapp.backend.workoutplan;

import com.sporapp.backend.common.BaseEntity;
import com.sporapp.backend.exercise.Exercise;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import lombok.Getter;
import lombok.Setter;

@Entity
@Table(name = "plan_exercises")
@Getter
@Setter
public class PlanExercise extends BaseEntity {

    @ManyToOne(fetch = FetchType.LAZY, optional = false) // hangi plana ait
    @JoinColumn(name = "plan_id", nullable = false)
    private WorkoutPlan plan;

    @ManyToOne(fetch = FetchType.LAZY, optional = false) // hangi egzersiz
    @JoinColumn(name = "exercise_id", nullable = false)
    private Exercise exercise;

    @Column(nullable = false)
    private Integer position; // plandaki sıra

    @Column(nullable = false)
    private Integer sets; // hedef set sayısı

    @Column(nullable = false)
    private Integer reps; // hedef tekrar sayısı

    @Column(name = "rest_seconds", nullable = false)
    private Integer restSeconds; // setler arası dinlenme (saniye)
}