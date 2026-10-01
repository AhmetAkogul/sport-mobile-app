package com.sporapp.backend.workoutplan;

import java.util.ArrayList;
import java.util.List;

import com.sporapp.backend.common.BaseEntity;
import com.sporapp.backend.user.User;

import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToMany;
import jakarta.persistence.OrderBy;
import jakarta.persistence.Table;
import lombok.Getter;
import lombok.Setter;

@Entity
@Table(name = "workout_plans")
@Getter
@Setter
public class WorkoutPlan extends BaseEntity { // id, createdAt, updatedAt BaseEntity'den

    @ManyToOne(fetch = FetchType.LAZY, optional = false) // planın sahibi
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Column(nullable = false, length = 100)
    private String name;

    @Column(length = 500)
    private String description;

    @OneToMany(mappedBy = "plan", cascade = CascadeType.ALL, orphanRemoval = true)
    @OrderBy("position ASC, id ASC") // plandaki sıra
    private List<PlanExercise> exercises = new ArrayList<>();
}