package com.sporapp.backend.workoutsession;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import com.sporapp.backend.common.BaseEntity;
import com.sporapp.backend.user.User;
import com.sporapp.backend.workoutplan.WorkoutPlan;

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
@Table(name = "workout_sessions")
@Getter
@Setter
public class WorkoutSession extends BaseEntity {

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @ManyToOne(fetch = FetchType.LAZY) // NULL olabilir: plansiz antrenman
    @JoinColumn(name = "plan_id")
    private WorkoutPlan plan;

    @Column(name = "started_at", nullable = false)
    private Instant startedAt;

    @Column(name = "finished_at") // NULL = devam ediyor
    private Instant finishedAt;

    @Column(length = 500)
    private String note;

    // finish'te hesaplanip saklanir. NULL = hesaplanmadi
    // (seans bitmemis, kilo girilmemis veya ozellik gelmeden once bitmis)
    @Column(precision = 7, scale = 1)
    private BigDecimal calories;

    @OneToMany(mappedBy = "session", cascade = CascadeType.ALL, orphanRemoval = true)
    @OrderBy("position ASC, id ASC")
    private List<SessionExercise> exercises = new ArrayList<>();
}