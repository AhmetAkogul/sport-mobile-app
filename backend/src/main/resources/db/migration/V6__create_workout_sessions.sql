CREATE TABLE workout_sessions (
    id          BIGSERIAL   PRIMARY KEY,
    user_id     BIGINT      NOT NULL,
    plan_id     BIGINT,
    started_at  TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ,
    note        VARCHAR(500),
    created_at  TIMESTAMPTZ NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL,
    CONSTRAINT fk_workout_sessions_user
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
    CONSTRAINT fk_workout_sessions_plan
        FOREIGN KEY (plan_id) REFERENCES workout_plans (id) ON DELETE SET NULL,
    CONSTRAINT chk_workout_sessions_sure
        CHECK (finished_at IS NULL OR finished_at >= started_at)
);

CREATE TABLE session_exercises (
    id                  BIGSERIAL   PRIMARY KEY,
    session_id          BIGINT      NOT NULL,
    exercise_id         BIGINT      NOT NULL,
    position            INTEGER     NOT NULL,
    target_sets         INTEGER,
    target_reps         INTEGER,
    target_rest_seconds INTEGER,
    created_at          TIMESTAMPTZ NOT NULL,
    updated_at          TIMESTAMPTZ NOT NULL,
    CONSTRAINT fk_session_exercises_session
        FOREIGN KEY (session_id) REFERENCES workout_sessions (id) ON DELETE CASCADE,
    CONSTRAINT fk_session_exercises_exercise
        FOREIGN KEY (exercise_id) REFERENCES exercises (id)
);