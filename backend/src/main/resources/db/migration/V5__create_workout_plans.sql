CREATE TABLE workout_plans (
    id          BIGSERIAL    PRIMARY KEY,
    user_id     BIGINT       NOT NULL,
    name        VARCHAR(100) NOT NULL,
    description VARCHAR(500),
    created_at  TIMESTAMPTZ  NOT NULL,
    updated_at  TIMESTAMPTZ  NOT NULL,
    CONSTRAINT fk_workout_plans_user
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);

CREATE TABLE plan_exercises (
    id           BIGSERIAL   PRIMARY KEY,
    plan_id      BIGINT      NOT NULL,
    exercise_id  BIGINT      NOT NULL,
    position     INTEGER     NOT NULL,
    sets         INTEGER     NOT NULL,
    reps         INTEGER     NOT NULL,
    rest_seconds INTEGER     NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL,
    CONSTRAINT fk_plan_exercises_plan
        FOREIGN KEY (plan_id) REFERENCES workout_plans (id) ON DELETE CASCADE,
    CONSTRAINT fk_plan_exercises_exercise
        FOREIGN KEY (exercise_id) REFERENCES exercises (id),
    CONSTRAINT chk_plan_exercises_sets  CHECK (sets > 0),
    CONSTRAINT chk_plan_exercises_reps  CHECK (reps > 0),
    CONSTRAINT chk_plan_exercises_rest  CHECK (rest_seconds >= 0)
);