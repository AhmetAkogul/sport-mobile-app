CREATE TABLE exercise_sets (
    id                  BIGSERIAL   PRIMARY KEY,
    session_exercise_id BIGINT      NOT NULL,
    set_number          INTEGER     NOT NULL,
    reps                INTEGER     NOT NULL,
    weight              NUMERIC(6,2),
    incorrect_reps      INTEGER,
    created_at          TIMESTAMPTZ NOT NULL,
    updated_at          TIMESTAMPTZ NOT NULL,
    CONSTRAINT fk_exercise_sets_session_exercise
        FOREIGN KEY (session_exercise_id) REFERENCES session_exercises (id) ON DELETE CASCADE,
    CONSTRAINT chk_exercise_sets_set_number CHECK (set_number > 0),
    CONSTRAINT chk_exercise_sets_reps       CHECK (reps >= 0),
    CONSTRAINT chk_exercise_sets_weight     CHECK (weight IS NULL OR weight >= 0),
    CONSTRAINT chk_exercise_sets_incorrect  CHECK (incorrect_reps IS NULL
                                                   OR (incorrect_reps >= 0 AND incorrect_reps <= reps))
);