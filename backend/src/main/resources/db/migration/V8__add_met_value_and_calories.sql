ALTER TABLE exercises
    ADD COLUMN met_value NUMERIC(3,1);

ALTER TABLE exercises
    ADD CONSTRAINT chk_exercises_met_value
        CHECK (met_value IS NULL OR met_value > 0);

ALTER TABLE workout_sessions
    ADD COLUMN calories NUMERIC(7,1);

ALTER TABLE workout_sessions
    ADD CONSTRAINT chk_workout_sessions_calories
        CHECK (calories IS NULL OR calories >= 0);
