CREATE TABLE users (
    id          BIGSERIAL     PRIMARY KEY,
    name        VARCHAR(100)  NOT NULL,
    email       VARCHAR(255)  NOT NULL,
    password    VARCHAR(255)  NOT NULL,
    age         INTEGER,
    gender      VARCHAR(20),
    height      NUMERIC(5,2),
    weight      NUMERIC(5,2),
    created_at  TIMESTAMP     NOT NULL,
    updated_at  TIMESTAMP     NOT NULL,
    CONSTRAINT uk_users_email UNIQUE (email),
    CONSTRAINT chk_users_age  CHECK (age IS NULL OR (age > 0 AND age < 120))
);