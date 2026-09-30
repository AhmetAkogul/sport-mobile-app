CREATE TABLE exercises (
    id           BIGSERIAL PRIMARY KEY,
    name         VARCHAR(100) NOT NULL UNIQUE,
    muscle_group VARCHAR(20)  NOT NULL,
    equipment    VARCHAR(20)  NOT NULL,
    description  VARCHAR(500),
    created_at   TIMESTAMPTZ  NOT NULL,
    updated_at   TIMESTAMPTZ  NOT NULL
);