package com.sporapp.backend.exercise;

import com.sporapp.backend.common.BaseEntity;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.Table;
import lombok.Getter;
import lombok.Setter;

@Entity // Bu Java sınıfı database'deki bir tabloyu temsil ediyor
@Table(name = "exercises") // databasedeki tablo adını belirliyor
@Getter
@Setter
public class Exercise extends BaseEntity { // baseentity içindeki değişkenleri inheritence alıyor

    @Column(nullable = false, length = 100, unique = true)
    private String name;

    @Enumerated(EnumType.STRING)
    @Column(name = "muscle_group", nullable = false, length = 20)
    private MuscleGroup muscleGroup;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private Equipment equipment;

    @Column(length = 500)
    private String description;
}