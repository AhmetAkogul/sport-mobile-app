package com.sporapp.backend.common;

import java.time.Instant;

import jakarta.persistence.Column;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.MappedSuperclass;
import jakarta.persistence.PrePersist;
import jakarta.persistence.PreUpdate;
import lombok.Getter;
import lombok.Setter;

@MappedSuperclass // Bu sınıfı tek başına bir database tablosu olarak oluşturma. Ama bundan miras alan Entity'lerin içine bu alanları dahil et.
@Getter
@Setter
public abstract class BaseEntity { // abstract : BaseEntity doğrudan oluşturulmak için değil, miras alınmak için var.

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY) // oto id üretir
    private Long id;

    @Column(name = "created_at", nullable = false, updatable = false) // updateable = oluştuktan sonra güncellenemez
    private Instant createdAt;

    @Column(name = "updated_at", nullable = false) // nullable = null olamaz
    private Instant updatedAt;

    @PrePersist // Entity database'e ilk kez kaydedilmeden hemen önce bu metodu çalıştır.
    protected void onCreate() {
        this.createdAt = Instant.now();
        this.updatedAt = this.createdAt;
    }

    @PreUpdate // Entity database'de update edilmeden hemen önce bu metodu çalıştır.
    protected void onUpdate() {
        this.updatedAt = Instant.now();
    }
}