-- created_at / updated_at: timezone'suz TIMESTAMP -> TIMESTAMPTZ
--
-- Neden: entity tarafi artik java.time.Instant kullaniyor. Instant her zaman
-- UTC'dir; karsilik gelen Postgres tipi timestamptz'dir. Hibernate
-- ddl-auto: validate ile tip uyusmazligini yakalar.
--
-- NOT: Mevcut satirlar (varsa) 'UTC' kabul edilerek donusturulur. Gelistirme
-- verisi oldugu icin kabul edilebilir; canli veri olsaydi sunucunun gercek
-- saat dilimi (ornegin 'Europe/Istanbul') kullanilmaliydi.
--
-- USING olmadan Postgres TIMESTAMP -> TIMESTAMPTZ donusumunu oturumun
-- TimeZone ayariyla yapar; belirsizlik olmasin diye acikca yaziyoruz.

ALTER TABLE users
    ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC',
    ALTER COLUMN updated_at TYPE TIMESTAMPTZ USING updated_at AT TIME ZONE 'UTC';
