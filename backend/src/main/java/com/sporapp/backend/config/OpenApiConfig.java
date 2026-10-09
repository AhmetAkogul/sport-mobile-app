package com.sporapp.backend.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import io.swagger.v3.oas.models.Components;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.security.SecurityScheme;

/**
 * Swagger UI ayarlari. Erisim: http://localhost:8080/swagger-ui.html
 * Makine okunur sozlesme (mobil takim Postman/Insomnia'ya import edebilir):
 * http://localhost:8080/v3/api-docs
 */
@Configuration
public class OpenApiConfig {

    /** Controller'lardaki @SecurityRequirement bu sabiti kullanir; yazim hatasi olmaz. */
    public static final String GUVENLIK_SEMASI = "bearerAuth";

    @Bean
    public OpenAPI sporAppOpenApi() {
        return new OpenAPI()
                .info(new Info()
                        .title("Spor App API")
                        .version("1.0")
                        .description("Bitirme projesi spor uygulamasi backend'i. "
                                + "Korumali uç noktalar için önce /api/auth/login ile token al; "
                                + "sağ üstteki Authorize butonuna sadece token'ı yapıştır (Bearer yazma)."))
                // Semayi sadece TANIMLAR, global sart koymaz. Global sart koysaydik
                // herkese acik /api/auth/register ve /api/auth/login uzerinde de
                // kilit simgesi cikardi - yanlis izlenim.
                .components(new Components().addSecuritySchemes(GUVENLIK_SEMASI,
                        new SecurityScheme()
                                .type(SecurityScheme.Type.HTTP)
                                .scheme("bearer")
                                .bearerFormat("JWT")));
    }
}
