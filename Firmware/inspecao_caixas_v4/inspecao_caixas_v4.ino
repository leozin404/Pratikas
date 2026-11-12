#include <Caixitas_inferencing.h>

/*#include <Caixas_inferencing.h>

/*
 * Inspeção de caixas v4 - ESP32-S3-CAM + Edge Impulse
 * Classes: aberta, fechada, amassada, vazio
 *
 * Como funciona:
 *  - A câmera e a rede neural rodam o tempo todo (~4 imagens por segundo)
 *  - Cada inferência é enviada ao monitor_caixas.html numa linha #INF
 *  - A cada INTERVALO_MS, o ESP tira a MÉDIA das imagens desse período
 *    e imprime UMA leitura, fácil de ler (some com LOG 0)
 *
 * Gravar pela Arduino IDE do PC: ESP32S3 Dev Module | Flash 16MB | PSRAM: OPI PSRAM
 *
 * Comandos (pelo monitor_caixas.html ou digitando no Monitor Serial + Enter):
 *   CAM 1 / CAM 0  liga/desliga o envio da imagem de entrada da rede a cada inferência
 *   SNAP           envia só a próxima imagem de entrada da rede
 *   FOTO           envia a foto original da câmera (JPEG 320x240) no formato do monitor
 *   LOG 1 / LOG 0  liga/desliga a leitura impressa a cada INTERVALO_MS
 *   PAUSE / RUN    pausa/retoma a inferência
 *   PING           responde com #HELLO (identificação do firmware)
 *   f              envia a foto atual (START_IMAGE ... END_IMAGE), como na v3
 */


#include "edge-impulse-sdk/dsp/image/image.hpp"
#include "esp_camera.h"

/* ============================ AJUSTES ============================ */
#define INTERVALO_MS  2500   // tempo entre uma leitura impressa e outra (ms)
#define LIMIAR        0.70f  // abaixo disso, o resultado aparece como INCERTO
#define PINO_ALERTA   -1     // LED/buzzer (-1 = desligado). Não use 4-18 nem 35-37.
#define MOSTRAR_TEMPO 0      // 1 = mostra também o tempo de cada etapa
#define BAUD          921600 // use o mesmo valor no monitor (na USB nativa não faz diferença)

/* ===================== PINOS (ESP32S3_EYE) ======================= */
#define PWDN_GPIO_NUM  -1
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM  15
#define SIOD_GPIO_NUM   4
#define SIOC_GPIO_NUM   5
#define Y2_GPIO_NUM    11
#define Y3_GPIO_NUM     9
#define Y4_GPIO_NUM     8
#define Y5_GPIO_NUM    10
#define Y6_GPIO_NUM    12
#define Y7_GPIO_NUM    18
#define Y8_GPIO_NUM    17
#define Y9_GPIO_NUM    16
#define VSYNC_GPIO_NUM  6
#define HREF_GPIO_NUM   7
#define PCLK_GPIO_NUM  13

#define CAM_W 320
#define CAM_H 240
#define N_CLASSES EI_CLASSIFIER_LABEL_COUNT

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_CAMERA
#error "Este modelo nao e de camera"
#endif

/* =========================== ESTADO ============================== */
static uint8_t *rgb_buf = nullptr;
static bool pedir_foto = false;

static float soma[N_CLASSES];         // soma das probabilidades no período
static int qtd_imagens = 0;           // imagens analisadas no período
static uint32_t ultimo_print = 0;
static uint32_t num_leitura = 0;
static uint32_t t_conv_ult = 0, t_nn_ult = 0;

// estado controlado pelo monitor
static bool dbg_cam = false;     // enviar imagem a cada inferência
static bool dbg_snap = false;    // enviar só a próxima imagem
static bool dbg_jpeg = false;    // enviar a próxima foto original em JPEG
static bool dbg_log = true;      // imprimir a leitura a cada INTERVALO_MS
static bool dbg_paused = false;  // inferência pausada

/* ===================== PROTOCOLO DO MONITOR ====================== */
static void dbg_hello() {
  Serial.print("#HELLO {\"proj\":\"");
#ifdef EI_CLASSIFIER_PROJECT_NAME
  Serial.print(EI_CLASSIFIER_PROJECT_NAME);
#else
  Serial.print("caixas");
#endif
  Serial.printf("\",\"w\":%d,\"h\":%d,\"labels\":[", EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT);
  for (int i = 0; i < N_CLASSES; i++) {
    Serial.printf("%s\"%s\"", i ? "," : "", ei_classifier_inferencing_categories[i]);
  }
  Serial.printf("],\"cam\":%s,\"log\":%s,\"paused\":%s}\n",
                dbg_cam ? "true" : "false", dbg_log ? "true" : "false", dbg_paused ? "true" : "false");
}

static void zerar_media() {
  for (int i = 0; i < N_CLASSES; i++) soma[i] = 0;
  qtd_imagens = 0;
  ultimo_print = millis();
}

static void executar_comando(char *c) {
  if (strcmp(c, "f") == 0) { pedir_foto = true; return; }   // comando antigo, mantido igual

  for (char *p = c; *p; p++) *p = toupper(*p);
  bool ok = true;
  if      (!strcmp(c, "CAM 1")) dbg_cam = true;
  else if (!strcmp(c, "CAM 0")) dbg_cam = false;
  else if (!strcmp(c, "SNAP"))  dbg_snap = true;
  else if (!strcmp(c, "FOTO"))  dbg_jpeg = true;
  else if (!strcmp(c, "LOG 1")) dbg_log = true;
  else if (!strcmp(c, "LOG 0")) dbg_log = false;
  else if (!strcmp(c, "PAUSE")) dbg_paused = true;
  else if (!strcmp(c, "RUN"))   { dbg_paused = false; zerar_media(); }
  else if (!strcmp(c, "PING"))  { Serial.println("#ACK PING"); dbg_hello(); return; }
  else ok = false;

  if (ok) Serial.printf("#ACK %s\n", c);
  else    Serial.printf("#ERR comando desconhecido: %s\n", c);
}

static void ler_comandos() {
  static char buf[40];
  static uint8_t n = 0;
  while (Serial.available()) {
    char ch = (char)Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (n == 0) continue;
      buf[n] = '\0';
      n = 0;
      executar_comando(buf);
    } else if (n < sizeof(buf) - 1) {
      buf[n++] = ch;
    }
  }
}

// Uma linha #INF por inferência (o monitor faz a própria estabilização)
static void enviar_resultado(const ei_impulse_result_t &r) {
  Serial.print("#INF {\"p\":{");
  for (int i = 0; i < N_CLASSES; i++) {
    Serial.printf("%s\"%s\":%.4f", i ? "," : "", r.classification[i].label, r.classification[i].value);
  }
  // "dsp" = conversão JPEG->RGB + redimensionamento + DSP do Edge Impulse
  Serial.printf("},\"dsp\":%lu,\"cls\":%d", (unsigned long)(t_conv_ult + r.timing.dsp), (int)r.timing.classification);
#if EI_CLASSIFIER_HAS_ANOMALY
  Serial.printf(",\"anom\":%.3f", r.anomaly);
#endif
  Serial.printf(",\"heap\":%u,\"temp\":%.1f}\n", (unsigned)ESP.getFreeHeap(), temperatureRead());
}

// Cabeçalho #IMG + bytes crus. fmt: "bgr888" para o rgb_buf, "jpeg" para o quadro da câmera
static void enviar_imagem(const uint8_t *buf, int w, int h, const char *fmt, size_t len) {
  Serial.printf("#IMG %d %d %s %u\n", w, h, fmt, (unsigned)len);
  Serial.write(buf, len);
  Serial.write('\n');
  Serial.flush();
}

static void enviar_entrada_rede() {
  // o rgb_buf fica em BGR (é o que o fmt2rgb888 entrega); o ler_pixels troca a ordem para a rede
  enviar_imagem(rgb_buf, EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT, "bgr888",
                (size_t)EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * 3);
  dbg_snap = false;
}

/* ============================ CÂMERA ============================= */
static bool camera_iniciar() {
  camera_config_t c = {};
  c.pin_pwdn = PWDN_GPIO_NUM;     c.pin_reset = RESET_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_sscb_sda = SIOD_GPIO_NUM; c.pin_sscb_scl = SIOC_GPIO_NUM;
  c.pin_d7 = Y9_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d4 = Y6_GPIO_NUM;
  c.pin_d3 = Y5_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d0 = Y2_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;   c.pin_href = HREF_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.ledc_timer = LEDC_TIMER_0;    c.ledc_channel = LEDC_CHANNEL_0;
  c.pixel_format = PIXFORMAT_JPEG;
  c.frame_size   = FRAMESIZE_QVGA;
  c.jpeg_quality = 12;
  c.fb_count     = 2;
  c.fb_location  = CAMERA_FB_IN_PSRAM;
  c.grab_mode    = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) {
    Serial.printf("ERRO ao iniciar a camera (0x%x)\n", err);
    return false;
  }
  return true;
}

static bool capturar() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return false;

  if (pedir_foto) {
    pedir_foto = false;
    Serial.println("START_IMAGE");
    Serial.write(fb->buf, fb->len);
    Serial.println("END_IMAGE");
  }
  if (dbg_jpeg) {
    dbg_jpeg = false;
    enviar_imagem(fb->buf, fb->width, fb->height, "jpeg", fb->len);
  }

  uint32_t t0 = millis();
  bool ok = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, rgb_buf);
  esp_camera_fb_return(fb);
  if (!ok) return false;

  ei::image::processing::crop_and_interpolate_rgb888(
      rgb_buf, CAM_W, CAM_H, rgb_buf, EI_CLASSIFIER_INPUT_WIDTH, EI_CLASSIFIER_INPUT_HEIGHT);
  t_conv_ult = millis() - t0;
  return true;
}

static int ler_pixels(size_t offset, size_t length, float *out) {
  size_t p = offset * 3;
  for (size_t i = 0; i < length; i++, p += 3) {
    out[i] = (float)((rgb_buf[p + 2] << 16) | (rgb_buf[p + 1] << 8) | rgb_buf[p]);
  }
  return 0;
}

/* ========================= IMPRESSÃO ============================= */
// Nome bonito para cada classe
static const char *nome_exibicao(const char *rotulo) {
  if (strcmp(rotulo, "aberta") == 0)   return "CAIXA ABERTA";
  if (strcmp(rotulo, "fechada") == 0)  return "CAIXA FECHADA";
  if (strcmp(rotulo, "amassada") == 0) return "CAIXA AMASSADA";
  if (strcmp(rotulo, "vazio") == 0)    return "SEM CAIXA";
  return rotulo;
}

static void alerta(bool ligado) {
#if PINO_ALERTA >= 0
  digitalWrite(PINO_ALERTA, ligado ? HIGH : LOW);
#else
  (void)ligado;
#endif
}

static void imprimir_leitura() {
  float media[N_CLASSES];
  int melhor = 0;
  for (int i = 0; i < N_CLASSES; i++) {
    media[i] = soma[i] / qtd_imagens;
    if (media[i] > media[melhor]) melhor = i;
  }
  const char *rotulo = ei_classifier_inferencing_categories[melhor];
  float certeza = media[melhor];
  bool confiavel = certeza >= LIMIAR;
  bool defeito = confiavel && (strcmp(rotulo, "aberta") == 0 || strcmp(rotulo, "amassada") == 0);

  num_leitura++;
  alerta(defeito);          // o alerta físico funciona mesmo com LOG 0
  if (!dbg_log) return;

  Serial.println();
  Serial.println("========================================");
  Serial.printf(" LEITURA %lu   (media de %d imagens)\n", (unsigned long)num_leitura, qtd_imagens);
  Serial.println("----------------------------------------");

  if (confiavel) {
    Serial.printf(" RESULTADO:  %s\n", nome_exibicao(rotulo));
    Serial.printf(" CERTEZA:    %d%%\n", (int)(certeza * 100 + 0.5f));
  } else {
    Serial.println(" RESULTADO:  INCERTO");
    Serial.printf(" PALPITE:    %s (%d%%)\n", nome_exibicao(rotulo), (int)(certeza * 100 + 0.5f));
  }

  // Barras com a probabilidade de cada classe
  Serial.println();
  for (int i = 0; i < N_CLASSES; i++) {
    int pct = (int)(media[i] * 100 + 0.5f);
    Serial.printf("   %-9s %3d%%  ", ei_classifier_inferencing_categories[i], pct);
    for (int b = 0; b < pct / 5; b++) Serial.print('#');
    Serial.println();
  }
  Serial.println();

  if (!confiavel)                          Serial.println(" STATUS:     >> verificar manualmente <<");
  else if (defeito)                        Serial.println(" STATUS:     !!! ALERTA: CAIXA COM DEFEITO !!!");
  else if (strcmp(rotulo, "vazio") == 0)   Serial.println(" STATUS:     esteira livre");
  else                                     Serial.println(" STATUS:     OK");

#if MOSTRAR_TEMPO
  Serial.printf(" (tempo por imagem: conversao %lu ms, rede neural %lu ms)\n",
                (unsigned long)t_conv_ult, (unsigned long)t_nn_ult);
#endif
  Serial.println("========================================");
}

/* ============================ SETUP ============================== */
void setup() {
  Serial.begin(BAUD);
  delay(1000);
  Serial.println("\n=== Inspecao de caixas v4 ===");

  if (!psramFound()) {
    Serial.println("ERRO: PSRAM nao encontrada.");
    Serial.println("Grave pela Arduino IDE do PC com Ferramentas > PSRAM = OPI PSRAM.");
    while (true) delay(1000);
  }

#if PINO_ALERTA >= 0
  pinMode(PINO_ALERTA, OUTPUT);
  alerta(false);
#endif

  if (!camera_iniciar()) {
    Serial.println("Confira o cabo flat da camera.");
    while (true) delay(1000);
  }

  rgb_buf = (uint8_t *)ps_malloc(CAM_W * CAM_H * 3);
  if (!rgb_buf) {
    Serial.println("ERRO: sem memoria para a imagem");
    while (true) delay(1000);
  }

  // Descarta os primeiros frames enquanto a exposição automática estabiliza
  for (int i = 0; i < 10; i++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
    delay(50);
  }

  Serial.printf("Pronto. Uma leitura a cada %.1f s.\n", INTERVALO_MS / 1000.0f);
  dbg_hello();
  ultimo_print = millis();
}

/* ============================= LOOP ============================== */
void loop() {
  ler_comandos();

  // Pausado: não roda a rede, mas ainda atende pedidos de imagem
  if (dbg_paused) {
    if (dbg_snap || dbg_jpeg || pedir_foto) {
      if (capturar() && dbg_snap) enviar_entrada_rede();
    }
    delay(20);
    return;
  }

  if (!capturar()) {
    Serial.println("ERRO na captura da imagem");
    delay(50);
    return;
  }

  ei::signal_t sinal;
  sinal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
  sinal.get_data = &ler_pixels;

  ei_impulse_result_t res = { 0 };
  if (run_classifier(&sinal, &res, false) != EI_IMPULSE_OK) {
    Serial.println("ERRO na rede neural");
    return;
  }
  t_nn_ult = res.timing.classification;

  enviar_resultado(res);
  if (dbg_cam || dbg_snap) enviar_entrada_rede();   // mesma imagem que a rede acabou de ver

  // Acumula as probabilidades para tirar a média
  for (int i = 0; i < N_CLASSES; i++) soma[i] += res.classification[i].value;
  qtd_imagens++;

  // Hora de imprimir?
  if (millis() - ultimo_print >= INTERVALO_MS) {
    imprimir_leitura();
    zerar_media();
  }
}
