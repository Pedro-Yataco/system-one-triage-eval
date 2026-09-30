# Dataset sintético de escalamiento (CECOM)

Correos que llegan al Centro de Comando (CECOM). La tarea es decidir cuáles se **escalan a una persona**. Solo cuentan tres razones de escalamiento: actualizar un ticket existente, hacerle seguimiento o pedir varias gestiones sobre tickets.

Todo el contenido es **ficticio**. Se redactó a mano tomando como inspiración 341 correos reales (`temp/DatasetMB.xlsx`, que no se versiona). Personas, empresas, proveedor, IPs, hosts, usuarios y números de ticket son inventados. Se verificó que ninguna IP, usuario, host ni dominio del original aparezca en el dataset.

## Archivos

| Archivo | Registros | Uso |
|---|---|---|
| `principal.csv` | 300 | Medir. No se ajusta nada mirando sus resultados. |
| `ajuste.csv` | 60 | Iterar las descripciones de las opciones y cualquier otro ajuste. |

Ambos tienen la misma composición. Salieron de un reparto al azar (semilla fija) dentro de cada subtipo.

## Columnas

| Columna | ¿La ve el modelo? | Contenido |
|---|---|---|
| `id` | no | `P001`–`P300` en principal, `A01`–`A60` en ajuste |
| `texto` | **sí** | `Asunto: …` y `Cuerpo: …` en un solo string, ya limpio: solo la esencia del pedido, sin firmas, hilos ni avisos legales |
| `accion` | no | `escalar`, `clasificar` o `descartar` |
| `casuistica` | no (es la etiqueta) | `ACTUALIZACION_TICKET`, `SEGUIMIENTO_TICKET`, `MULTI_GESTION` u `OTROS` |
| `subtipo` | no | Solo en `OTROS`: `NUEVA_SOLICITUD`, `NOTIFICACION`, `CONFIRMACION` o `CIERRE_CANCELACION` |
| `dificultad` | no | `directa`, `indirecta` o `engañosa` (ver abajo) |
| `referencias` | no | `ID FLUJO` de los correos originales que inspiraron el registro, separados por `;` |

## Cómo se plantea la tarea

El modelo responde **una sola pregunta `choice`** con cuatro opciones. Se escala cuando la respuesta no es `OTROS`, así que P(escalar) = 1 − P(`OTROS`).

- **Pocas opciones:** tener solo cuatro favorece la precisión del System One.
- **Coherencia:** la decisión binaria y la razón salen de la misma distribución, así que no pueden contradecirse.
- **Descripciones de las opciones:** son las de `configs/escalamiento.yaml`. Están redactadas en positivo, porque las negaciones se interpretan peor.

## Casuísticas

| Casuística | `accion` | Qué incluye | Señales típicas |
|---|---|---|---|
| `ACTUALIZACION_TICKET` | escalar | Actualizar, modificar, reasignar o derivar un ticket existente; agregarle información; **reabrirlo** porque el problema persiste | "favor actualizar ticket a @…", "es una actualización del tk", "agregar al ticket…", "derivar al de turno de…", "reabrir el ticket" |
| `SEGUIMIENTO_TICKET` | escalar | Consultar el estado, el número o la referencia de un ticket o solicitud existente; reclamar que no hay respuesta; dar continuidad sin pedir trabajo nuevo | "status del ticket", "favor de enviar el número de ticket", "favor indicar el ticket registrado", "no se tiene respuesta aún", "amable recordatorio" |
| `MULTI_GESTION` | escalar | Más de una acción sobre tickets: crear, actualizar, cerrar o cancelar varios, o combinarlas | "generar un ticket de Solicitud y cancelar el ticket…", "crear 2 tickets y derivar a…", "generar los siguientes tickets: 1…, 2…" |
| `OTROS` | clasificar / descartar | Todo lo demás (ver subtipos) | |

**Subtipos de `OTROS`:**

| Subtipo | `accion` | Qué es |
|---|---|---|
| `NUEVA_SOLICITUD` | clasificar | Trabajo técnico nuevo para **un** equipo, aunque cite un ticket del cliente (`#SR-…`, `RITM…`) o diga "generar ticket a…" |
| `NOTIFICACION` | descartar | Informa algo ya hecho o por hacer: "se ha generado el TK #…", avisos de mantenimiento, "favor de no generar incidencia", respuestas automáticas |
| `CONFIRMACION` | descartar | Agradecimientos, conformidades, aprobaciones ("Proceder", "Validado") |
| `CIERRE_CANCELACION` | descartar | Cerrar o cancelar **un solo** ticket |

## Reglas para los casos límite

1. **Citar un ticket no basta.** Si pide trabajo técnico nuevo, es `NUEVA_SOLICITUD`, aunque diga "con respecto al ticket #SR-…". Si solo aporta datos a ese ticket, es `ACTUALIZACION_TICKET`. Si solo pregunta por él, es `SEGUIMIENTO_TICKET`.
2. **Las palabras engañan.** "Actualizar", "estado" o "seguimiento" referidos a software, datos de usuario o reuniones no son señal de escalamiento.
3. **Uno vs. varios.** Cerrar o cancelar un solo ticket es `OTROS`. Dos o más acciones sobre tickets es `MULTI_GESTION`; eso incluye cerrar o cancelar varios, o cerrar uno y reabrir otro. Si el trabajo nuevo necesita tickets para más de un equipo, también es `MULTI_GESTION`; si son varios pasos para un mismo equipo, es `OTROS`.
4. **Informar no es pedir.** Avisar que un ticket se generó, se reasignó, se resolvió o se canceló es `NOTIFICACION`.
5. **Reabrir.** Si el mismo problema sigue en un ticket resuelto o cerrado y se pide retomarlo, es `ACTUALIZACION_TICKET`. Si es un problema distinto, es `NUEVA_SOLICITUD`.
6. **El destinatario no influye.** Las menciones a CECOM o a personas aparecen con frecuencia similar en todas las clases (≈ 10 % cada una) para que no sirvan de pista. Se dejaron fuera los correos cuyo sentido depende solo del destinatario.

## Dificultad

- **`directa`**: la señal es explícita. En `OTROS`, se ve a simple vista que no pide gestión sobre tickets.
- **`indirecta`**: se escala sin usar las frases típicas y hay que inferir la gestión. Solo existe en escalar.
- **`engañosa`**: es `OTROS` pero se parece a un escalamiento (cita tickets, usa "actualizar", "estado" o "generar ticket", o cierra un ticket). Solo existe en `OTROS`.

## Composición

| | principal | ajuste |
|---|---|---|
| `ACTUALIZACION_TICKET` | 45 | 9 |
| `SEGUIMIENTO_TICKET` | 45 | 9 |
| `MULTI_GESTION` | 30 | 6 |
| `OTROS` · `NUEVA_SOLICITUD` | 75 | 15 |
| `OTROS` · `NOTIFICACION` | 45 | 9 |
| `OTROS` · `CONFIRMACION` | 35 | 7 |
| `OTROS` · `CIERRE_CANCELACION` | 25 | 5 |
| **escalar** | **120 (40 %)** | **24 (40 %)** |

`ACTUALIZACION_TICKET` incluye 12 reaperturas en total entre ambos lotes.

## Prevalencia y precisión real

En el original, estas tres casuísticas son solo ~5 % de los correos. Aquí se enriquecieron al 40 % para tener suficientes positivos por clase. La precisión medida sobre este dataset es optimista respecto de producción. Para estimar la real se usa la tasa de verdaderos positivos (TPR) y la de falsos positivos (FPR) medidas, con π = 0,05:

```
precisión_real = π·TPR / (π·TPR + (1 − π)·FPR)
```

Por ejemplo, con TPR = 0,90 y FPR = 0,05, la precisión real sería ≈ 0,49, aunque aquí se vea mucho más alta.

## Uso

- `configs/escalamiento.yaml` evalúa `principal.csv`.
- Para iterar sobre `ajuste.csv` hay que apuntar `dataset.path` a ese archivo. Pendiente para la sesión principal: agregar una opción `--dataset` en la CLI.
- `tests/test_escalamiento_dataset.py` verifica que el harness cargue ambos lotes, que las etiquetas sean coherentes y que la composición se mantenga.
