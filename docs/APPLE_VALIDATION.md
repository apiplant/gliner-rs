# Port Apple: implementazione e verifiche

31 ottobre 2026. Riferimento Rust upstream:
`c5c65948478d687190f9c83fcc51d3afa15b5f2a`.

## Confronto Rust/Candle Metal e Swift/MLX Metal

È disponibile anche il backend Rust Metal: feature Cargo `metal`, flag CLI
`--metal` e libreria statica con interfaccia C per la demo iPhone/macOS.
Il modello Rust viene riutilizzato; l'unica correzione numerica necessaria evita
un temporaneo F64 nella maschera DeBERTa. Dettagli in [CANDLE_METAL.md](CANDLE_METAL.md).

Le build Release dei due motori producono output equivalenti su **28/28 casi
Small e 28/28 Multi**, sia su Mac sia su iPhone, senza eccezioni d'ordine nel
confronto diretto. Rispetto alla CPU, Multi conserva una sola eccezione near-tie
Berlin: 27/28 strict e 28/28 con la regola già documentata. La tolleranza sulle
confidenze resta `1e-4`; le tracce encoder Candle sono state verificate su Mac,
non esportate dall'interfaccia C su iPhone.

Benchmark: tre richieste italiane, Small e Multi, tre trial per motore, cinque
warm-up e trenta misure per richiesta. Sono 1.260 esecuzioni GPU e 630 coppie
di output strict per ciascun dispositivo. Sul Mac M3 Pro, MLX riduce la mediana
di circa 23–35%; sull'iPhone 17 Pro di circa 30–43%. Il picco RSS del processo,
incluso il caricamento, è circa dimezzato. Questo non dimostra un dimezzamento
permanente della memoria residente. Candle carica il modello più rapidamente
nei processi con cache di sistema/shader già calde.

La condizione concordata per rimuovere Swift richiede output, prestazioni e
memoria comparabili. Le misure non la soddisfano: il backend Swift/MLX rimane.
Passano 17 test Rust, 2 test FFI e 41 test Python. Il precedente run XCTest
rimane di 32 test senza skip.

Metodi, p95, campioni, limiti e hash sono nel
[report HTML](../reports/gliner-apple-report.html) e nei dati
[Mac](benchmarks/candle-vs-mlx-mac.json),
[iPhone](benchmarks/candle-vs-mlx-iphone.json) e
[validazione](benchmarks/candle-metal-evaluation.json).
Le sezioni seguenti conservano anche le verifiche MLX precedenti; il confronto
CPU/GPU storico non è un confronto Candle Metal contro MLX Metal.

## Cosa è disponibile

- Package Swift `GLiNERMLX`, con inferenza MLX e GPU Metal su Apple Silicon.
- App SwiftUI per macOS e iPhone: modello locale, testo ed etichette modificabili,
  risultati leggibili, confidenze e JSON opzionale.
- CLI `gliner-mlx`, downloader con revisioni e checksum verificati, comparatori e
  script per ripetere le verifiche su Mac e iPhone.
- Checkpoint GLiNER2.5 `small`, `base` e `multi`, architettura Boundary con encoder
  DeBERTa-v2. Entità, classificazione, relazioni, strutture natural/latent/anchorless,
  choice fields, attributi, batch e documenti lunghi.

Nel port Swift/MLX, i checkpoint legacy Span e gli encoder ModernBERT sono rifiutati esplicitamente.
La quantizzazione non è implementata. Non è stato pubblicato nulla su App Store.

## Ambiente effettivo

- Mac M3 Pro, 18 GiB RAM, macOS 27.2; Xcode 27.2 e Swift 6.4.2.
- iPhone 17 Pro fisico, installazione con firma e profilo di sviluppo locali.
- MLX Swift 0.33.0; dipendenze fissate in `swift/Package.resolved`.
- Riferimento CPU compilato con Rust 1.99.1, senza cambiare la toolchain globale.
- Deployment target dichiarati: macOS 14 e iOS 17. Le versioni minime non sono
  state provate su hardware con quelle specifiche versioni di sistema.

## Confronto con Rust

`scripts/apple_cases.json` contiene 21 casi; `apple_advanced_cases.json` aggiunge
sette casi con record latent/anchorless, ancore esplicite, descrizioni ed esempi.
Le strutture avanzate producono risultati non vuoti: il confronto non passa per
assenza di predizioni.

| Piattaforma / checkpoint | Risultati completi F32 | Nota |
| --- | --- | --- |
| Mac / small | 28/28 | Ordine esatto |
| Mac / base | 28/28 | Ordine esatto |
| Mac / multi | 28/28 | 27 con ordine esatto; un riordino quasi pari |
| iPhone / small | 28/28 | Ordine esatto |
| iPhone / multi | 28/28 | 27 con ordine esatto; un riordino quasi pari |

Tolleranza assoluta sulle confidenze: `0.0001`. Testo, etichette, numero di
risultati e offset rimangono esatti. L'unica eccezione di ordinamento ammessa e
registrata riguarda due occorrenze di “Berlin” nel caso lungo: le confidenze Rust
differiscono di `1.1920928955078125e-7`. Non sono stati modificati modello, soglie o
risultati attesi per forzare l'accordo.

Il comparatore consente questo riordino soltanto con `--allow-near-ties`, per
blocchi contigui di entità con identità esatte e confidenze quasi uguali su
entrambi i lati. Il conteggio rigoroso resta nel report. Record, relazioni e
classificazioni non vengono riordinati.

Sono stati confrontati anche tutti i valori dell'encoder:

| Test iPhone F32, 28 casi | Valori confrontati | Massimo errore assoluto |
| --- | ---: | ---: |
| small | 1.024.128 | 0,00007248 |
| multi | 2.592.768 | 0,00011028 |

La tolleranza dichiarata per questi tensori è `0.001`. Tutti i valori devono
essere finiti. Sul Mac sono stati controllati analogamente i 21 casi base.

La preparazione degli input coincide esattamente con Rust per small e multi:
21 casi base, 64 casi Unicode/tokenizer e sette casi avanzati, cioè 92 casi per
tokenizer. Sono inclusi emoji, ZWJ, accenti combinati, CJK, whitespace, marker ed
esempi. Il confronto ha individuato e corretto la sigma greca finale.

Test automatici: 32 XCTest passati, senza test saltati, e 36 test Python passati.
Dieci combinazioni di argomenti CLI non validi sono state respinte correttamente.

## Correzioni trovate sul dispositivo reale

**Precisione sui chip recenti.** MLX 0.33 abilita TF32 per default sul percorso
NAX, anche con tensori F32. Sul telefono questo produceva differenze di confidenza
fino a circa `0.0094`. Disabilitando TF32 prima del primo calcolo, gli stessi
casi passano entro `0.0001`. La configurazione è incorporata nell'SDK e nell'app.

TF32 è ora un'opzione esplicita. MLX memorizza il flag al primo calcolo: cambiarlo
dopo il caricamento richiede un nuovo processo. L'SDK respinge il cambio tardivo
e l'app richiede di chiudere e riaprire. Le app che usano anche altri modelli MLX
devono configurare il runtime prima del loro primo calcolo.

**Memoria nelle richieste consecutive.** La prima esportazione diagnostica
multilingue si interrompeva dopo 11 casi. Sono stati limitati i buffer conservati
e drenati gli oggetti temporanei fra le richieste. Il test completo di 28 casi e
l'esportazione di oltre 2,5 milioni di valori ora terminano correttamente.
Il default iOS segue la
[raccomandazione MLX di 20 MiB](https://github.com/ml-explore/mlx-swift/blob/0.33.0/Source/MLX/Documentation.docc/Articles/running-on-ios.md).

Nel test diagnostico multilingue sono stati osservati circa 1,49 GB di picco RSS
e 2,67 GB di picco attivo registrato da MLX. Sono misure diverse e sovrapposte,
non quantità da sommare; non sono un limite garantito per altri input o dispositivi.

## Prestazioni Mac e opzioni

Build **Release**, GPU Metal, checkpoint multilingue fissato a
`cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3`.
Tre richieste italiane di estrazione di entità da 64, 97 e 130 parole; per ogni
configurazione e testo, cinque richieste di riscaldamento e 30 misure. Ogni
combinazione usa un nuovo processo. Totale: 525 inferenze.

| Configurazione | Mediane a caldo sui tre testi, ms |
| --- | --- |
| F32, cache automatica | 18,68 / 24,79 / 33,18 |
| F32, cache disattivata e pulita | 20,90 / 27,75 / 37,82 |
| F32, cache 20 MiB | 20,22 / 28,07 / 36,65 |
| F16, cache automatica | 16,95 / 22,35 / 29,77 |
| F16, cache 20 MiB e pulita | 18,10 / 24,14 / 31,77 |

Gli output F32 sono identici tra le configurazioni di cache. I tre testi del
benchmark passano anche il confronto con Rust. F16 guadagna circa il 9–10%, con
differenze nelle confidenze e un riordino di entità: non viene attivato per default.
TF32 non è stato misurato sul chip M3, dove questo percorso non è applicabile.

I tempi a caldo escludono il caricamento dei pesi. Il primo avvio assoluto ha
richiesto circa 1,68 s per il caricamento e 1,69 s per la prima richiesta, che può
includere la compilazione degli shader. Gli altri processi hanno beneficiato delle
cache del sistema. Le cache di filesystem e shader non sono state svuotate.
Non si tratta di un benchmark di tutti i task o di tutti gli iPhone.

Il [report riproducibile](benchmarks/mac-m3-pro-multi.json) contiene p95, dati del
dispositivo, revisione dei pesi, metodologia e controlli di qualità.

**Profilo consigliato sul Mac:** Precisione/F32, cache automatica, riuso dei
buffer. L'app offre anche Velocità e Memoria ridotta; cache, pulizia e TF32 sono
controllabili separatamente. I parametri effettivi sono riportati dalla CLI.

F16 resta approssimato: nella suite completa ha modificato un confine di relazione
con small e alcune confidenze/candidati con multi. Non ha superato il criterio di
parità rigorosa, pur funzionando come modalità opzionale.

## Confronto Rust/Candle CPU e Swift/MLX GPU

Confronto Release sullo stesso Mac Apple Silicon M3 Pro, sullo stesso checkpoint
`fastino/gliner2.5-multi-v1` alla revisione
`cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3` e con tre richieste italiane identiche
da 64, 97 e 130 parole. Le richieste complete e le etichette sono nel
[report JSON riproducibile](benchmarks/rust-vs-mlx.json), insieme ai checksum del
manifest e dei quattro file del checkpoint.

| Richiesta (parole) | Rust/Candle CPU F32, 8 thread: mediana / p95 | Swift/MLX Metal F32: mediana / p95 | Rapporto mediane |
| --- | ---: | ---: | ---: |
| `michele_milano` (64) | 200,734 / 217,214 ms | 20,926 / 22,658 ms | 9,59× |
| `apple_ceo` (97) | 284,686 / 317,775 ms | 28,141 / 29,381 ms | 10,12× |
| `technology_report` (130) | 386,076 / 481,621 ms | 37,311 / 39,248 ms | 10,35× |

Ogni richiesta e configurazione è stata eseguita in un processo fresco. Sono stati
fatti cinque passaggi di riscaldamento, incluso il primo passaggio freddo, poi 30
misure; CPU e GPU sono state misurate in sequenza. In totale: nove processi e 315
esecuzioni della richiesta, di cui 270 misurate e 45 di riscaldamento. Il timer
comprende preparazione dell’input, inferenza, formattazione, serializzazione e
scrittura JSON su stdout. I tempi di caricamento del modello e i tempi del primo
passaggio sono riportati separatamente nel JSON.

Il riferimento CPU usa Rust 1.99.1 e Candle con GEMM puro Rust, senza MKL,
Accelerate o CUDA. La configurazione predefinita usa 8 thread (`RAYON_NUM_THREADS=8`);
la configurazione alternativa sintonizzata usa 4 (`RAYON_NUM_THREADS=4` e le
variabili ambiente correlate impostate a 4), lasciando il pool kernel Candle al
default di 6 thread su questo Mac (sei core performance). La mediana CPU sintonizzata per le tre richieste è
179,877 / 269,625 / 380,241 ms. Swift usa MLX F32 sulla GPU Metal, TF32 disattivato
e cache automatica, senza pulire le cache di filesystem o shader fra i processi.

Per ogni richiesta, tutte le 35 ripetizioni di ciascuna configurazione hanno
risultati discreti stabili e restano entro la tolleranza sulle confidenze `0.0001`.
Il massimo scarto di confidenza osservato fra Rust e MLX è `0.00000578165` nel
testo lungo. Sono quindi stabili i risultati di questo benchmark.

Questi numeri confrontano due configurazioni diverse: Rust/Candle su CPU contro
Swift/MLX su GPU Metal. Non isolano il linguaggio di programmazione e non sono un
confronto Rust-versus-Swift. I risultati valgono per questo Mac, questo checkpoint
e queste tre richieste; non predicono la velocità su altri dispositivi o task.

## Ripetere le verifiche

Partire dalla [guida per Mac e iPhone](../swift/README.md). Dalla radice del repo:

```sh
bash scripts/verify_apple.sh multi
python3 scripts/benchmark_apple.py --help
bash scripts/verify_iphone.sh DEVICE_ID multi --trace
```

Per l'ultima istruzione l'app deve essere già firmata e installata sul telefono.
Il verificatore usa un identificatore univoco di esecuzione, così un vecchio file
di successo non può essere scambiato per il risultato di un nuovo test.
I pesi, i log completi e gli output diagnostici sono in `.models/`, esclusa da Git.
