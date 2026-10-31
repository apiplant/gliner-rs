# GLiNER2 su iOS con MLX e Metal

Analisi del 31 ottobre 2026 sul commit upstream
`c5c65948478d687190f9c83fcc51d3afa15b5f2a`.

Questa è l'analisi iniziale. Il port è stato successivamente implementato in
[`swift/`](../swift/README.md), con app dimostrativa in `ios/`. I risultati effettivi
sono nel [rapporto di verifica](APPLE_VALIDATION.md).

## Che cosa contiene il repository

`gliner-rs` è un runtime di inferenza Rust basato su Candle per GLiNER2.
Riceve testo e uno schema di etichette definito dal chiamante e produce entità,
classificazioni, relazioni, attributi e record JSON. Schema e testo passano insieme
attraverso un encoder bidirezionale; teste specializzate assegnano punteggi alle
etichette e agli intervalli del testo. Non è un generatore autoregressivo.

Il codice supporta due famiglie di architetture (`src/model.rs`, `src/config.rs`):

- **Boundary**, usata dai checkpoint GLiNER2.5: entità, classificazione,
  relazioni, record e attributi.
- **Span**, usata da precedenti checkpoint PII/guardrail: classificazione ed
  entità; relazioni e strutture vengono rifiutate da questo runtime.

Gli encoder implementati sono DeBERTa-v2 e ModernBERT. I tre checkpoint GLiNER2.5
verificati sotto usano tutti Boundary + DeBERTa-v2. Il README upstream iniziale
non descrive completamente questa varietà.

## Percorso proposto

Per l'obiettivo MLX, realizzare un package **Swift + MLX Swift**: MLX esegue già
le operazioni GPU tramite Metal. Non occorre partire scrivendo kernel Metal;
la necessità di kernel dedicati va valutata dopo il profiling.

Usare il runtime Rust come riferimento numerico. Valutare un piccolo bridge Rust
per il tokenizer se permette di ottenere subito la parità; il runtime tensoriale
del port resta MLX. Un port interamente Swift richiede anche la corretta
implementazione della tokenizzazione Unigram e delle sue normalizzazioni.

La strada alternativa **Rust + Candle Metal + FFI Swift** conserva una parte molto
maggiore del codice, ma non realizza il port MLX. Il crate attuale non espone una
feature Metal, una API C o un target `staticlib`: questi elementi, il packaging
XCFramework e la verifica dei kernel sul target iOS sarebbero lavoro aggiuntivo.
Candle include già gli shader e ne gestisce la compilazione runtime.

Non esiste un'astrazione backend che consenta di sostituire Candle con MLX
semplicemente cambiando un flag.

## Requisiti Apple verificati

[MLX Swift 0.33.0](https://github.com/ml-explore/mlx-swift/blob/0.33.0/Package.swift)
dichiara iOS 17+, macOS 14+ e Swift 6.3. Il
[README della versione](https://github.com/ml-explore/mlx-swift/blob/0.33.0/README.md)
richiede Xcode/xcodebuild per compilare gli shader Metal: il solo `swift build`
non è la verifica finale dell'app.

Ambiente locale rilevato: Mac arm64, Xcode 27.2, Swift 6.4.2, SDK iPhoneOS e
iPhoneSimulator 27.2. Questi strumenti soddisfano i requisiti dichiarati; non è
ancora stata compilata un'app MLX né verificato un iPhone fisico.

[Candle 0.11](https://github.com/huggingface/candle/blob/0.11.0/candle-core/Cargo.toml)
ha un backend Metal e percorsi specifici iOS. Il suo
[wrapper Metal](https://github.com/huggingface/candle/blob/0.11.0/candle-metal-kernels/src/metal/device.rs)
usa `MTLCopyAllDevices`: l'header del SDK locale lo dichiara disponibile da iOS 18.
Un deployment target inferiore per questa alternativa richiede adattamento e
verifica. La presenza del backend non prova la compatibilità end-to-end su iPhone.

## Checkpoint e memoria

Dati verificati tramite API Hugging Face, configurazioni e header safetensors,
senza scaricare i pesi completi. MB e GB sono decimali. Le dimensioni non sono
misure della RAM di un processo iOS.

| Checkpoint | Parametri totali | File pesi F32 | Pesi encoder F16 + tutte le teste F32, stima |
| --- | ---: | ---: | ---: |
| `fastino/gliner2.5-small-v1` | 73.881.879 | 295,57 MB | 154,23 MB |
| `fastino/gliner2.5-base-v1` | 193.581.591 | 774,37 MB | 406,80 MB |
| `fastino/gliner2.5-multi-v1` | 287.355.159 | 1.149,46 MB | 594,35 MB |

Le stime miste sono `2 × parametri encoder + 4 × parametri altre teste`.
Escludono attivazioni, temporanei, allocator, cache Metal, tokenizer e copie
durante il caricamento. Non costituiscono un budget RAM garantito.

Revisioni immutabili e fonti dei dati:

- [small: df5910e44bc4ffdb0d95399a83b0ca4516349aa5](https://huggingface.co/fastino/gliner2.5-small-v1/tree/df5910e44bc4ffdb0d95399a83b0ca4516349aa5).
- [base: 54785d51df8d86a0d8f2eb212fec15871aad2c4d](https://huggingface.co/fastino/gliner2.5-base-v1/tree/54785d51df8d86a0d8f2eb212fec15871aad2c4d).
- [multi: cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3](https://huggingface.co/fastino/gliner2.5-multi-v1/tree/cf5593a5d45e3bbf204b9df621b13b1c0cf25ee3).

Le configurazioni dichiarano 12 layer, hidden size 384/768/768, 6/12/12 teste di
attenzione e `max_position_embeddings=512`. Non trattare 512 come limite effettivo
già imposto dal runtime: usa posizioni relative e contiene test con input più
lunghi. Su mobile va introdotto un budget esplicito di subword che includa lo
schema, quindi validato il chunking.

Per il primo prototipo piccolo e rapido scegliere `small`; per una prima
versione italiana/multilingue scegliere `multi`. La qualità in italiano di
`small` e `base`, presentati come checkpoint inglesi nelle model card, non è
dimostrata da questa analisi. Non scegliere il modello solo dal numero di pesi.

## Mappa del porting

| Componente | Sorgente Rust | Lavoro per MLX/iOS |
| --- | --- | --- |
| Configurazione e pesi | `src/config.rs`, `src/model.rs` | Loader, validazione, mapping di nomi, shape e dtype |
| Prompt e tokenizer | `src/processor.rs` | Preservare schema, marker, ID e indici dei primi subword |
| DeBERTa-v2 | `src/deberta.rs` | Attention c2p/p2c, log bucket, gather, maschere, GELU erf e LayerNorm |
| Boundary e scoring | `src/heads.rs` | Attention locale, SwiGLU, marginali start/end/inside, pool condiviso, FiLM e astensione |
| Classificazione | `src/heads.rs` | MLP sugli stati dei marker delle etichette |
| Relazioni | `src/heads.rs`, `src/decode.rs` | Proposta coppie, scorer e dedup degli archi |
| Record e attributi | `src/records.rs`, `src/heads.rs` | Modalità natural/latent/anchorless, Hungarian, choice e reranker |
| Postprocessing | `src/decode.rs`, `src/schema.rs` | Soglie, overlap, validatori, offset e JSON |
| Documenti lunghi | `src/chunking.rs`, `src/model.rs` | Finestre sovrapposte, riallineamento e merge |

I quattro asset richiesti sono `config.json`, `encoder_config/config.json`,
`tokenizer.json` e `model.safetensors` (`src/download.rs`).
[MLX Swift legge safetensors](https://github.com/ml-explore/mlx-swift/blob/0.33.0/Source/MLX/IO.swift)
direttamente: non serve cambiare necessariamente il formato, ma serve implementare
e caricare tutte le parti del modello, non soltanto l'encoder.

## Rischi da verificare per primi

1. **Tokenizzazione.** `Processor::prepare` tokenizza separatamente ciascun
   elemento dello schema e ciascuna parola con special token automatici disattivati,
   poi concatena gli ID. Applica lowercase alle parole del testo e aggiunge `.`
   quando non termina con `.`, `!` o `?`. Tokenizzare una sola stringa in Swift
   non riproduce automaticamente gli input Rust. Tutti e tre i tokenizer
   verificati sono Unigram con Metaspace e normalizzazione NFC. Il registry
   [swift-transformers verificato al 31 ottobre](https://github.com/huggingface/swift-transformers/blob/93e4db644f7fda4460c61211c4b10fb67c1d0bdb/Sources/Tokenizers/Tokenizer.swift)
   non offre un caricamento diretto verificato di `DebertaV2Tokenizer`; il fallback
   BPE non sostituisce questo tokenizer. Confrontare gli ID esatti prima dei logits.
2. **Precisione.** Il codice converte a F32 gli stati dell'encoder prima delle
   teste e carica le teste in F32 anche con encoder F16. Partire da F32 per la
   parità, poi provare encoder F16 conservando le teste F32. Piccole variazioni
   possono cambiare il top-k o attraversare le soglie.
3. **Decisioni discrete.** Preservare ordine stabile, quote per query, top-k,
   dedup, Hungarian e tie-breaking nella risoluzione degli overlap. Queste parti
   possono restare inizialmente su CPU.
4. **Unicode.** Gli offset restituiti sono code point Unicode. Swift `String.count`
   conta grapheme cluster e UIKit usa spesso UTF-16: occorre una mappa esplicita,
   verificata con emoji e accenti combinati.
5. **Sincronizzazioni.** Le teste Rust fanno diverse conversioni tensore-vettore.
   Una traduzione letterale può forzare molte valutazioni del grafo lazy MLX e
   sincronizzazioni GPU/CPU. Misurare e aggregare questi passaggi dopo la parità.
6. **Caricamento e sandbox.** Preferire file locali a copie integrali dei pesi in
   memoria. Il downloader Rust usa `~/.cache`, variabili d'ambiente e HTTP
   sincrono: su iOS il lifecycle degli asset va gestito nella sandbox dell'app.

Preprocessing e decoder sono riutilizzabili tramite Rust FFI, ma non già isolati
in un crate indipendente: `records.rs` mescola matematica tensoriale e decoder,
e alcuni tipi CPU dipendono dai moduli delle teste. Un'eventuale estrazione del
core condiviso richiede refactoring.

## Fasi e criteri di completamento

1. **Reference riproducibile:** fissare revisione del checkpoint e configurazioni;
   generare fixture Rust F32 di token ID, indici parola/marker, hidden states,
   logits e JSON. Includere testi italiani, inglesi, CJK, Unicode e nessun match.
2. **Package Swift MLX:** loader dei quattro asset, tokenizer verificato e
   DeBERTa-v2. Primo task: classificazione, che evita la complessità del pool
   Boundary e permette di controllare l'encoder.
3. **Entità:** portare Boundary, pool candidati, astensione e overlap;
   confrontare span ed etichette esatti e logits/confidenze con tolleranze esplicite.
4. **iPhone fisico:** compilazione Xcode, inferenza locale senza rete, budget
   token, cancellazione tra richieste e lavoro fuori dal main actor. Misurare
   caricamento, latenza warm p50/p95, memoria di picco e comportamento termico.
5. **Precisione mista:** encoder F16 e teste F32, ripetendo i controlli di qualità.
   Stabilire soglie accettabili sui dati reali prima di approvare questa modalità.
6. **Copertura completa:** relazioni, record, choice, attributi, batching e
   chunking. ModernBERT e legacy Span sono estensioni distinte, da non promettere
   con il solo supporto iniziale Boundary + DeBERTa-v2.
7. **Ottimizzazione:** quantizzazione selettiva di embedding/linear e riduzione
   delle sincronizzazioni, guidate da misure e rivalutazione della qualità.

La suite esistente comprende `scripts/parity.py`, `examples/parity.rs`,
`scripts/parity_cases.json` e `scripts/compare.py`. I test JSON finali vanno
affiancati a fixture intermedie. Prima di usare il comparatore come gate,
irrobustire il controllo dei valori non finiti; il caso lungo attuale non prova
direttamente le API `_long`.

## Stato della verifica al termine dell'analisi iniziale

Repository clonato nel workspace corrente. Nessuna inferenza MLX/iOS implementata
in questa fase di analisi; nessuna misura di velocità su iPhone disponibile.

Il Cargo predefinito 1.82 non legge alcune dipendenze Edition 2024 del lockfile.
Con Rust 1.90 la compilazione incontra il tipo NEON `float16x8_t` usato da
Candle 0.11; il tipo è
[stabile da Rust 1.94](https://doc.rust-lang.org/stable/core/arch/aarch64/struct.float16x8_t.html).
Per la verifica usare una toolchain recente senza cambiare quella globale:

```sh
rustup toolchain install 1.99.1 --profile minimal --no-self-update
cargo +1.99.1 test --locked --no-default-features
```

Verifica eseguita con successo: **17 test unitari e 1 doctest passati**.
La toolchain 1.99.1 è stata installata separatamente, senza cambiare il default
globale. Questi test verificano la libreria CPU e la compilazione dell'esempio
documentale; non eseguono un checkpoint completo né provano la parità MLX/iOS.
