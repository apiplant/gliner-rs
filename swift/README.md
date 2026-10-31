# GLiNER per Mac e iPhone

Questa cartella contiene il port Apple di GLiNER2. Puoi usarlo come programma da Terminale su un Mac Apple Silicon oppure aprire la demo per macOS o iPhone.

## Requisiti

- Mac con chip Apple Silicon e macOS 14 o successivo.
- Xcode con Swift 6.3 o successivo. Il progetto è stato provato con Xcode 27.2 e Swift 6.4.2.
- Toolchain Metal per compilare la versione MLX. Se Xcode segnala che manca, installalo con `xcodebuild -downloadComponent MetalToolchain`.
- Per compilare la versione iPhone serve anche un account Apple personale configurato in Xcode. Non servono credenziali per scaricare i modelli o usare la versione Mac.
- Rust 1.99.1 serve soltanto per compilare il programma Rust di riferimento usato nei confronti.

## Scaricare un modello

Dal terminale, nella cartella principale del repository:

```sh
python3 scripts/download_apple_model.py --variant multi
```

Il modello viene salvato in `.models/gliner2.5-multi-v1`. Il download serve una sola volta; dopo, il modello può essere usato senza rete. `multi` supporta l’italiano. Il modello `small` è pensato soprattutto per l’inglese ed è più contenuto: circa 304 MB per `small` e circa 1,17 GB per `multi`.

Per scaricare la variante piccola:

```sh
python3 scripts/download_apple_model.py --variant small
```

## Usare GLiNER dal terminale

Questo comando estrae persone, aziende e luoghi dal testo:

```sh
bash scripts/run_apple.sh --model .models/gliner2.5-multi-v1 --text 'Michele vive a Milano e lavora per Acme.' --entities person,company,location
```

Puoi chiedere più attività nella stessa richiesta:

```sh
bash scripts/run_apple.sh --model .models/gliner2.5-multi-v1 \
  --text 'Michele vive a Milano e lavora per Acme.' \
  --entities person,company,location \
  --classify sentiment=positive,negative,neutral \
  --relations works_for \
  --structure 'employment=employee::str,company::str'
```

Altre opzioni utili:

- `--fp16` prova l’encoder a 16 bit. È sperimentale e può cambiare alcuni risultati; F32 resta il valore predefinito.
- `--chunked` divide testi lunghi in finestre sovrapposte.
- `--cases scripts/apple_cases.json` elabora una raccolta di esempi JSON invece di un singolo testo.
- `--trace` stampa i valori interni dell’encoder in formato JSON, utile per confronti tecnici. I risultati possono essere molto grandi.

## Aprire la demo per Mac

Compila l’app in configurazione Release e aprila:

```sh
bash scripts/build_apple.sh app-macos
open .derivedData/Build/Products/Release/GLiNERDemo.app
```

Nell’app scegli `Multi` o `Small`, premi **Scarica modello** e attendi che compaiano tutti e quattro i file. Il primo download richiede una connessione Internet; poi l’app conserva il modello e può lavorare offline. L’app iPhone usa la stessa procedura e salva il modello sul dispositivo.

## Avviare la demo su iPhone

Apri il progetto Xcode:

```sh
open ios/GLiNERDemo.xcodeproj
```

In Xcode seleziona la destinazione iPhone, imposta il tuo account personale nella sezione **Signing & Capabilities**, poi seleziona l’iPhone collegato e premi **Run**. La firma personale è richiesta per installare l’app su un dispositivo reale.

## Usare i tipi Swift nell’app

Il modello si carica con `GLiNERModel` in [Model.swift](Sources/GLiNERMLX/Model.swift). Per costruire richieste personalizzate puoi usare i tipi pubblici `Schema`, `EntitySpec`, `ClassificationSpec`, `RelationSpec` e `StructureSpec` definiti in [Schema.swift](Sources/GLiNERMLX/Schema.swift). Il modello accetta schemi con estrazione di entità, classificazione, relazioni e strutture.

```swift
import Foundation
import GLiNERMLX

let directory = URL(fileURLWithPath: ".models/gliner2.5-multi-v1")
let profile = InferenceProfile.accurate
try GLiNERRuntime.configure(profile.runtimeConfiguration) // prima di altre operazioni MLX nell’app
let model = try GLiNERModel(
    modelDirectory: directory,
    precision: profile.precision,
    runtimeConfiguration: profile.runtimeConfiguration
)
let schema = Schema(entities: ["person", "company", "location"])
let result = try model.extract(
    "Michele vive a Milano e lavora per Acme.",
    schema: schema,
    options: .init(includeSpans: true, includeConfidence: true)
)
print(try result.prettyPrinted())
```

`GLiNERModel` contiene stato MLX e il processor modificabile: in un’app con richieste concorrenti, serializza le chiamate, per esempio tramite un actor dedicato.

`GLiNERModel` configura automaticamente il runtime prima di caricare i pesi. Se l’app usa già MLX, chiama `GLiNERRuntime.configure(...)` all’avvio, prima di qualunque calcolo MLX. La scelta TF32 è memorizzata da MLX al primo calcolo: dopo aver eseguito calcoli, per cambiarla occorre chiudere e riaprire il processo. La cache si può aggiornare fra richieste con `model.updateRuntimeConfiguration(...)`; anche in questo caso è meglio serializzare le chiamate al modello.

## Profili e cache

F32 è la scelta predefinita per accuratezza e riproducibilità. Da riga di comando puoi selezionare:

```sh
# Accurate: F32 (valore predefinito)
bash scripts/run_apple.sh --model .models/gliner2.5-multi-v1 --text 'Michele vive a Milano.' --entities person,location --profile accurate

# Fast: encoder F16 sperimentale
bash scripts/run_apple.sh --model .models/gliner2.5-multi-v1 --text 'Michele vive a Milano.' --entities person,location --profile fast

# Low memory: F16, cache da 20 MiB e pulizia della cache dopo ogni richiesta
bash scripts/run_apple.sh --model .models/gliner2.5-multi-v1 --text 'Michele vive a Milano.' --entities person,location --profile low-memory
```

`--fp16` e `--fp32` sovrascrivono la precisione scelta dal profilo. `--tf32` e `--no-tf32` controllano TF32 indipendentemente dal profilo e sono mutuamente esclusivi. `--cache-mb auto|0|20|128|512|N` seleziona la cache automatica del sistema, disabilita la cache (`0`) o imposta un limite intero in MiB. `--clear-cache` pulisce la cache MLX dopo ogni richiesta; `--retain-cache` la conserva. Anche queste due opzioni sono mutuamente esclusive.

L’impostazione automatica di piattaforma usa la cache MLX del sistema su Mac e non la pulisce dopo ogni richiesta. Su iPhone imposta invece un limite di 20 MiB e pulisce la cache dopo ogni richiesta per contenere la memoria. `low-memory` applica lo stesso limite e la pulizia su entrambe le piattaforme. La CLI stampa su stderr il profilo, la precisione, TF32, limite cache e pulizia effettivamente usati; `--repeat N` ripete ogni richiesta e stampa i tempi su stderr.

## Supporto e stato delle verifiche

Il port supporta checkpoint con architettura **boundary** e encoder **DeBERTa-v2**. I checkpoint basati sull’architettura span e gli encoder ModernBERT non sono ancora supportati. La quantizzazione non è ancora disponibile.

### Risultati verificati

- **Input e tokenizer:** 21 casi di base, 64 casi limite e sette casi avanzati passano con corrispondenza esatta (`92/92`) per `small` e `multi`. I casi limite includono emoji con sequenze ZWJ e tonalità della pelle, bandiere, accenti composti e scomposti, greco e turco, spazi e righe multiple, email, URL, testo vuoto, marker letterali, CJK e prompt di classificazione.
- **Inferenza F32 su Mac con GPU:** `small`, `base` e `multi` passano tutti i `28/28` casi con tolleranza `0.0001` sulle confidenze. `multi` include un riordino di risultati con punteggi quasi uguali.
- **Verifica encoder su iPhone:** `small` passa i `28` casi e confronta `1.024.128` valori. `multi` passa i `28` casi, con un riordino near-tie, confronta `2.592.768` valori e ha un massimo errore assoluto di `0.00011027` rispetto al riferimento F32. La traccia completa `multi` è impegnativa: durante la verifica ha raggiunto circa 1,49 GB di RSS e 2,67 GB di picco MLX; sono misure sovrapposte, da non sommare.
- **Memoria su iPhone:** il limite cache di 20 MiB e il rilascio temporaneo delle risorse dopo ogni richiesta hanno risolto il crash durante l’elaborazione prolungata.
- **Test automatici:** 32 test XCTest e 36 test Python passati, senza test XCTest saltati. Dieci combinazioni CLI non valide vengono respinte.
- **Demo macOS:** aperta e provata con “Michele vive a Milano e lavora per Acme.”; ha estratto correttamente Michele, Milano e Acme.
- **Demo iPhone:** installata e provata su iPhone 17 Pro fisico. Anche la versione finale con profili configurabili supera tutti i 28 casi. Compilazioni riuscite per Mac, iPhone e simulatore.

### Prestazioni verificate sul Mac

Mac M3 Pro con 18 GiB RAM, build Release, modello multilingue, estrazione di entità
da tre testi italiani di 64, 97 e 130 parole. Dopo cinque richieste di riscaldamento,
30 misure per testo e configurazione hanno dato queste mediane:

| Impostazione | 64 parole | 97 parole | 130 parole |
| --- | ---: | ---: | ---: |
| Precisione, cache Auto | 18,68 ms | 24,79 ms | 33,18 ms |
| Precisione, cache disattivata | 20,90 ms | 27,75 ms | 37,82 ms |
| Velocità, cache Auto | 16,95 ms | 22,35 ms | 29,77 ms |

Sul Mac consigliamo **Precisione e cache Auto**: i risultati rimangono confrontabili
con Rust e il riuso dei buffer favorisce la velocità. Il profilo Velocità guadagna
circa il 9–10%, con precisione ridotta. Il primo utilizzo richiede anche caricamento
e possibili compilazioni degli shader, quindi può durare alcuni secondi.
TF32 non accelera il chip M3: l'opzione riguarda hardware compatibile più recente.
Metodologia, p95 e risultati completi sono nel [rapporto di verifica](../docs/APPLE_VALIDATION.md)
e nel [report del benchmark](../docs/benchmarks/mac-m3-pro-multi.json).

### Confronto con Rust/Candle su CPU

Con le stesse tre richieste e lo stesso checkpoint `multi` in F32, il riferimento Rust/Candle su CPU e l’implementazione Swift/MLX su GPU Metal hanno prodotto questi tempi mediani a caldo:

| Richiesta | Parole | Rust CPU F32, 8 thread | MLX GPU F32 | Rapporto CPU/GPU |
| --- | ---: | ---: | ---: | ---: |
| `michele_milano` | 64 | 201 ms | 21 ms | 9,59× |
| `apple_ceo` | 97 | 285 ms | 28 ms | 10,12× |
| `technology_report` | 130 | 386 ms | 37 ms | 10,35× |

Sono 30 misure dopo cinque passaggi di riscaldamento per ciascuna richiesta; il p95, i tempi di caricamento, i primi passaggi a freddo e una seconda configurazione Rust a 4 thread sono nel [report riproducibile](../docs/benchmarks/rust-vs-mlx.json). Questo confronto misura due combinazioni di runtime e hardware — Rust/Candle su CPU contro Swift/MLX su GPU — e non la velocità dei linguaggi Rust e Swift.

### Precisione F16

F16 è una modalità sperimentale, non la modalità predefinita. Nel confronto precedente sui 21 casi, `small` corrisponde in `20/21`: una relazione ha uno span diverso. `multi` corrisponde in `18/21`: alcune confidenze differiscono di oltre `0.02` e il testo lungo produce un insieme di candidati diverso. Questi dati non sono un benchmark prestazionale. Per risultati confrontabili usa F32; abilita `--fp16` o `--profile fast` solo se accetti possibili differenze.

Per ripetere le verifiche dal terminale, scaricare i pesi e compilare prima le dipendenze richieste:

```sh
bash scripts/verify_apple.sh small
bash scripts/verify_apple.sh multi
```

Questi comandi salvano i risultati intermedi sotto `.models/` e compilano la CLI macOS in Release. La compilazione dei test resta Debug se non imposti `CONFIGURATION`. La demo iPhone si firma con il tuo account personale in Xcode per l’installazione locale; questa guida non descrive una pubblicazione su App Store.
