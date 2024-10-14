# Human and Object Detection for Security Systems

## Introduzione
Questo progetto simula il funzionamento di un sistema di sicurezza per la sorveglianza domestica, con l’obiettivo di rilevare in modo rapido e accurato la presenza di intrusi. Il sistema analizza fotogrammi video in tempo reale utilizzando la libreria OpenCV per migliorare la qualità delle immagini, mentre il modello YOLOv5, basato su PyTorch, rileva e classifica gli oggetti in umani, animali (gatti e cani), e automobili. Se viene rilevato un essere umano, il sistema attiva un allarme.

## Tecnologie Utilizzate
- **OpenCV**: Per migliorare le immagini (equalizzazione dell'istogramma, sharpening, CLAHE, regolazione di luminosità e contrasto).
- **YOLOv5**: Per il rilevamento di oggetti in tempo reale.
- **PyTorch**: Per eseguire il modello di rete neurale YOLOv5.
- **Playsound**: Per riprodurre un segnale acustico in caso di intrusione.

## Funzionalità Principali
- Rilevamento in tempo reale di esseri umani, animali e automobili.
- Allarme attivato al rilevamento di un essere umano persistente.
- Analisi delle prestazioni del rilevamento su immagini originali, migliorate, e di bassa qualità.
- Statistiche sui rilevamenti per monitorare la performance del modello.

## Funzioni

#### `detect(frame)`
Esegue l'inferenza utilizzando il modello YOLOv5 su un singolo frame. Questa funzione riceve un'immagine o un fotogramma video e restituisce i risultati del rilevamento sotto forma di un DataFrame di Pandas, contenente le coordinate degli oggetti rilevati, il loro nome e la confidenza.

#### `draw_frame(frame, res, name, color)`
Disegna i risultati del rilevamento direttamente sul frame. Utilizzando i dati del DataFrame generato da `detect()`, crea rettangoli attorno agli oggetti rilevati e aggiunge il nome dell'oggetto insieme alla confidenza percentuale. Il colore e il nome degli oggetti sono personalizzabili.

#### `play_sound(stop_thread)`
Riproduce un suono di allarme su un thread separato finché non viene emesso un segnale per fermarlo. Utilizza la libreria `playsound` per suonare un file audio specificato continuamente, finché l'evento di stop non viene attivato.

#### `detect_human(res, slwin)`
Gestisce il rilevamento di persone utilizzando una finestra mobile (deque) per ridurre i falsi positivi. Quando il numero di rilevamenti di persone supera una soglia definita, la funzione avvia l'allarme sonoro. Il rilevamento viene considerato affidabile solo se la presenza viene confermata per un numero sufficiente di frame consecutivi.

#### `yuv_equalization(image)`
Esegue l'equalizzazione dell'istogramma del canale Y di un'immagine convertita nello spazio colore YUV. L'obiettivo è migliorare il contrasto, rendendo più evidenti i dettagli delle aree scure e riducendo l'intensità delle zone luminose. L'immagine viene poi riconvertita nello spazio BGR.

#### `adjust_brightness_contrast(image, brightness=0, contrast=0)`
Regola la luminosità e il contrasto di un'immagine. Utilizza valori di input per modificare queste proprietà, applicando trasformazioni pesate sull'immagine per bilanciare le aree più scure o luminose e rendere più visibili i dettagli desiderati.

#### `apply_model()`
Questa funzione esegue il modello YOLOv5 su ciascun frame o immagine. Riceve in input il flusso video e applica il modello per rilevare e classificare gli oggetti, restituendo poi i risultati per ulteriori elaborazioni.

### Funzione `enhance_image()`
La funzione `enhance_image(image)` richiama altre tre funzioni di miglioramento delle immagini:
- **`sharpen_image()`**: Applica un filtro di sharpening per migliorare i bordi e rendere più nitidi i dettagli nell'immagine.
- **`clahe()`**: Migliora il contrasto delle aree locali utilizzando il CLAHE, migliorando i dettagli in scene con luminosità variabile.
- **`gaussian_blur()`**: Applica una sfocatura gaussiana per ridurre il rumore e ammorbidire i dettagli fini.

### Funzione `reduce_quality()`
La funzione `reduce_quality(image)` richiama due funzioni per degradare la qualità delle immagini:
- **`add_noise()`**: Aggiunge rumore all'immagine tramite una distribuzione casuale, riducendo la qualità e simulando un'immagine disturbata.
- **`down_sample()`**: Riduce la risoluzione dell'immagine e successivamente la riporta alla risoluzione originale, perdendo così dettagli e creando un'immagine più sfocata.

## Installazione
1. Clona il repository.
2. Installa le dipendenze elencate nel `requirements.txt`:
   ```bash
   pip install -r requirements.txt
## Statistiche e Risultati
Il sistema offre una valutazione statistica della precisione del rilevamento, distinguendo tra immagini originali, migliorate e di bassa qualità.
