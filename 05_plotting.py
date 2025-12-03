import json
import folium
import os
import glob
import re
import matplotlib.pyplot as plt

# =========================
# NORMALIZATION INFO
# =========================

def load_normalization_info():
    # Percorsi possibili (Desktop normale e OneDrive)
    home = os.path.expanduser("~")
    possible_desktops = [
        os.path.join(home, "Desktop"),
        os.path.join(home, "OneDrive", "Desktop"),
    ]

    print("Cerco normalization_info.json in:")
    for d in possible_desktops:
        print("  -", d)

    for d in possible_desktops:
        info_path = os.path.join(d, "normalization_info.json")
        if os.path.exists(info_path):
            print(f"Trovato normalization_info.json: {info_path}")
            with open(info_path, "r") as f:
                info = json.load(f)

            lat_min = info["columns"]["Latitude"]["min"]
            lat_max = info["columns"]["Latitude"]["max"]
            lon_min = info["columns"]["Longitude"]["min"]
            lon_max = info["columns"]["Longitude"]["max"]

            return lat_min, lat_max, lon_min, lon_max

    raise FileNotFoundError(
        "normalization_info.json non trovato sul Desktop o Desktop OneDrive."
    )

# Variabili globali per normalizzazione (verranno sovrascritte)
LAT_MIN = LAT_MAX = LON_MIN = LON_MAX = None

def denormalize_lat(lat_norm):
    return LAT_MIN + lat_norm * (LAT_MAX - LAT_MIN)

def denormalize_lon(lon_norm):
    return LON_MIN + lon_norm * (LON_MAX - LON_MIN)

def denormalize_coords(coord_list):
    return [(denormalize_lat(p["lat"]), denormalize_lon(p["lon"])) for p in coord_list]

# =========================
# FIND FILES ON DESKTOP
# =========================

def possible_desktops():
    home = os.path.expanduser("~")
    return [
        os.path.join(home, "Desktop"),
        os.path.join(home, "OneDrive", "Desktop"),
    ]

def find_json_on_desktop(basename="test_predictions"):
    print("Cerco il file JSON in queste cartelle:")
    for d in possible_desktops():
        print("  -", d)

    for d in possible_desktops():
        if not os.path.isdir(d):
            continue
        pattern = os.path.join(d, f"{basename}*.json")
        matches = glob.glob(pattern)
        if matches:
            # prendo il primo che trovo
            return matches[0]
    return None

def find_log_in_same_dir(json_path):
    """
    Cerca un file di log nella stessa cartella del JSON.
    Priorità: *.out, poi *.log. Prende il più recente.
    """
    directory = os.path.dirname(json_path)
    candidates = []

    candidates.extend(glob.glob(os.path.join(directory, "*.out")))
    candidates.extend(glob.glob(os.path.join(directory, "*.log")))

    if not candidates:
        return None

    # Ordina per data di modifica (più recente per primo)
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]

# =========================
# LOG PARSING & PLOTTING
# =========================

def parse_log(log_path):
    """
    Estrae:
    - epoch_times: durata per epoch (in minuti)
    - train_losses: train loss per epoch
    - val_losses: validation loss per epoch
    """
    epoch_times = []
    train_losses = []
    val_losses = []

    time_re = re.compile(r"Epoch train loop finished\. Time = ([0-9]+\.[0-9]+) min")
    train_re = re.compile(r"Epoch\s+(\d+)/\d+\s+Train Loss:\s+([0-9.eE+-]+)")
    val_re = re.compile(r"Epoch\s+(\d+)/\d+\s+Validation Loss:\s+([0-9.eE+-]+)")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            m_time = time_re.search(line)
            if m_time:
                epoch_times.append(float(m_time.group(1)))
                continue

            m_train = train_re.search(line)
            if m_train:
                train_losses.append(float(m_train.group(2)))
                continue

            m_val = val_re.search(line)
            if m_val:
                val_losses.append(float(m_val.group(2)))
                continue

    n = min(len(epoch_times), len(train_losses), len(val_losses))
    epoch_times = epoch_times[:n]
    train_losses = train_losses[:n]
    val_losses = val_losses[:n]

    cumulative_times = []
    total = 0.0
    for t in epoch_times:
        total += t
        cumulative_times.append(total)

    return cumulative_times, train_losses, val_losses

def plot_losses(cumulative_times, train_losses, val_losses, save_path=None):
    plt.figure()
    plt.plot(cumulative_times, train_losses, marker="o", label="Train loss")
    plt.plot(cumulative_times, val_losses, marker="o", linestyle="--", label="Val loss")
    plt.xlabel("Tempo (minuti dall'inizio)")
    plt.ylabel("Loss")
    plt.title("Andamento loss vs tempo")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        print(f"Grafico della loss salvato in: {save_path}")
    else:
        plt.show()

# =========================
# VISUALIZZAZIONE TRAIETTORIE
# =========================

def visualize_sequences(json_path, output_html="sequences_map.html", max_sequences=None):
    print(f"\nUso il file JSON: {json_path}")

    global LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
    LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = load_normalization_info()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # limita il numero di sequenze da plottare, se richiesto
    if max_sequences is not None:
        max_sequences = max(1, min(int(max_sequences), len(data)))
        data_to_plot = data[:max_sequences]
        print(f"Plotto {max_sequences} sequenze (su {len(data)} totali).")
    else:
        data_to_plot = data
        print(f"Plotto tutte le {len(data)} sequenze disponibili.")

    # prendo solo un sottoinsieme per calcolare il centro della mappa (es. 40 sequenze)
    all_coords = []
    for seq in data_to_plot[:40]:
        all_coords.extend(denormalize_coords(seq["known_path"]))
        all_coords.extend(denormalize_coords(seq["ground_truth"]))
        all_coords.extend(denormalize_coords(seq["prediction"]))


    if not all_coords:
        print("[ERRORE] Nessuna coordinata trovata nel JSON.")
        return None

    avg_lat = sum(lat for lat, lon in all_coords) / len(all_coords)
    avg_lon = sum(lon for lat, lon in all_coords) / len(all_coords)

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=6)

    # Disegno tutte le sequenze
    for seq in data_to_plot:

        seq_id = seq.get("sequence_id", "N/A")

        known_coords = denormalize_coords(seq["known_path"])
        gt_coords = denormalize_coords(seq["ground_truth"])
        pred_coords = denormalize_coords(seq["prediction"])

        folium.PolyLine(
            known_coords,
            color="blue",
            weight=3,
            opacity=0.7,
            tooltip=f"Sequence {seq_id} Known Path"
        ).add_to(m)

        folium.PolyLine(
            gt_coords,
            color="green",
            weight=3,
            opacity=0.7,
            tooltip=f"Sequence {seq_id} Ground Truth"
        ).add_to(m)

        folium.PolyLine(
            pred_coords,
            color="red",
            weight=3,
            opacity=0.7,
            tooltip=f"Sequence {seq_id} Prediction"
        ).add_to(m)

    out_path = os.path.join(os.path.dirname(json_path), output_html)
    m.save(out_path)
    print(f"Mappa salvata in: {out_path}")

    return out_path

# =========================
# MAIN
# =========================

def main():
    # 1) Trovo il JSON di prediction sul Desktop
    json_path = find_json_on_desktop(basename="test_predictions")
    if json_path is None:
        print("\n[ERRORE] Nessun file 'test_predictions*.json' trovato sul Desktop.")
        print("Controlla che il file NON sia un collegamento (.lnk) e che abbia estensione .json.")
        return

        # 2) Menu per scegliere quante sequenze plottare
    while True:
        print("\nQuante sequenze vuoi plottare?")
        print("  [1] 10 sequenze")
        print("  [2] 30 sequenze")
        print("  [3] 100 sequenze")
        print("  [4] Tutte le sequenze")
        print("  [5] Numero personalizzato")
        choice = input("Seleziona un'opzione (1-5): ").strip()

        if choice == "1":
            max_sequences = 10
            break
        elif choice == "2":
            max_sequences = 30
            break
        elif choice == "3":
            max_sequences = 100
            break
        elif choice == "4":
            max_sequences = None  # tutte
            break
        elif choice == "5":
            custom = input("Inserisci il numero di sequenze da plottare: ").strip()
            try:
                max_sequences = int(custom)
                if max_sequences <= 0:
                    print("Per favore inserisci un numero positivo.")
                    continue
                break
            except ValueError:
                print("Valore non valido, riprova.")
        else:
            print("Scelta non valida, riprova.")

    # 3) Genero la mappa con il numero scelto
    visualize_sequences(json_path, max_sequences=max_sequences)



    # 3) Trovo il file di log nella stessa cartella e plotto le loss
    log_path = find_log_in_same_dir(json_path)
    if log_path is None:
        print("\n[ATTENZIONE] Nessun file di log (*.out / *.log) trovato nella stessa cartella del JSON.")
        print("Salterò il grafico della loss.")
        return

    print(f"\nUso il file di log: {log_path}")
    cumulative_times, train_losses, val_losses = parse_log(log_path)

    if not cumulative_times:
        print("[ATTENZIONE] Nessuna epoch riconosciuta nel log, niente grafico della loss.")
        return

    # Salvo il grafico della loss nello stesso posto del JSON
    loss_png_path = os.path.join(os.path.dirname(json_path), "loss_vs_time.png")
    plot_losses(cumulative_times, train_losses, val_losses, save_path=loss_png_path)


if __name__ == "__main__":
    main()