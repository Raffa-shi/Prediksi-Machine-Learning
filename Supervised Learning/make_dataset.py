import random
import csv
import os

# Nama file dataset
FILENAME = "network_data.csv"

def generate_row():
    # Range realistis
    bandwidth = random.uniform(5, 150)      # Mbps
    latency = random.uniform(5, 300)        # ms
    packet_loss = random.uniform(0, 8)      # %
    uptime = random.uniform(85, 100)        # %

    # Skor risiko gangguan berdasarkan kombinasi parameter
    score = 0

    # Bandwidth rendah -> cenderung gangguan
    if bandwidth < 20:
        score += 2
    elif bandwidth < 50:
        score += 1

    # Latency tinggi -> cenderung gangguan
    if latency > 220:
        score += 2
    elif latency > 130:
        score += 1

    # Packet loss tinggi -> cenderung gangguan
    if packet_loss > 4:
        score += 2
    elif packet_loss > 1:
        score += 1

    # Uptime rendah -> cenderung gangguan
    if uptime < 92:
        score += 2
    elif uptime < 96:
        score += 1

    # Konversi skor jadi probabilitas gangguan (dengan noise biar gak terlalu rapi)
    base_prob = 0.08 * score  # 0, 0.08, 0.16, ... sampai sekitar 0.64
    noise = random.uniform(-0.08, 0.08)
    p_gangguan = base_prob + noise

    # Clamp supaya tetap di [0.05, 0.95]
    p_gangguan = max(0.05, min(0.95, p_gangguan))

    label = "gangguan" if random.random() < p_gangguan else "normal"

    return [
        round(bandwidth, 2),
        round(latency, 2),
        round(packet_loss, 2),
        round(uptime, 2),
        label,
    ]


def main():
    path = os.path.join(os.path.dirname(__file__), FILENAME)
    n_rows = 11000

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bandwidth", "latency", "packet_loss", "uptime", "label"])
        for _ in range(n_rows):
            writer.writerow(generate_row())

    print(f"Dataset {FILENAME} berhasil dibuat dengan {n_rows} baris di {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
