import random
import csv
import os

FILENAME = "real_network.csv"


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def generate_normal():
    # NORMAL (kondisi stabil)
    bandwidth = random.uniform(30, 150)            # Mbps
    latency = random.uniform(5, 60)               # ms
    packet_loss = random.uniform(0, 1.2)          # %
    uptime = random.uniform(98.5, 100)            # %

    # noise tambahan kecil biar ga rapih
    latency += random.uniform(-3, 3)
    packet_loss += random.uniform(-0.1, 0.1)

    latency = clamp(latency, 1, 80)
    packet_loss = clamp(packet_loss, 0, 3)
    uptime = clamp(uptime, 97, 100)

    return [bandwidth, latency, packet_loss, uptime, "normal"]


def generate_gangguan_congestion():
    # Gangguan tipe 1: congestion/overload
    bandwidth = random.uniform(5, 40)
    latency = random.uniform(80, 220)
    packet_loss = random.uniform(1.0, 4.5)
    uptime = random.uniform(97.0, 99.0)

    # spike kecil
    if random.random() < 0.2:
        latency += random.uniform(20, 60)

    return [bandwidth, latency, packet_loss, uptime, "gangguan"]


def generate_gangguan_link_unstable():
    # Gangguan tipe 2: link unstable / packet loss tinggi
    bandwidth = random.uniform(1, 30)
    latency = random.uniform(150, 320)
    packet_loss = random.uniform(4.0, 15.0)
    uptime = random.uniform(90.0, 97.8)

    # random spike extrim
    if random.random() < 0.25:
        packet_loss += random.uniform(2, 8)
        latency += random.uniform(30, 90)

    packet_loss = clamp(packet_loss, 0, 30)
    latency = clamp(latency, 1, 500)

    return [bandwidth, latency, packet_loss, uptime, "gangguan"]


def generate_gangguan_routing():
    # Gangguan tipe 3: routing/peering issue
    bandwidth = random.uniform(30, 120)  # bandwidth masih bisa ok
    latency = random.uniform(120, 300)   # latency tinggi
    packet_loss = random.uniform(0.5, 5.0)
    uptime = random.uniform(96.0, 99.5)

    # jitter routing (kadang loss kecil tapi latency parah)
    if random.random() < 0.35:
        packet_loss = random.uniform(0.0, 1.8)

    return [bandwidth, latency, packet_loss, uptime, "gangguan"]


def generate_row():
    """
    Distribusi realistis:
    - 75% normal
    - 25% gangguan (tiga tipe)
    """
    p = random.random()
    if p < 0.75:
        row = generate_normal()
    else:
        # pilih tipe gangguan
        g = random.random()
        if g < 0.45:
            row = generate_gangguan_congestion()
        elif g < 0.80:
            row = generate_gangguan_link_unstable()
        else:
            row = generate_gangguan_routing()

    # rapihin angka
    row[0] = round(row[0], 2)  # bandwidth
    row[1] = round(row[1], 2)  # latency
    row[2] = round(row[2], 2)  # packet_loss
    row[3] = round(row[3], 2)  # uptime
    return row


def main():
    path = os.path.join(os.path.dirname(__file__), FILENAME)

    # ubah sesuai kebutuhan: 5000 / 7000 / 10000
    n_rows = 10000

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bandwidth", "latency", "packet_loss", "uptime", "label"])
        for _ in range(n_rows):
            writer.writerow(generate_row())

    print(f"Dataset monitoring realistis berhasil dibuat: {n_rows} baris")
    print("Lokasi:", os.path.abspath(path))


if __name__ == "__main__":
    main()
