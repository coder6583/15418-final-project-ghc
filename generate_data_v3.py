import numpy as np

SPEED_OF_SOUND = 343.0


def generate_source(fs, num_samples, kind="chirp_burst"):
    t = np.arange(num_samples) / fs
    source = np.zeros(num_samples)

    if kind == "chirp_burst":
        start_time = 0.025
        duration = 0.035

        start = int(start_time * fs)
        length = int(duration * fs)
        end = min(start + length, num_samples)

        bt = np.arange(end - start) / fs
        env = np.hanning(end - start)

        f0 = 500.0
        f1 = 4500.0
        k = (f1 - f0) / duration

        phase = 2 * np.pi * (f0 * bt + 0.5 * k * bt * bt)
        source[start:end] = np.sin(phase) * env

    elif kind == "noise_burst":
        start_time = 0.025
        duration = 0.035

        start = int(start_time * fs)
        length = int(duration * fs)
        end = min(start + length, num_samples)

        env = np.hanning(end - start)
        source[start:end] = np.random.normal(0, 1, end - start) * env

    elif kind == "speech_like":
        # crude speech-like signal: sum of modulated tones + noise
        carrier = (
            0.6 * np.sin(2 * np.pi * 700 * t) +
            0.3 * np.sin(2 * np.pi * 1400 * t) +
            0.2 * np.sin(2 * np.pi * 2300 * t)
        )
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 6 * t)
        source = carrier * envelope
        source += 0.15 * np.random.normal(0, 1, num_samples)

    return source


def fractional_delay(signal, delay_samples):
    idx = np.arange(len(signal)) - delay_samples
    return np.interp(idx, np.arange(len(signal)), signal, left=0.0, right=0.0)


def add_echoes(signal, fs, echoes):
    """
    echoes: list of (delay_seconds, gain)
    """
    out = signal.copy()
    for delay_s, gain in echoes:
        delay_samples = delay_s * fs
        out += gain * fractional_delay(signal, delay_samples)
    return out


def generate_mic_signals(
    mic_positions,
    source_pos,
    fs=32000,
    duration=0.125,
    source_kind="chirp_burst",
    noise_std=0.015,
    mic_gain_std=0.08,
    timing_jitter_samples=0.15,
    include_echoes=True,
    seed=0,
):
    rng = np.random.default_rng(seed)

    mic_positions = np.asarray(mic_positions, dtype=float)
    source_pos = np.asarray(source_pos, dtype=float)

    num_mics = mic_positions.shape[0]
    num_samples = int(fs * duration)

    np.random.seed(seed)
    source = generate_source(fs, num_samples, source_kind)

    distances = np.linalg.norm(mic_positions - source_pos, axis=1)
    arrival_times = distances / SPEED_OF_SOUND
    relative_times = arrival_times - np.min(arrival_times)
    delays = relative_times * fs

    signals = np.zeros((num_mics, num_samples))

    for m in range(num_mics):
        # small per-mic timing jitter
        jitter = rng.normal(0.0, timing_jitter_samples)
        delayed = fractional_delay(source, delays[m] + jitter)

        # distance attenuation
        delayed /= max(distances[m], 0.15)

        # per-mic gain mismatch
        gain = 1.0 + rng.normal(0.0, mic_gain_std)
        delayed *= gain

        # simple multipath / room reflection
        if include_echoes:
            echoes = [
                (0.003 + rng.normal(0, 0.0003), 0.25),
                (0.007 + rng.normal(0, 0.0005), 0.12),
            ]
            delayed = add_echoes(delayed, fs, echoes)

        # low-frequency hum + white sensor noise
        t = np.arange(num_samples) / fs
        hum = 0.01 * np.sin(2 * np.pi * 60 * t + rng.uniform(0, 2*np.pi))
        noise = rng.normal(0.0, noise_std, num_samples)

        signals[m] = delayed + hum + noise

    return signals, delays, distances


def float_to_int24(x):
    x = np.clip(x, -1.0, 1.0)
    return (x * (2**23 - 1)).astype(np.int32)


def write_output(signals, mic_positions, filename="mic_data.txt"):
    num_mics, num_samples = signals.shape

    max_val = np.max(np.abs(signals))
    if max_val > 0:
        signals = signals / max_val

    int24 = float_to_int24(signals)

    with open(filename, "w") as f:
        f.write(f"{num_mics}\n")
        f.write(" ".join(f"{x:.4f} {y:.4f}" for x, y in mic_positions) + "\n")

        for t in range(num_samples):
            f.write(" ".join(str(int24[m, t]) for m in range(num_mics)) + "\n")

    print(f"wrote {num_samples} samples x {num_mics} mics to {filename}")


def main():
    mic_positions = np.array([
        [-0.10, -0.10],
        [ 0.10, -0.10],
        [ 0.10,  0.10],
        [-0.10,  0.10],
    ])

    source_pos = np.array([0.30, 0.20])

    signals, delays, distances = generate_mic_signals(
        mic_positions=mic_positions,
        source_pos=source_pos,
        fs=32000,
        duration=0.125,
        source_kind="chirp_burst",
        noise_std=0.015,
        mic_gain_std=0.08,
        timing_jitter_samples=0.15,
        include_echoes=True,
        seed=42,
    )

    print("true relative delays, samples:")
    for i, d in enumerate(delays):
        print(f"mic {i}: {d:.3f}")

    print("true pair lags relative to mic 0:")
    for i in range(1, len(delays)):
        print(f"pair (0,{i}): {delays[i] - delays[0]:.3f}")

    write_output(signals, mic_positions, "mic_data.txt")


if __name__ == "__main__":
    main()
