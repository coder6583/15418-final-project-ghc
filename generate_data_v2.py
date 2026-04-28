import numpy as np

SPEED_OF_SOUND = 343.0


def generate_chirp_burst(fs, num_samples, start_time, duration):
    source = np.zeros(num_samples)

    start_idx = int(start_time * fs)
    length = int(duration * fs)
    end_idx = min(start_idx + length, num_samples)

    t = np.arange(end_idx - start_idx) / fs
    envelope = np.hanning(len(t))

    # Chirp: 500 Hz → 3500 Hz
    f0 = 500
    f1 = 3500
    T = duration
    k = (f1 - f0) / T

    phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    burst = np.sin(phase) * envelope

    source[start_idx:end_idx] = burst
    return source


def generate_mic_signals(mic_positions, source_pos, fs=16000, duration=0.25):
    num_mics = len(mic_positions)
    num_samples = int(fs * duration)

    # Generate stable chirp burst
    source = generate_chirp_burst(fs, num_samples, start_time=0.05, duration=0.04)

    signals = np.zeros((num_mics, num_samples))

    # Compute delays
    distances = np.linalg.norm(mic_positions - source_pos, axis=1)
    arrival_times = distances / SPEED_OF_SOUND
    relative_times = arrival_times - np.min(arrival_times)
    delays = relative_times * fs

    for i in range(num_mics):
        shifted = np.arange(num_samples) - delays[i]

        signals[i] = np.interp(
            shifted,
            np.arange(num_samples),
            source,
            left=0.0,
            right=0.0
        )

        # distance attenuation
        signals[i] /= max(distances[i], 0.1)

    return signals, delays


def float_to_int24(x):
    x = np.clip(x, -1.0, 1.0)
    return (x * (2**23 - 1)).astype(np.int32)


def write_output(signals, mic_positions, filename="mic_data.txt"):
    num_mics, num_samples = signals.shape

    # normalize
    max_val = np.max(np.abs(signals))
    if max_val > 0:
        signals = signals / max_val

    int24 = float_to_int24(signals)

    with open(filename, "w") as f:
        # line 1: number of mics
        f.write(f"{num_mics}\n")

        # line 2: mic positions
        pos_line = " ".join(f"{x:.4f} {y:.4f}" for x, y in mic_positions)
        f.write(pos_line + "\n")

        # data
        for t in range(num_samples):
            row = " ".join(str(int24[m, t]) for m in range(num_mics))
            f.write(row + "\n")

    print(f"Saved to {filename}")


def main():
    # Square mic layout (meters)
    mic_positions = np.array([
        [-0.10, -0.10],
        [ 0.10, -0.10],
        [ 0.10,  0.10],
        [-0.10,  0.10],
    ])

    # Source position (change this to test)
    source_pos = np.array([-0.7, 0.2])

    signals, delays = generate_mic_signals(mic_positions, source_pos, fs=32000)

    print("True delays (samples):")
    for i, d in enumerate(delays):
        print(f"mic {i}: {d:.2f}")

    write_output(signals, mic_positions)


if __name__ == "__main__":
    main()
