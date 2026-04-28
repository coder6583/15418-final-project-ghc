import numpy as np

SPEED_OF_SOUND = 343.0


def generate_mic_data(
    mic_positions,
    source_position,
    fs=16000,
    duration=0.25,
    source_freq=1000,
    noise_std=0.02,
    source_start=0.05,
    source_duration=0.04,
):
    mic_positions = np.asarray(mic_positions, dtype=float)
    source_position = np.asarray(source_position, dtype=float)

    num_mics = mic_positions.shape[0]
    num_samples = int(fs * duration)

    # Source signal
    source = np.zeros(num_samples)
    start_idx = int(source_start * fs)
    burst_len = int(source_duration * fs)
    end_idx = min(start_idx + burst_len, num_samples)

    burst_t = np.arange(end_idx - start_idx) / fs
    envelope = np.hanning(end_idx - start_idx)
    burst = np.sin(2 * np.pi * source_freq * burst_t) * envelope

    source[start_idx:end_idx] = burst

    # Compute delays
    distances = np.linalg.norm(mic_positions - source_position, axis=1)
    arrival_times = distances / SPEED_OF_SOUND
    relative_times = arrival_times - np.min(arrival_times)
    delays = relative_times * fs

    signals = np.zeros((num_mics, num_samples))

    for i in range(num_mics):
        shifted = np.arange(num_samples) - delays[i]
        signals[i] = np.interp(
            shifted,
            np.arange(num_samples),
            source,
            left=0.0,
            right=0.0,
        )

        signals[i] /= max(distances[i], 0.1)
        signals[i] += np.random.normal(0, noise_std, num_samples)

    return signals


def float_to_int24(x):
    """
    Convert float [-1,1] to signed 24-bit integer
    """
    x = np.clip(x, -1.0, 1.0)
    return (x * (2**23 - 1)).astype(np.int32)

def write_interleaved_txt(signals, mic_positions, filename="mic_data.txt"):
    """
    Format:
    line 1: number of mics
    line 2: mic positions (x0 y0 x1 y1 ...)
    rest: interleaved samples
    """

    num_mics, num_samples = signals.shape

    # normalize globally
    max_val = np.max(np.abs(signals))
    if max_val > 0:
        signals = signals / max_val

    int24 = float_to_int24(signals)

    with open(filename, "w") as f:
        # Line 1: number of microphones
        f.write(f"{num_mics}\n")

        # Line 2: positions
        pos_line = " ".join(f"{x:.4f} {y:.4f}" for x, y in mic_positions)
        f.write(pos_line + "\n")

        # Remaining lines: samples
        for t in range(num_samples):
            row = " ".join(str(int24[m, t]) for m in range(num_mics))
            f.write(row + "\n")

    print(f"Saved {num_samples} samples × {num_mics} mics to {filename}")

def main():
    mic_positions = np.array([
        [-0.10, -0.10],
        [ 0.10, -0.10],
        [ 0.10,  0.10],
        [-0.10,  0.10],
    ])

    source_position = np.array([-0.7, 0.15])

    signals = generate_mic_data(
        mic_positions,
        source_position,
        fs=32000,
        duration=0.25,
    )

    write_interleaved_txt(signals, mic_positions, "mic_data.txt")


if __name__ == "__main__":
    main()
