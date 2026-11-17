import { CALLER_RATE, concatInt16, encodeWav, rms, trimSilence } from "./wav";
import { openMic } from "./worklet";

/** Raw-voice recorder for baseline clips: echo cancellation, noise suppression and auto-gain are OFF. */
export class BaselineRecorder {
  private ctx = new AudioContext();
  private frames: Int16Array[] = [];
  private mic: { stop(): void } | null = null;

  async start(onLevel: (v: number) => void): Promise<void> {
    this.mic = await openMic(
      this.ctx,
      { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      (f) => {
        this.frames.push(f);
        onLevel(rms(f) / 120);
      },
    );
  }

  /** Stops and returns a trimmed 16 kHz WAV, or null if nothing audible was captured. */
  stop(): Blob | null {
    this.mic?.stop();
    void this.ctx.close();
    const trimmed = trimSilence(concatInt16(this.frames));
    return trimmed ? encodeWav(trimmed, CALLER_RATE) : null;
  }
}
