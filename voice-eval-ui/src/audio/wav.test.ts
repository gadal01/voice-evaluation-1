import { describe, expect, it } from "vitest";
import {
  buildTimeline, concatInt16, encodeWav, fitTo, framesDue, sessionSeconds, toFrames, trimSilence, truncateAgent,
  type AgentChunk,
} from "./wav";

const ramp = (n: number, v = 1000) => Int16Array.from({ length: n }, () => v);

describe("encodeWav", () => {
  it("writes a valid 16-bit mono header", async () => {
    const blob = encodeWav(ramp(100), 16000);
    const v = new DataView(await blob.arrayBuffer());
    expect(blob.size).toBe(44 + 200);
    expect(String.fromCharCode(v.getUint8(0), v.getUint8(1), v.getUint8(2), v.getUint8(3))).toBe("RIFF");
    expect(v.getUint16(22, true)).toBe(1); // channels
    expect(v.getUint32(24, true)).toBe(16000); // rate
    expect(v.getUint16(34, true)).toBe(16); // bits
    expect(v.getUint32(40, true)).toBe(200); // data bytes
  });
});

describe("truncateAgent (barge-in)", () => {
  const rate = 1000;
  const chunks: AgentChunk[] = [
    { start: 0, pcm: ramp(500) }, // 0.0-0.5 s
    { start: 0.5, pcm: ramp(500) }, // 0.5-1.0 s
    { start: 1.0, pcm: ramp(500) }, // 1.0-1.5 s
  ];
  it("drops chunks after the interrupt and cuts the one spanning it", () => {
    const out = truncateAgent(chunks, 0.75, rate);
    expect(out).toHaveLength(2);
    expect(out[1]!.pcm.length).toBe(250);
  });
  it("keeps everything already fully played", () => {
    expect(truncateAgent(chunks, 5, rate)).toHaveLength(3);
  });
});

describe("timeline", () => {
  it("places chunks at their play position and clips at the end", () => {
    const t = buildTimeline([{ start: 0.5, pcm: ramp(600, 7) }], 1, 1000);
    expect(t.length).toBe(1000);
    expect(t[499]).toBe(0);
    expect(t[500]).toBe(7);
    expect(t[999]).toBe(7);
  });
  it("fitTo pads and cuts", () => {
    expect(fitTo(ramp(10), 1, 100)).toHaveLength(100);
    expect(fitTo(ramp(500), 1, 100)).toHaveLength(100);
  });
  it("sessionSeconds is the longest of the three clocks", () => {
    expect(sessionSeconds(16000 * 3, 2, [{ start: 4, pcm: new Int16Array(24000) }], 24000)).toBe(5);
  });
});

describe("clip sender maths", () => {
  it("splits into 480-sample frames and pads the tail", () => {
    const f = toFrames(ramp(1000));
    expect(f).toHaveLength(3);
    expect(f.every((x) => x.length === 480)).toBe(true);
    expect(f[2]![519 - 480 + 1]).toBe(0); // padded
  });
  it("catches up when a timer tick is late", () => {
    expect(framesDue(90, 0)).toBe(3);
    expect(framesDue(300, 4)).toBe(6); // 10 due, 4 sent
    expect(framesDue(10, 5)).toBe(0);
  });
});

describe("trimSilence", () => {
  it("trims quiet edges and keeps a margin", () => {
    const a = concatInt16([new Int16Array(10000), ramp(8000, 5000), new Int16Array(10000)]);
    const out = trimSilence(a)!;
    expect(out.length).toBe(8000 + 2 * 2400);
  });
  it("returns null for near-silence", () => {
    expect(trimSilence(new Int16Array(20000))).toBeNull();
  });
});
