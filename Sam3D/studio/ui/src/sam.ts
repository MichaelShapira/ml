// In-browser SAM segmentation (transformers.js + WebGPU). The image encoder runs
// once per image; each click only runs the lightweight prompt decoder.
import { SamModel, AutoProcessor, RawImage, Tensor } from "@huggingface/transformers";

const MODEL_ID = "Xenova/slimsam-77-uniform";

export interface MaskResult { data: Uint8Array; width: number; height: number; }

export class SamSession {
  private model: any = null;
  private processor: any = null;
  private inputs: any = null;        // processor outputs for the current image
  private embeddings: any = null;    // cached image embedding
  private maskHead: number | null = null; // locked mask candidate for a selection
  device = "wasm";

  async load(): Promise<string> {
    // WebGPU on mobile GPUs/drivers is unreliable for this model (onnxruntime-web
    // throws bind-group/Softmax validation errors, and fp16 yields noisy masks).
    // Mobile is small/distilled enough to run on WASM (CPU) reliably, so we only
    // use WebGPU on desktop. fp32 keeps masks numerically clean.
    const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent)
      || (navigator.maxTouchPoints > 1 && /Macintosh/.test(navigator.userAgent)); // iPadOS
    let device = "wasm";
    if (!isMobile) {
      try {
        const gpu = (navigator as any).gpu;
        const adapter = gpu?.requestAdapter ? await gpu.requestAdapter() : null;
        if (adapter) device = "webgpu";
      } catch {
        /* keep wasm */
      }
    }
    this.device = device;
    this.model = await SamModel.from_pretrained(MODEL_ID, { dtype: "fp32", device: device as any });
    this.processor = await AutoProcessor.from_pretrained(MODEL_ID);
    return this.device;
  }

  // Encode an image once (expensive); returns the RawImage for display.
  async setImage(url: string): Promise<RawImage> {
    const image = await RawImage.read(url);
    this.inputs = await this.processor(image);
    this.embeddings = await this.model.get_image_embeddings(this.inputs);
    this.maskHead = null; // new image → start a fresh selection
    return image;
  }

  // points/labels in NATURAL image pixel coords; label 1 = foreground, 0 = background.
  async segment(width: number, height: number, points: number[][], labels: number[]): Promise<MaskResult | null> {
    if (!this.embeddings || points.length === 0) return null;
    const reshaped = this.inputs.reshaped_input_sizes[0]; // [h, w]
    const pts = points.map(([x, y]) => [
      (x / width) * reshaped[1],
      (y / height) * reshaped[0],
    ]);
    const input_points = new Tensor("float32", pts.flat(Infinity) as number[], [1, 1, pts.length, 2]);
    const input_labels = new Tensor("int64", labels.map((l) => BigInt(l)), [1, 1, labels.length]);

    const { pred_masks, iou_scores } = await this.model({
      ...this.embeddings,
      input_points,
      input_labels,
    });
    const masks = await this.processor.post_process_masks(
      pred_masks,
      this.inputs.original_sizes,
      this.inputs.reshaped_input_sizes
    );
    const maskTensor = masks[0][0];          // [num, H, W]
    const [num, H, W] = maskTensor.dims as number[];
    const scores = iou_scores.data as Float32Array;
    const md = maskTensor.data as Uint8Array | Float32Array;

    // SlimSAM returns 3 candidate masks at different granularities and has no
    // mask-feedback input, so switching candidates between clicks makes the mask
    // "jump" and re-cover removed areas. To keep refinement stable, lock onto one
    // candidate (head) for the whole selection: the first point picks it by IoU,
    // later Keep/Remove points refine within that same head. A fresh first point
    // (points.length === 1) re-locks, so starting over works.
    if (this.maskHead == null || points.length <= 1) {
      let head = 0;
      for (let i = 1; i < num; i++) if (scores[i] > scores[head]) head = i;
      this.maskHead = head;
    }
    const best = Math.min(this.maskHead, num - 1);

    const out = new Uint8Array(H * W);
    const off = best * H * W;
    let set = 0;
    for (let i = 0; i < H * W; i++) {
      const v = (md[off + i] as number) > 0 ? 255 : 0; // handle bool or logit dtypes
      out[i] = v;
      if (v) set++;
    }
    // No pixels selected (e.g. tapped empty space, or a bad GPU run): signal the
    // caller so the UI can ask the user to try a different spot.
    if (set === 0) return null;
    return { data: out, width: W, height: H };
  }
}
