/* tslint:disable */
/* eslint-disable */

export class WasmModel {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Full-detail classification: every label's probability for every
     * task, with per-task multi-label/threshold/activation/prompt/examples
     * (mirrors every option `gliner-classify`'s interactive shell exposes).
     *
     * `tasks_json` is a JSON array of
     * `{task, labels: [{label, description?}], multiLabel?, threshold?,
     * activation?: "auto"|"sigmoid"|"softmax", prompt?, examples?: [[input, label]]}`.
     *
     * Returns JSON `[{task, labels: [{label, prob}]}]`, labels sorted by
     * descending probability.
     */
    classify(text: string, tasks_json: string): string;
    /**
     * Runs extraction built from the same flag syntax as the `gliner` CLI
     * binary (`--entities`, `--relations`, `--json`, `--classify`). Powers
     * the "generic schema" and PII/guardrail demos. Returns pretty JSON.
     *
     * `entities`: `label` or `label:description`, one per array entry.
     * `structures`: `name=field1::str,field2::[a|b],field3::list::description`.
     * `classify`: `task=label1,label2` (prefix task with `+` for multi-label).
     */
    extractCli(text: string, entities: string[], relations: string[], structures: string[], classify: string[], legacy_structures: boolean, opts_json: string): string;
    /**
     * Loads a checkpoint from its four file contents, in float32 on CPU
     * (the only combination that makes sense in a browser tab).
     *
     * `weights` is `model.safetensors`; `tokenizer` is `tokenizer.json`;
     * `config_json`/`encoder_config_json` are `config.json` and
     * `encoder_config/config.json`.
     */
    static load(weights: Uint8Array, tokenizer: Uint8Array, config_json: string, encoder_config_json: string): WasmModel;
    /**
     * Use the character-level word splitter (Chinese, Japanese, ...).
     */
    setCharSplit(on: boolean): void;
}

/**
 * Call once from JS before anything else, to get readable panic messages
 * (from candle shape mismatches etc.) in the browser console.
 */
export function init_panic_hook(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmmodel_free: (a: number, b: number) => void;
    readonly init_panic_hook: () => void;
    readonly wasmmodel_classify: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly wasmmodel_extractCli: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => [number, number, number, number];
    readonly wasmmodel_load: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly wasmmodel_setCharSplit: (a: number, b: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
