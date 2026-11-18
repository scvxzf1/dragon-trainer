declare module "*domain/labels-options.js" {
  export const FIELD_LABEL_ZH: Record<string, string>;
  export const FIELD_OPTIONS: Record<string, (string | number | boolean)[]>;
  export const FIELD_OPTION_LABELS: Record<string, Record<string, string>>;
}
declare module "*domain/config-field-types.js" {
  export function configFieldInputKind(
    key: string,
    value: unknown,
    options?: unknown[] | null,
  ): "text" | "number" | "boolean" | "select" | "json";
  export function normalizeBooleanConfigValue(
    key: string,
    value: unknown,
    fallback?: boolean,
  ): boolean;
  export function isBooleanConfigField(key: string, value?: unknown): boolean;
}
declare module "*domain/field-help-summary.js" {
  export const FIELD_HELP_SUMMARY_ZH: Record<string, string>;
}
declare module "*domain/config-field-catalog.js" {
  export const CONFIG_STAGE_META: readonly { id: string; clusters: readonly { id: string; label: string }[] }[];
  export const CONFIG_FIELD_CATALOG: Record<
    string,
    {
      stage: "input" | "method" | "training" | "resources";
      location: string;
      cluster: string;
      siblingOrder: number;
      controlledBy: string[];
      dependsOn: string[];
    }
  >;
  export function configFieldCatalogEntry(key: string): {
    location: string;
    stage: string;
    cluster: string;
    siblingOrder?: number;
  };
}
declare module "*domain/config-values.js" {
  export function displayConfigValue(
    key: string,
    config: Record<string, unknown>,
  ): unknown;
  export function prepareConfigPatch(
    changes: Record<string, unknown>,
    original: Record<string, unknown>,
  ): Record<string, unknown>;
}
declare module "*domain/config-field-availability.js" {
  export function configFieldAvailability(
    key: string,
    context: Record<string, unknown>,
  ): { enabled: boolean; reason: string; code: string | null };
  export function resolveConfigAdapterKind(
    values: Record<string, unknown>,
  ): string;
}
declare module "*domain/config-field-disclosure-rules.js" {
  export function configFieldDisclosure(
    key: string,
    context: Record<string, unknown>,
  ): { visible: boolean; reason: string; code: string | null };
}
declare module "*domain/model-family.js" {
  export const FALLBACK_MODEL_FAMILIES: readonly {
    name: string;
    display_name: string;
    aliases: string[];
  }[];
  export function configureModelFamilyCapabilities(payload: unknown): unknown[];
  export function normalizeModelFamily(value: unknown): string;
  export function isKrea2ModelFamily(value: unknown): boolean;
  export function modelFamilyOptionSupported(
    key: string,
    family: string,
    option: string,
    context?: Record<string, unknown> | string,
  ): boolean;
}
