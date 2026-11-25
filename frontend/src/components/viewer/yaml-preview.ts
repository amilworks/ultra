// YAML aliases share objects, but JSON repeats each referenced value. Charge
// each visit (including repeated keys, strings and indentation) during the
// actual traversal so small source files cannot expand without bound.
export function stringifyYamlPreview(value: unknown, budget: number): string {
  const childDepth = new WeakMap<object, number>();
  let remaining = budget;

  return JSON.stringify(value, function (this: object, key: string, entry: unknown) {
    const depth = childDepth.get(this) ?? 0;
    remaining -= 1 + depth * 2 + key.length;
    if (typeof entry === "string") remaining -= entry.length;
    if (remaining < 0) throw new RangeError("YAML exceeds the JSON preview budget");
    if (entry !== null && typeof entry === "object") childDepth.set(entry, depth + 1);
    return entry;
  }, 2);
}
