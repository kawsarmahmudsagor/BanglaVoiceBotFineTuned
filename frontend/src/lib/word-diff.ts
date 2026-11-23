export type DiffOp = { op: "eq" | "del" | "ins" | "sub"; ref?: string; hyp?: string };

const words = (s: string) => s.split(/\s+/).filter(Boolean);

/** Word-level edit alignment of ref against hyp (Levenshtein; adjacent del+ins pairs become sub). */
export function wordDiff(ref: string, hyp: string): DiffOp[] {
  const a = words(ref);
  const b = words(hyp);
  const d: number[][] = Array.from({ length: a.length + 1 }, () => new Array<number>(b.length + 1).fill(0));
  for (let i = 0; i <= a.length; i++) d[i][0] = i;
  for (let j = 0; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++)
    for (let j = 1; j <= b.length; j++)
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
  const out: DiffOp[] = [];
  let i = a.length;
  let j = b.length;
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && a[i - 1] === b[j - 1] && d[i][j] === d[i - 1][j - 1]) {
      out.push({ op: "eq", ref: a[i - 1], hyp: b[j - 1] });
      i--; j--;
    } else if (i > 0 && j > 0 && d[i][j] === d[i - 1][j - 1] + 1) {
      out.push({ op: "sub", ref: a[i - 1], hyp: b[j - 1] });
      i--; j--;
    } else if (i > 0 && d[i][j] === d[i - 1][j] + 1) {
      out.push({ op: "del", ref: a[i - 1] });
      i--;
    } else {
      out.push({ op: "ins", hyp: b[j - 1] });
      j--;
    }
  }
  return out.reverse();
}
