function pad(n: number, width = 2) {
  return String(n).padStart(width, "0");
}

export function formatTimestamp(date = new Date()): string {
  const Y = date.getFullYear();
  const M = pad(date.getMonth() + 1); // months are 0-based
  const D = pad(date.getDate());
  const h = pad(date.getHours());
  const m = pad(date.getMinutes());
  const s = pad(date.getSeconds());
  return `[${Y}-${M}-${D} ${h}:${m}:${s}]`;
}
