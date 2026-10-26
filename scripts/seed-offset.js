// Preloaded by scripts/seed-sweep.js: shifts every mulberry32 seed that
// tests and simulators draw from conscious-agent by SEED_OFFSET, so the suite
// can be rerun on fresh randomness without editing it.
const ca = require('conscious-agent');
const offset = Number(process.env.SEED_OFFSET || 0);
const base = ca.mulberry32;
ca.mulberry32 = (seed) => base((seed + offset) >>> 0);
