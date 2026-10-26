// Robustness sweep: rerun `node --test` with every seed shifted by k·7919,
// k = 1..N (default 10), and report which offsets fail.
//   node scripts/seed-sweep.js [N] [test files...]
const { spawnSync } = require('node:child_process');
const path = require('node:path');

const N = Number(process.argv[2] || 10);
const files = process.argv.slice(3);
const preload = path.join(__dirname, 'seed-offset.js');
const failed = [];
for (let k = 1; k <= N; k++) {
  const offset = k * 7919;
  const r = spawnSync(process.execPath, ['--test', '--test-reporter=tap', ...files], {
    cwd: path.join(__dirname, '..'),
    env: { ...process.env, SEED_OFFSET: String(offset), NODE_OPTIONS: `${process.env.NODE_OPTIONS || ''} --require ${preload}` },
    encoding: 'utf8',
  });
  const ok = r.status === 0;
  if (!ok) failed.push(offset);
  const fails = (r.stdout.match(/^not ok .*$/gm) || []).join('; ');
  console.log(`offset ${offset}: ${ok ? 'pass' : `FAIL ${fails}`}`);
}
console.log(failed.length ? `${failed.length}/${N} offsets failed: ${failed.join(', ')}` : `all ${N} offsets pass`);
process.exit(failed.length ? 1 : 0);
