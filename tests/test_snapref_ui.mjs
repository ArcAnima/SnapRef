import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

const source = readFileSync(new URL('../comfyui_nodes/snapref/web/snapref.js', import.meta.url), 'utf8');

test('a removed slot cannot receive its pending upload', async () => {
  const body = source.match(/const setFile = async \(f\) => \{([\s\S]*?)\n  \};/)[1];
  let finish;
  const context = vm.createContext({ uploadVersion: 0, uploadFile: () => new Promise(resolve => { finish = resolve; }),
    w: { name: { value: 'B.png', options: { values: [] } } }, render() {}, node: { setDirtyCanvas() {} } });
  const setFile = vm.runInContext(`(async (f) => {${body}})`, context);
  const pending = setFile({ type: 'image/png' });
  context.uploadVersion++;
  context.w.name.value = 'C.png';
  finish('replacement-B.png');
  await pending;
  assert.equal(context.w.name.value, 'C.png');
  const current = setFile({ type: 'image/png' });
  finish('new-C.png');
  await current;
  assert.equal(context.w.name.value, 'new-C.png');
});

const topLevel = (name) => source.match(new RegExp(`(?:async )?function ${name}\\([\\s\\S]*?\\n\\}\\n`))[0];

test('pasted or node-dropped pictures go to a free slot, not always the first', () => {
  const pickSlot = vm.runInNewContext(`(${topLevel('pickSlot')})`);
  const plain = (r) => JSON.parse(JSON.stringify(r));
  assert.deepEqual(plain(pickSlot([false, false, false], 2, 8)), { slot: 0, shown: 2 });
  assert.deepEqual(plain(pickSlot([true, false, false], 2, 8)), { slot: 1, shown: 2 });
  assert.deepEqual(plain(pickSlot([true, true, false], 2, 8)), { slot: 2, shown: 3 });
  assert.deepEqual(plain(pickSlot([true, false, true, true], 4, 8)), { slot: 1, shown: 4 });
  assert.deepEqual(plain(pickSlot(Array(8).fill(true), 8, 8)), { slot: 0, shown: 8 });
});

test('cards take files and images dragged from ComfyUI itself, not other sites', async () => {
  const fetched = [];
  const context = vm.createContext({
    location: { href: 'http://127.0.0.1:8188/', origin: 'http://127.0.0.1:8188' }, URL,
    File: class { constructor(parts, name, opts) { this.name = name; this.type = opts.type; } },
    fetch: async (url) => { fetched.push(String(url)); return { ok: true, blob: async () => ({ type: 'image/png' }) }; },
  });
  vm.runInContext(`${topLevel('transferSource')}\n${topLevel('resolveImage')}`, context);
  const dt = (files, uri = '') => ({ files, getData: (t) => (t === 'text/uri-list' ? uri : '') });
  const png = { type: 'image/png', name: 'a.png' };
  const pick = (d) => vm.runInContext('(d) => resolveImage(transferSource(d))', context)(d);
  assert.equal(await pick(dt([png])), png);
  assert.equal(await pick(dt([{ type: 'text/plain' }])), null);
  const own = await pick(dt([], 'http://127.0.0.1:8188/api/view?filename=out_0001.png&type=output'));
  assert.equal(own.name, 'out_0001.png');
  assert.equal(await pick(dt([], 'https://example.com/x.png')), null);
  assert.equal(await pick(dt([], '# comment')), null);
  assert.deepEqual(fetched, ['http://127.0.0.1:8188/api/view?filename=out_0001.png&type=output']);
});

test('restoring saved rows preserves height; adding a row grows once', () => {
  const body = source.match(/const syncAll = \(resize = true\) => \{([\s\S]*?)\n      \};/)[1];
  const node = { properties: { snapref_slots: 3 }, size: [440, 800], computeSize: () => [440, 620], setSize(size) { this.size = size; } };
  const context = vm.createContext({ rows: 1, MAX_REFERENCES: 8, node, segButtons: [], modeW: {},
    refWidgets: [{ name: { value: 'A' } }, { name: { value: 'B' } }, { name: { value: 'C' } }],
    pictures: Array.from({ length: 8 }, () => ({ el: {}, render() {} })),
    viewUrl: name => name || null, add: {}, panel: { style: {} }, minHeight: () => 620, fid: { sync() {} } });
  const sync = vm.runInContext(`((resize = true) => {${body}})`, context);
  sync(false);
  sync();
  assert.equal(node.size[1], 800);
  node.properties.snapref_slots = 5;
  sync();
  sync();
  assert.equal(node.size[1], 1040);
});
