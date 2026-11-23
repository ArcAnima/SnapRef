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

const plain = (r) => JSON.parse(JSON.stringify(r));
const KEEP = '1 keeps every detail of the picture · lower follows your prompt more freely';

test('modeCopy gives the style copy and falls back to cs', () => {
  const modeCopy = vm.runInNewContext(`(${topLevel('modeCopy')})`);
  assert.deepEqual(plain(modeCopy('Style')), { key: 'style', accent: '#a98bf5', tag: 'STYLE',
    empty1: 'Artwork in the style you want', emptyN: 'optional · same artist', prompt: 'Write the character and the scene',
    fidelity: "1 follows the picture's look closely · lower follows your prompt more freely" });
  assert.deepEqual(plain(modeCopy('Character')), { key: 'char', accent: '#5b9cf5', tag: 'REFERENCE',
    empty1: 'Character picture', emptyN: 'optional · same character',
    prompt: 'Write the scene · add artist tags for the style', fidelity: KEEP });
  const cs = { key: 'cs', accent: '#ff6b35', tag: 'REFERENCE', empty1: 'Character picture',
    emptyN: 'optional · same character', prompt: 'Write only the scene', fidelity: KEEP };
  assert.deepEqual(plain(modeCopy('Character & Style')), cs);
  for (const unknown of ['Something else', undefined, null, '']) assert.deepEqual(plain(modeCopy(unknown)), cs);
  assert.equal(modeCopy('style').key, 'style');   // the node's mode keys work too
});

test('styleSupport reads the adapter table', () => {
  const styleSupport = vm.runInNewContext(`(${topLevel('styleSupport')})`);
  const info = { adapters: {
    'snapref_v2s_pre.safetensors': { modes: ['cs', 'char', 'style'], style: true },
    'snapref_v2a.safetensors': { modes: ['cs', 'char'], style: false },
    'snapref_partial.safetensors': { error: 'ValueError: header cut short (file incomplete)' },
  } };
  assert.equal(styleSupport(info, 'snapref_v2s_pre.safetensors'), true);
  assert.equal(styleSupport(info, 'snapref_v2a.safetensors'), false);
  assert.equal(styleSupport(info, 'snapref_partial.safetensors'), null);   // error: unknown, keep Style usable
  assert.equal(styleSupport(info, 'snapref_new.safetensors'), null);       // not in the table
  assert.equal(styleSupport(null, 'snapref_v2a.safetensors'), null);       // the request failed
  assert.equal(styleSupport({}, 'snapref_v2a.safetensors'), null);
  assert.equal(styleSupport(info.adapters, 'snapref_v2a.safetensors'), false);   // the bare table works too
});

test('style mode kept with a warning when the adapter lacks it', () => {
  const ctx = vm.createContext({});
  vm.runInContext(`${topLevel('modeNotice')}\n${topLevel('pickMode')}\n${topLevel('modeButton')}`, ctx);
  const notice = 'This adapter has no Style mode. Choose snapref_v2s_pre or another style adapter.';
  assert.equal(ctx.modeNotice('Style', false), notice);
  assert.equal(ctx.modeNotice('Style', null), '');
  assert.equal(ctx.modeNotice('Style', true), '');
  assert.equal(ctx.modeNotice('Character', false), '');
  // a saved Style value stays Style; clicking the dim Style button changes nothing, the others still work
  assert.equal(ctx.pickMode('Style', 'Style', false), 'Style');
  assert.equal(ctx.pickMode('Character', 'Style', false), 'Character');
  assert.equal(ctx.pickMode('Style', 'Character', false), 'Character');
  assert.equal(ctx.pickMode('Character', 'Style', null), 'Style');
  assert.equal(ctx.pickMode('Character', 'Style', true), 'Style');
  const tip = 'Style needs an adapter with a style branch (e.g. snapref_v2s_pre)';
  assert.deepEqual(plain(ctx.modeButton('Style', 'Style', false)), { on: true, dis: true, title: tip });
  assert.deepEqual(plain(ctx.modeButton('Style', 'Character', false)), { on: false, dis: true, title: tip });
  assert.deepEqual(plain(ctx.modeButton('Style', 'Character', null)), { on: false, dis: false, title: '' });
  assert.deepEqual(plain(ctx.modeButton('Character', 'Character', false)), { on: true, dis: false, title: '' });
});

test('the mode bar lays out every mode in one row', () => {
  const seg = source.match(/\.sr-seg \{[^}]*\}/)[0];
  assert.match(seg, /grid-auto-flow: column/);
  assert.doesNotMatch(seg, /grid-template-columns/);
  assert.match(source, /\.sr-seg button\.dis \{/);
});
