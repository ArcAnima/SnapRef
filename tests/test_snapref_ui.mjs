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
