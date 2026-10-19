"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "../realtime_translate/web/index.html"), "utf8");
const source = html.slice(html.indexOf("function stampDoa("), html.indexOf("function sendCmd("));
assert.ok(source.includes("function handle(msg)"));

function fixture(){
  const cards = new Map();
  function card(){
    const cursor = {removed: false, remove(){ this.removed = true; }};
    const nodes = {
      ".cursor": cursor,
      ".trans": {textContent: "", className: "trans", title: ""},
      ".pill.src": {textContent: "partial", className: "pill src"},
      ".pill.doa": {hidden: true, textContent: ""},
      ".src-text": {textContent: ""}
    };
    return {
      dataset: {}, removed: false,
      querySelector(selector){ return selector === ".cursor" && cursor.removed ? null : nodes[selector]; },
      remove(){ this.removed = true; }
    };
  }
  const context = {
    cards, targets: {zh_to: "en", other_to: "zh"},
    cardFor(id){ if (!cards.has(id)) cards.set(id, card()); return cards.get(id); },
    removeEmpty(){}, nearBottom(){ return true; }, scrollIfStuck(){}, refreshRuns(){},
    setSide(el, side){ el.side = side; }, guessSide(){ return "left"; },
    langName(code){ return code; },
    stream: {appendChild(){}}, empty: {}, toast(){},
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  return context;
}

for (const type of ["source", "final"]){
  const ctx = fixture();
  ctx.handle({type: "partial", id: "seg", text: "临时字幕"});
  ctx.handle({type, id: "seg", text: "最终字幕", lang: "zh", doa: "front"});
  const el = ctx.cards.get("seg");
  assert.equal(el.dataset.finalized, "true");
  assert.equal(el.querySelector(".cursor"), null);
  assert.equal(el.querySelector(".trans").className, "trans pending");
  ctx.handle({type: "partial", id: "seg", text: "迟到字幕"});
  assert.equal(el.querySelector(".src-text").textContent, "最终字幕");
  assert.equal(el.side, "right");
  ctx.handle({type: "translation", id: "seg", text: "", error: "GPU failure"});
  assert.equal(el.querySelector(".trans").textContent, "翻译失败");
  assert.equal(el.querySelector(".trans").className, "trans");
  assert.equal(el.querySelector(".trans").title, "GPU failure");
  ctx.handle({type: "discard", id: "seg"});
  assert.equal(ctx.cards.size, 0);
  ctx.handle({type: "translation", id: "seg", text: "迟到翻译"});
  assert.equal(ctx.cards.size, 0);
}

const ctx = fixture();
ctx.handle({type: "final", id: "seg", text: "原文", lang: "zh"});
ctx.handle({type: "translation", id: "seg", text: "Hello.", target_lang: "en"});
assert.equal(ctx.cards.get("seg").querySelector(".trans").textContent, "Hello.");
assert.equal(ctx.cards.get("seg").querySelector(".trans").className, "trans");

function scrollFixture(){
  const context = fixture();
  const scrolls = [];
  const stream = {
    scrollHeight: 1000, clientHeight: 500, scrollTop: 500,
    scrollTo(options){
      scrolls.push(options);
      this.scrollTop = Math.max(0, Math.min(options.top,
                                          this.scrollHeight - this.clientHeight));
    },
  };
  context.stream = stream;
  const originalCardFor = context.cardFor;
  context.cardFor = id => {
    const fresh = !context.cards.has(id);
    const card = originalCardFor(id);
    if (fresh){
      stream.scrollHeight += 200;
      for (const selector of [".src-text", ".trans"]){
        const node = card.querySelector(selector);
        let text = node.textContent;
        Object.defineProperty(node, "textContent", {
          get(){ return text; },
          set(value){
            const next = String(value);
            stream.scrollHeight += Math.max(0, next.length - text.length);
            text = next;
          },
        });
      }
    }
    return card;
  };
  const scrollSource = html.slice(html.indexOf("function nearBottom("),
                                 html.indexOf("function removeEmpty("));
  vm.runInContext(scrollSource, context);
  return {context, stream, scrolls};
}

// A new card taller than the 80px threshold must remain visible when following
// live subtitles, including subsequent partial growth and translation growth.
{
  const {context, stream, scrolls} = scrollFixture();
  for (const message of [
    {type: "partial", id: "scroll", text: "早上好"},
    {type: "partial", id: "scroll", text: "早上好".repeat(80)},
    {type: "final", id: "scroll", text: "早上好".repeat(100), lang: "zh"},
    {type: "translation", id: "scroll", text: "Good morning. ".repeat(80)},
    {type: "final", id: "next", text: "下一句", lang: "zh"},
  ]){
    context.handle(message);
    assert.equal(stream.scrollTop, stream.scrollHeight - stream.clientHeight,
                 `latest ${message.type} should remain visible`);
    assert.equal(scrolls.at(-1).behavior, "auto");
  }
}

// Reading earlier cards must preserve the reader's position across new cards,
// final text replacement, and translations.
{
  const {context, stream, scrolls} = scrollFixture();
  stream.scrollTop = 0;
  for (const message of [
    {type: "partial", id: "reading", text: "临时字幕"},
    {type: "final", id: "reading", text: "完整字幕".repeat(80), lang: "zh"},
    {type: "translation", id: "reading", text: "Translation. ".repeat(80)},
  ]){
    context.handle(message);
    assert.equal(stream.scrollTop, 0, "new text must not interrupt history reading");
  }
  assert.equal(scrolls.length, 0);
}
console.log("Subtitle frontend regressions passed.");
