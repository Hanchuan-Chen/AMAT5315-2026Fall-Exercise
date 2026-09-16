#!/usr/bin/env node
/**
 * Save the course viewer's own stamped proof image for one temperature.
 *
 * The published page (https://giggleliu.github.io/AMAT5315-2026Fall/week3-viewer.html)
 * exposes `window.composeProofPNG()`, the exact data URL behind its "Save PNG"
 * button, and it auto-loads `./spins.jsonl` from its own origin when the page is
 * served over HTTP without a `?src=` argument. `?T=<temperature>` jumps to the
 * last recorded frame at that temperature.
 *
 * This script therefore serves a folder holding the viewer as `index.html`
 * beside a copy of `week3/spins.jsonl` (and `runs/ramp/run.json`, so the stamp
 * gains "run.json ✓"), opens `index.html?T=<T>` in headless Chrome, polls
 * `composeProofPNG()` until the recording has loaded, and writes the PNG.
 *
 * Setup (the scratch folder is gitignored):
 *   mkdir -p week3/.viewer && cd week3/.viewer && npm --cache /tmp/npmcache i puppeteer-core
 *
 * Run from the week3 folder:
 *   node scripts/capture_viewer.mjs 1.8 evidence/viewer-T1.8.png
 */

import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const WEEK = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const VIEWER_URL = "https://giggleliu.github.io/AMAT5315-2026Fall/week3-viewer.html";
const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";

// puppeteer-core lives in the gitignored scratch folder, not in the repo.
const require = createRequire(path.join(WEEK, ".viewer", "package.json"));
const puppeteer = require("puppeteer-core");

const [, , temperature = "1.8", outArg = `evidence/viewer-T${temperature}.png`] = process.argv;
const out = path.resolve(WEEK, outArg);
const serve = path.resolve(process.env.SERVE_DIR || path.join(WEEK, ".viewer", "serve"));

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".jsonl": "application/x-ndjson",
  ".json": "application/json",
  ".png": "image/png",
};

async function prepare() {
  fs.mkdirSync(serve, { recursive: true });
  const index = path.join(serve, "index.html");
  if (!fs.existsSync(index)) {
    const response = await fetch(VIEWER_URL);
    if (!response.ok) throw new Error(`could not download the viewer: HTTP ${response.status}`);
    fs.writeFileSync(index, await response.text());
    console.log("downloaded the viewer to", index);
  }
  const recording = path.join(serve, "spins.jsonl");
  if (!fs.existsSync(recording)) {
    fs.copyFileSync(path.join(WEEK, "spins.jsonl"), recording);
  }
  const runJson = path.join(serve, "run.json");
  if (!fs.existsSync(runJson)) {
    const ramp = path.join(WEEK, "runs", "ramp", "run.json");
    if (fs.existsSync(ramp)) fs.copyFileSync(ramp, runJson);
  }
}

function startServer() {
  const server = http.createServer((request, response) => {
    const url = new URL(request.url, "http://localhost");
    const relative = path.normalize(decodeURIComponent(url.pathname)).replace(/^(\.\.[/\\])+/, "");
    const file = path.join(serve, relative);
    if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      response.writeHead(404);
      response.end("not found");
      return;
    }
    const body = fs.readFileSync(file);
    response.writeHead(200, {
      "Content-Type": MIME[path.extname(file)] || "application/octet-stream",
      "Content-Length": body.length,
    });
    response.end(body);
  });
  const port = 8731 + Math.floor(Math.random() * 500);
  return new Promise((resolve) =>
    server.listen(port, "127.0.0.1", () => resolve({ server, port })),
  );
}

async function main() {
  await prepare();
  const { server, port } = await startServer();
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: true,
    args: ["--no-sandbox", "--disable-gpu", "--force-device-scale-factor=1"],
  });
  try {
    const page = await browser.newPage();
    page.on("pageerror", (error) => console.log("[pageerror]", error.message));
    await page.goto(`http://127.0.0.1:${port}/index.html?T=${temperature}`, {
      waitUntil: "load",
      timeout: 60_000,
    });

    let dataUrl = null;
    for (let attempt = 0; attempt < 80 && !dataUrl; attempt++) {
      dataUrl = await page.evaluate(() =>
        window.composeProofPNG ? window.composeProofPNG() : null,
      );
      if (!dataUrl) await new Promise((resolve) => setTimeout(resolve, 250));
    }
    console.log("viewer status:", await page.evaluate(() =>
      (document.getElementById("load-status") || {}).textContent,
    ));
    if (!dataUrl) throw new Error("composeProofPNG() never returned an image");

    fs.mkdirSync(path.dirname(out), { recursive: true });
    fs.writeFileSync(out, Buffer.from(dataUrl.split(",")[1], "base64"));
    console.log(`T = ${temperature}: wrote ${out} (${fs.statSync(out).size} bytes)`);
  } finally {
    await browser.close();
    server.close();
  }
}

await main();
