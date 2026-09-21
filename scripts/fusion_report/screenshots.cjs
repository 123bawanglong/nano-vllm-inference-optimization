const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const {chromium} = require('C:/Users/47996/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async () => {
  const assets = path.resolve(__dirname, '../../docs/fusion_experiment_assets');
  const browser = await chromium.launch({channel:'msedge',headless:true,args:['--disable-gpu']});
  try {
    const page = await browser.newPage({viewport:{width:1500,height:1000},deviceScaleFactor:1});
    for (const name of JSON.parse(fs.readFileSync(path.join(assets,'pages.json'),'utf8'))) {
      await page.goto(pathToFileURL(path.join(assets,name+'.html')).href);
      await page.locator('main').screenshot({path:path.join(assets,name+'.png')});
      console.log('SCREENSHOT',name);
    }
  } finally { await browser.close(); }
})();
