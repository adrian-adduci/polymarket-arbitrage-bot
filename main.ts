import dotenv from "dotenv";

dotenv.config();

import { GlobalConfigs } from "./src/configs";
import { FourMemeSniper } from "./src/executors";
import { printBanner } from "./src/utils";

async function main() {
  printBanner();

  const globalConfig = new GlobalConfigs();
  // const loaded = await globalConfig.loadBotAccounts();

  // if (!loaded) await globalConfig.generateBotAccounts(10);

  const fourMemeSniper = new FourMemeSniper(globalConfig);
  await fourMemeSniper.startTokenCreateSniper();
}

main().catch((err) => {
  console.log(err);
});
