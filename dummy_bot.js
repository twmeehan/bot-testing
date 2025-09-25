// minimal-bot.js
const minimist = require("minimist");
const mineflayer = require("mineflayer");
const { pathfinder } = require("mineflayer-pathfinder");
const mineflayerViewerhl = require("prismarine-viewer-colalab").headless;


const args = minimist(process.argv.slice(2), {
  default: {
    host: "127.0.0.1",
    port: 25565,
    receiver_host: "127.0.0.1",
    receiver_port: 8091,
    bot_name: "Alpha",
  },
});

function makeBot({ username, host, port }) {
  const bot = mineflayer.createBot({
    host,
    port,
    username,
    version: "1.21.1",
    checkTimeoutInterval: 10 * 60 * 1000,
  });

  bot.loadPlugin(pathfinder);

  bot.on("end", () => console.log(`[${bot.username}] disconnected.`));
  bot.on("kicked", (reason) =>
    console.log(`[${bot.username}] kicked:`, reason)
  );
  bot.on("error", (err) => console.log(`[${bot.username}] error:`, err));

  return bot;
}

async function main() {
  console.log(`Starting bot: ${args.bot_name}`);

  const bot = makeBot({
    username: args.bot_name,
    host: args.host,
    port: args.port,
  });

  // When the bot spawns, attach the headless viewer
  bot.once("spawn", () => {
    console.log(`[${bot.username}] spawned at`, bot.entity.position);

    mineflayerViewerhl(bot, {
      output: `${args.receiver_host}:${args.receiver_port}`,
      width: 640,
      height: 360,
      frames: 400,
    });

    console.log(
      `[${bot.username}] Headless viewer streaming to ${args.receiver_host}:${args.receiver_port}`
    );
  });

  // Optional: log system chat for debugging
  bot._client.on("packet", (data, meta) => {
    if (meta.name === "system_chat" && data?.content) {
      console.log("SYSTEM:", JSON.stringify(data.content));
    }
  });
}

main().catch(console.error);
