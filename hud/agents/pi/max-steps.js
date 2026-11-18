// Aborts the run before the model call that would start turn HUD_PI_MAX_STEPS + 1.
const limit = Number(process.env.HUD_PI_MAX_STEPS);

export default function (pi) {
  let turns = 0;
  pi.on("turn_end", (event, ctx) => {
    turns += 1;
    if (turns >= limit && event.message.content.some((block) => block.type === "toolCall")) {
      ctx.abort();
    }
  });
}
