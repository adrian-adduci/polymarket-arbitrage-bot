export * from "./logger";
export * from "./timestamp";

export const sleep = async (ms: number) => {
  await new Promise((resolve, _reject) => {
    setTimeout(() => {
      resolve(true);
    }, ms);
  });
};
